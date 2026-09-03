//! ChatGPT/Codex OAuth ("Sign in with ChatGPT") and token maintenance.
//!
//! Mirrors the Codex CLI flow: a PKCE authorization through
//! `auth.openai.com` with a localhost callback, a token exchange, and later
//! refreshes. The resulting access token authenticates against the Codex
//! backend (`chatgpt.com/backend-api/codex`), which speaks the OpenAI
//! Responses API rather than chat completions.

use anyhow::{Context as _, Result};
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const OPENAI_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const AUTHORIZE_URL: &str = "https://auth.openai.com/authorize";
pub const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const CALLBACK_PORT: u16 = 1455;

/// Default cache location for Codex OAuth tokens.
pub fn default_token_file() -> String {
    format!(
        "{}/.cache/selfhost-server/codex-tokens.json",
        std::env::var("HOME").unwrap_or_default()
    )
}

/// ChatGPT OAuth tokens, persisted to disk between server runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexTokens {
    pub access_token: String,
    pub refresh_token: String,
    /// ChatGPT account id, required as a header on Codex backend calls.
    #[serde(default)]
    pub account_id: Option<String>,
    /// Unix seconds after which the access token needs refreshing.
    pub expires_at: u64,
}

impl CodexTokens {
    pub fn expired(&self) -> bool {
        now_secs() + 60 >= self.expires_at
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

/// PKCE verifier + S256 challenge.
pub fn pkce_pair() -> (String, String) {
    let verifier: String = uuid::Uuid::new_v4()
        .simple()
        .to_string()
        .chars()
        .chain(uuid::Uuid::new_v4().simple().to_string().chars())
        .take(64)
        .collect();
    let digest = Sha256::digest(verifier.as_bytes());
    let challenge = URL_SAFE_NO_PAD.encode(digest);
    (verifier, challenge)
}

/// Extracts the ChatGPT account id from the `id_token` JWT.
fn account_id_from_id_token(id_token: &str) -> Option<String> {
    let payload_b64 = id_token.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD.decode(payload_b64).ok().or_else(|| {
        base64::engine::general_purpose::STANDARD
            .decode(payload_b64)
            .ok()
    })?;
    let claims: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    claims["https://api.openai.com/auth"]["chatgpt_account_id"]
        .as_str()
        .map(ToOwned::to_owned)
}

fn save_tokens(path: &str, tokens: &CodexTokens) -> Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(path, serde_json::to_string(tokens)?)?;
    Ok(())
}

pub fn load_tokens(path: &str) -> Option<CodexTokens> {
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Runs the interactive login: prints the authorize URL, waits for the
/// localhost callback, exchanges the code, and persists the tokens.
pub async fn login(
    http: &reqwest::Client,
    token_file: &str,
    open_browser: impl FnOnce(String),
) -> Result<CodexTokens> {
    let (verifier, challenge) = pkce_pair();
    let state = uuid::Uuid::new_v4().simple().to_string();
    let authorize_url = format!(
        "{AUTHORIZE_URL}?response_type=code&client_id={OPENAI_CLIENT_ID}\
         &redirect_uri=http%3A%2F%2Flocalhost%3A{CALLBACK_PORT}%2Fauth%2Fcallback\
         &scope=openid%20profile%20email%20offline_access\
         &code_challenge={challenge}&code_challenge_method=S256\
         &state={state}&id_token_add_organizations=true&codex_cli_simplified_flow=true"
    );
    println!("Open this URL to sign in with ChatGPT:\n\n{authorize_url}\n");
    open_browser(authorize_url);

    let (code_tx, code_rx) = tokio::sync::oneshot::channel::<String>();
    let code_tx = std::sync::Arc::new(std::sync::Mutex::new(Some(code_tx)));
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", CALLBACK_PORT))
        .await
        .with_context(|| {
            format!(
                "could not bind the OAuth callback port {CALLBACK_PORT}; is another \
                 login in progress?"
            )
        })?;
    let app = axum::Router::new().route(
        "/auth/callback",
        axum::routing::get(move |raw_query: axum::extract::RawQuery| {
            let code_tx = code_tx.clone();
            async move {
                let mut code = None;
                if let Some(query) = raw_query.0 {
                    for pair in query.split('&') {
                        if let Some(value) = pair.strip_prefix("code=") {
                            code = Some(value.replace("%2F", "/").to_owned());
                        }
                    }
                }
                let body = match code {
                    Some(_) => "<html><body><h2>Login complete.</h2> \
                                <p>You can close this window and return to the terminal.</p></body></html>",
                    None => "<html><body><h2>Login failed:</h2> no authorization code in callback.</body></html>",
                };
                if let Some(code) = code
                    && let Some(sender) = code_tx.lock().ok().and_then(|mut guard| guard.take())
                {
                    let _ = sender.send(code);
                }
                (
                    [(axum::http::header::CONTENT_TYPE, "text/html")],
                    body,
                )
            }
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await });

    let code = tokio::time::timeout(std::time::Duration::from_secs(5 * 60), code_rx)
        .await
        .context("timed out waiting for the OAuth callback")?
        .context("callback server dropped before receiving the authorization code")?;

    let tokens = exchange_code(http, &code, &verifier).await?;
    save_tokens(token_file, &tokens)?;
    Ok(tokens)
}

async fn exchange_code(http: &reqwest::Client, code: &str, verifier: &str) -> Result<CodexTokens> {
    let response = http
        .post(TOKEN_URL)
        .form(&[
            ("grant_type", "authorization_code"),
            ("client_id", OPENAI_CLIENT_ID),
            ("code", code),
            ("code_verifier", verifier),
            (
                "redirect_uri",
                &format!("http://localhost:{CALLBACK_PORT}/auth/callback"),
            ),
        ])
        .send()
        .await
        .context("OpenAI token endpoint unreachable")?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!("OpenAI token endpoint returned {status}: {}", body.trim());
    }
    tokens_from_reply(&body, "token exchange")
}

fn tokens_from_reply(body: &str, stage: &str) -> Result<CodexTokens> {
    #[derive(Deserialize)]
    struct Reply {
        access_token: String,
        #[serde(default)]
        refresh_token: Option<String>,
        #[serde(default)]
        id_token: Option<String>,
        #[serde(default)]
        expires_in: Option<u64>,
    }
    let reply: Reply = serde_json::from_str(body)
        .with_context(|| format!("{stage} reply was not the expected shape"))?;
    let account_id = reply.id_token.as_deref().and_then(account_id_from_id_token);
    Ok(CodexTokens {
        access_token: reply.access_token,
        refresh_token: reply
            .refresh_token
            .context("token reply missing refresh_token")?,
        account_id,
        expires_at: now_secs() + reply.expires_in.unwrap_or(3600),
    })
}

/// Loads cached tokens, refreshing them when expired. Errors when no login
/// has been performed.
pub async fn load_or_refresh(http: &reqwest::Client, token_file: &str) -> Result<CodexTokens> {
    let mut tokens = load_tokens(token_file)
        .context("no Codex tokens cached; run the server once with --codex-login first")?;
    if !tokens.expired() {
        return Ok(tokens);
    }
    tokens = refresh(http, &tokens).await?;
    save_tokens(token_file, &tokens)?;
    Ok(tokens)
}

async fn refresh(http: &reqwest::Client, tokens: &CodexTokens) -> Result<CodexTokens> {
    let response = http
        .post(TOKEN_URL)
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", OPENAI_CLIENT_ID),
            ("refresh_token", tokens.refresh_token.as_str()),
        ])
        .send()
        .await
        .context("OpenAI token endpoint unreachable")?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(
            "OpenAI token refresh returned {status}: {} (run --codex-login again)",
            body.trim()
        );
    }
    tokens_from_reply(&body, "token refresh")
}

#[cfg(test)]
#[path = "codex_tests.rs"]
mod tests;
