use std::net::SocketAddr;

use anyhow::{Context as _, Result};
use clap::Parser;
use selfhost_server::config::{LlmBackend, ProviderKind};
use selfhost_server::{Config, detect};

/// Self-hosted Warp agent backend.
///
/// Serves the subset of Warp's server API that Agent Mode needs, translating
/// between Warp's multi-agent wire protocol and an OpenAI- or
/// Anthropic-compatible LLM endpoint. Point a Warp client at it with
/// `WARP_SELF_HOSTED_SERVER_URL` and agent traffic never reaches
/// Warp-operated servers.
#[derive(Debug, Parser)]
#[command(name = "selfhost-server")]
struct Args {
    /// Persistent config file. Flags and SELFHOST_* environment variables
    /// override it; defaults to the file beside the UI's data
    /// (~/Library/Application Support/dev.parw.PRAW/server.toml on macOS).
    #[arg(long)]
    config: Option<String>,

    /// Print the config path in use and whether a file was found, then exit.
    #[arg(long, default_value_t = false)]
    print_config: bool,

    /// Address to bind the server to.
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,

    /// Base URL of the OpenAI-compatible LLM endpoint. The server appends
    /// `/chat/completions` (OpenAI schema) or `/messages` (Anthropic schema).
    /// When unset, the backend is autodetected (Ollama, LM Studio, MLX-LM).
    #[arg(long)]
    llm_base_url: Option<String>,

    /// Which local backend to autodetect: `auto`, `ollama`, `lmstudio`, or
    /// `mlx`. Only used when `--llm-base-url` is unset.
    #[arg(long, default_value = "auto")]
    llm_backend: String,

    /// API key sent to the LLM endpoint, if it requires one.
    #[arg(long, env = "SELFHOST_LLM_API_KEY")]
    llm_api_key: Option<String>,

    /// LLM schema spoken by the endpoint: `openai` or `anthropic`.
    #[arg(long, default_value = "openai")]
    llm_schema: String,

    /// Model name to request from the LLM endpoint. When unset, the model ID
    /// selected in the Warp client is passed through unchanged.
    #[arg(long)]
    llm_model: Option<String>,

    /// If set, Warp clients must present this value as their bearer token
    /// (`WARP_API_KEY`).
    #[arg(long, env = "SELFHOST_API_KEY")]
    api_key: Option<String>,

    /// Path to a file whose contents replace the default agent system prompt.
    #[arg(long)]
    system_prompt_file: Option<String>,

    /// Base URL of an OpenAI-compatible speech-to-text endpoint;
    /// `/audio/transcriptions` is appended. Voice transcription is disabled
    /// when unset.
    #[arg(long, env = "SELFHOST_TRANSCRIBE_BASE_URL")]
    transcribe_base_url: Option<String>,

    /// API key sent to the speech-to-text endpoint, if it requires one.
    #[arg(long, env = "SELFHOST_TRANSCRIBE_API_KEY")]
    transcribe_api_key: Option<String>,

    /// Speech-to-text model to request (e.g. `whisper-1`, `base.en`).
    #[arg(long, default_value = "whisper-1")]
    transcribe_model: String,

    /// Comma-separated model catalog exposed to the Warp client's model picker
    /// (e.g. `--llm-models qwen3-coder:30b,qwen3:8b`). Selected IDs are passed
    /// through to the LLM endpoint.
    #[arg(long)]
    llm_models: Option<String>,

    /// Context-window budget in estimated tokens at which older history is
    /// compacted into a summary. 0 (the default) auto-detects: the client's
    /// per-model limit when set, else 96k.
    #[arg(long, default_value_t = 0)]
    context_window_tokens: u64,

    /// SearXNG-compatible search endpoint enabling the server-side web-search
    /// tool (e.g. http://127.0.0.1:8888/search).
    #[arg(long, env = "SELFHOST_WEB_SEARCH_URL")]
    web_search_url: Option<String>,

    /// Run the ChatGPT (Codex) OAuth login flow now, save the tokens, and
    /// exit. Required once before `--provider chatgpt`.
    #[arg(long, default_value_t = false)]
    codex_login: bool,
    /// Run the Azure Foundry device-login flow now (work account), save the
    /// refresh grant, and exit. Required once before `--provider
    /// azure-foundry` unless a service principal is configured; still needs
    /// `--azure-foundry-url`.
    #[arg(long, default_value_t = false)]
    foundry_login: bool,
    /// Where the Foundry device-login grant is cached (default
    /// ~/.cache/selfhost-server/foundry-tokens.json).
    #[arg(long)]
    foundry_token_file: Option<String>,
    /// Run the Google device-login flow now and write gcloud-style
    /// application-default credentials, then exit. Required once before
    /// `--provider vertex` (unless you already ran
    /// `gcloud auth application-default login`).
    #[arg(long, default_value_t = false)]
    vertex_login: bool,
    /// Where Codex OAuth tokens are cached (default
    /// ~/.cache/selfhost-server/codex-tokens.json).
    #[arg(long)]
    codex_token_file: Option<String>,

    /// Route requests carrying the client's BYOK API keys directly to the
    /// provider named by the key (Anthropic/OpenAI/Google/OpenRouter/Grok
    /// OAuth) based on the requested model's name. On by default; disable
    /// with `--byok-direct false`.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    byok_direct: bool,

    /// Validate every client bearer token by POSTing it as JSON
    /// `{"token": ...}` to this URL and requiring a 2xx response. Lets an
    /// external auth service (SSO, an auth gateway, Foundry-style brokers)
    /// own authentication instead of `--api-key`.
    #[arg(long, env = "SELFHOST_AUTH_INTROSPECT_URL")]
    auth_introspect_url: Option<String>,

    /// Force all non-custom-endpoint traffic to this provider: openai,
    /// anthropic, google, openrouter, xai, zai, opencode, azure-foundry, or
    /// vertex.
    #[arg(long)]
    provider: Option<String>,

    /// API key for OpenAI routing (also `--provider openai`).
    #[arg(long, env = "SELFHOST_OPENAI_API_KEY")]
    openai_api_key: Option<String>,
    #[arg(long, env = "SELFHOST_ANTHROPIC_API_KEY")]
    anthropic_api_key: Option<String>,
    #[arg(long, env = "SELFHOST_GOOGLE_API_KEY")]
    google_api_key: Option<String>,
    /// API key for xAI routing (`--provider xai`, or grok* models).
    #[arg(long, env = "SELFHOST_XAI_API_KEY")]
    xai_api_key: Option<String>,
    /// Z.ai coding-plan API key (https://api.z.ai/api/coding/paas/v4).
    #[arg(long, env = "SELFHOST_ZAI_API_KEY")]
    zai_api_key: Option<String>,
    /// OpenCode Zen API key (https://opencode.ai/zen/v1).
    #[arg(long, env = "SELFHOST_OPENCODE_API_KEY")]
    opencode_api_key: Option<String>,
    /// OpenRouter API key (https://openrouter.ai/api/v1); routes any model
    /// id containing `/`.
    #[arg(long, env = "SELFHOST_OPENROUTER_API_KEY")]
    openrouter_api_key: Option<String>,
    /// Meta Model API key (https://api.meta.ai/v1); serves the Muse Spark
    /// family (`muse-spark-*`).
    #[arg(long, env = "SELFHOST_META_API_KEY")]
    meta_api_key: Option<String>,

    /// Azure tenant for Azure AI Foundry Entra-ID authentication.
    #[arg(long)]
    azure_tenant: Option<String>,
    #[arg(long)]
    azure_client_id: Option<String>,
    #[arg(long)]
    azure_client_secret: Option<String>,
    /// Foundry endpoint prefix, e.g.
    /// `https://<resource>.services.ai.azure.com/models`.
    #[arg(long)]
    azure_foundry_url: Option<String>,
    #[arg(long, default_value = "2024-05-01-preview")]
    azure_api_version: String,
    /// Enable Google Vertex AI routing via gcloud application-default
    /// credentials (~/.config/gcloud/application_default_credentials.json).
    #[arg(long, default_value_t = false)]
    vertex: bool,
    #[arg(
        long,
        default_value = "https://aiplatform.googleapis.com/v1beta1/publishers/google/models"
    )]
    vertex_base_url: String,
    #[arg(long)]
    vertex_adc_path: Option<String>,

    /// Print diagnostics (local backends, configured providers, port) and
    /// exit without serving.
    #[arg(long, default_value_t = false)]
    doctor: bool,
}

impl Args {
    async fn into_config(mut self) -> Result<Config> {
        let system_prompt = match &self.system_prompt_file {
            Some(path) => Some(
                std::fs::read_to_string(path)
                    .with_context(|| format!("failed to read system prompt file {path}"))?,
            ),
            None => None,
        };

        // ChatGPT provider traffic never touches a local backend.
        let chatgpt_provider = self
            .provider
            .as_deref()
            .is_some_and(|name| matches!(name.parse::<ProviderKind>(), Ok(ProviderKind::Chatgpt)));

        // Local backend: an explicit base URL wins; otherwise autodetect
        // among Ollama / LM Studio / MLX-LM (or probe only the requested one).
        let mut detected_models = Vec::new();
        if chatgpt_provider && self.llm_base_url.is_none() {
            self.llm_base_url = Some("https://chatgpt.com/backend-api/codex".to_owned());
        }
        if self.llm_base_url.is_none() {
            let backend: LlmBackend = self.llm_backend.parse()?;
            let http = reqwest::Client::new();
            let targets = match backend {
                LlmBackend::Auto => detect::default_targets(),
                LlmBackend::Ollama => detect::default_targets()
                    .into_iter()
                    .filter(|target| target.backend == LlmBackend::Ollama)
                    .collect(),
                LlmBackend::LmStudio => detect::default_targets()
                    .into_iter()
                    .filter(|target| target.backend == LlmBackend::LmStudio)
                    .collect(),
                LlmBackend::Mlx => detect::default_targets()
                    .into_iter()
                    .filter(|target| target.backend == LlmBackend::Mlx)
                    .collect(),
            };
            match detect::detect(&http, &targets).await {
                Some(detected) => {
                    tracing::info!(
                        "autodetected {} at {} with {} model(s)",
                        detected.backend,
                        detected.base_url,
                        detected.models.len()
                    );
                    self.llm_base_url =
                        Some(format!("{}/v1", detected.base_url.trim_end_matches('/')));
                    detected_models = detected.models;
                }
                None => {
                    tracing::warn!(
                        "no local LLM backend detected (tried {}); falling back to the Ollama default",
                        targets
                            .iter()
                            .map(|target| target.base_url.to_owned())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                    self.llm_base_url = Some("http://127.0.0.1:11434/v1".to_owned());
                }
            }
        }

        let provider = match &self.provider {
            Some(name) => Some(name.parse::<ProviderKind>()?),
            None => None,
        };

        if self.vertex_login {
            let http = reqwest::Client::new();
            let flow = selfhost_server::device_login::gcloud_flow();
            let adc_path = self
                .vertex_adc_path
                .clone()
                .unwrap_or_else(selfhost_server::config::default_adc_path);
            let grant = selfhost_server::device_login::run(&http, &flow, |url, user_code| {
                println!("Visit {url} and enter code {user_code}");
                let _ = command::blocking::Command::new("open").arg(url).spawn();
            })
            .await?;
            let refresh_token = grant
                .refresh_token
                .context("Google did not return a refresh token; re-run --vertex-login")?;
            std::fs::create_dir_all(
                std::path::Path::new(&adc_path)
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new(".")),
            )
            .ok();
            std::fs::write(
                &adc_path,
                serde_json::json!({
                    "type": "authorized_user",
                    "client_id": selfhost_server::device_login::GCLOUD_CLIENT_ID,
                    "client_secret": selfhost_server::device_login::GCLOUD_CLIENT_SECRET,
                    "refresh_token": refresh_token,
                    "token_uri": "https://oauth2.googleapis.com/token",
                })
                .to_string(),
            )
            .with_context(|| format!("failed to write {adc_path}"))?;
            println!("Application-default credentials saved to {adc_path}.");
            std::process::exit(0);
        }

        if self.foundry_login {
            let foundry_url = self
                .azure_foundry_url
                .clone()
                .context("--foundry-login also needs --azure-foundry-url")?;
            let http = reqwest::Client::new();
            let flow = selfhost_server::device_login::azure_foundry_flow();
            let grant = selfhost_server::device_login::run(&http, &flow, |url, user_code| {
                println!("Visit {url} and enter code {user_code}");
                let _ = command::blocking::Command::new("open").arg(url).spawn();
            })
            .await?;
            let refresh_token = grant
                .refresh_token
                .context("Entra ID did not return a refresh token; re-run --foundry-login")?;
            let token_file = self
                .foundry_token_file
                .clone()
                .unwrap_or_else(default_foundry_token_file);
            std::fs::create_dir_all(
                std::path::Path::new(&token_file)
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new(".")),
            )
            .ok();
            std::fs::write(
                &token_file,
                serde_json::json!({
                    "token_url": flow.token_url,
                    "client_id": flow.client_id,
                    "scope": flow.scope,
                    "refresh_token": refresh_token,
                    "foundry_url": foundry_url,
                })
                .to_string(),
            )
            .with_context(|| format!("failed to write {token_file}"))?;
            println!("Foundry grant saved to {token_file}.");
            std::process::exit(0);
        }

        let codex_token_file = self
            .codex_token_file
            .unwrap_or_else(selfhost_server::codex::default_token_file);
        if self.codex_login {
            let http = reqwest::Client::new();
            let tokens = selfhost_server::codex::login(&http, &codex_token_file, |url| {
                println!("Opening browser (if it does not open, visit the URL)...");
                let opener = if cfg!(target_os = "macos") {
                    "open"
                } else if cfg!(target_os = "windows") {
                    "cmd"
                } else {
                    "xdg-open"
                };
                let mut command = command::blocking::Command::new(opener);
                if cfg!(target_os = "windows") {
                    command.arg("/c").arg("start");
                }
                command.arg(url).spawn().ok();
            })
            .await?;
            println!(
                "Codex tokens saved to {codex_token_file} (account: {}).",
                tokens.account_id.as_deref().unwrap_or("unknown")
            );
            std::process::exit(0);
        }
        let mut llm_models = match self.llm_models {
            Some(models) => models
                .split(',')
                .map(str::trim)
                .filter(|model| !model.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
            None => std::mem::take(&mut detected_models),
        };
        if provider == Some(ProviderKind::Chatgpt) && llm_models.is_empty() {
            llm_models = vec!["gpt-5-codex".to_owned(), "gpt-5".to_owned()];
        }
        // Native key routing: when a provider key is present, its models
        // belong in the picker so they can be selected.
        if self.zai_api_key.is_some() {
            for model in ["glm-4.7", "glm-4.6", "glm-4.5-air"] {
                if !llm_models.iter().any(|existing| existing == model) {
                    llm_models.push(model.to_owned());
                }
            }
        }
        if self.opencode_api_key.is_some() {
            for model in ["grok-code", "code-supernova", "kimi-k2", "qwen3-coder"] {
                if !llm_models.iter().any(|existing| existing == model) {
                    llm_models.push(model.to_owned());
                }
            }
        }
        if self.meta_api_key.is_some() {
            for model in ["muse-spark-1.3", "muse-spark-1.2", "muse-spark-1.1"] {
                if !llm_models.iter().any(|existing| existing == model) {
                    llm_models.push(model.to_owned());
                }
            }
        }

        // Live provider catalogs: fetch each keyed provider's model list so
        // its models appear in the picker and route correctly. Best-effort;
        // static fallbacks (and prefix routing) still apply on failure.
        let mut provider_catalog = std::collections::HashMap::new();
        {
            let http = reqwest::Client::new();
            let mut keyed: Vec<(ProviderKind, String)> = Vec::new();
            for (kind, key) in [
                (ProviderKind::Openai, &self.openai_api_key),
                (ProviderKind::Anthropic, &self.anthropic_api_key),
                (ProviderKind::Google, &self.google_api_key),
                (ProviderKind::Xai, &self.xai_api_key),
                (ProviderKind::Zai, &self.zai_api_key),
                (ProviderKind::Opencode, &self.opencode_api_key),
                (ProviderKind::Meta, &self.meta_api_key),
            ] {
                if let Some(key) = key.as_deref().filter(|key| !key.trim().is_empty()) {
                    keyed.push((kind, key.trim().to_owned()));
                }
            }
            for (kind, key) in keyed {
                let Some(endpoint) = selfhost_server::provider_catalog::endpoint_for(&kind) else {
                    continue;
                };
                let models =
                    selfhost_server::provider_catalog::fetch_models(&http, &endpoint, &key).await;
                if models.is_empty() {
                    tracing::warn!("no model catalog from {kind:?}; static fallbacks apply");
                    continue;
                }
                tracing::info!("{kind:?} catalog: {} model(s)", models.len());
                provider_catalog.insert(format!("{kind:?}").to_lowercase(), models);
            }
        }

        // Fetched provider models join the picker after locally detected
        // models; the default stays the local backend's first model.
        for models in provider_catalog.values() {
            for model in models {
                if !llm_models.contains(model) {
                    llm_models.push(model.clone());
                }
            }
        }

        Ok(Config {
            provider_catalog,
            llm_base_url: self.llm_base_url.expect("set above"),
            llm_api_key: self.llm_api_key,
            llm_schema: self.llm_schema.parse()?,
            llm_models,
            llm_model: self.llm_model,
            api_key: self.api_key,
            system_prompt,
            transcribe_base_url: self.transcribe_base_url.filter(|url| !url.is_empty()),
            transcribe_api_key: self.transcribe_api_key,
            transcribe_model: self.transcribe_model,

            context_window_tokens: self.context_window_tokens,
            web_search_url: self.web_search_url.filter(|url| !url.is_empty()),
            provider,
            openai_api_key: self.openai_api_key.filter(|key| !key.is_empty()),
            anthropic_api_key: self.anthropic_api_key.filter(|key| !key.is_empty()),
            google_api_key: self.google_api_key.filter(|key| !key.is_empty()),
            xai_api_key: self.xai_api_key.filter(|key| !key.is_empty()),
            zai_api_key: self.zai_api_key.filter(|key| !key.is_empty()),
            opencode_api_key: self.opencode_api_key.filter(|key| !key.is_empty()),
            openrouter_api_key: self.openrouter_api_key.filter(|key| !key.is_empty()),
            meta_api_key: self.meta_api_key.filter(|key| !key.is_empty()),
            foundry_oauth: load_foundry_oauth(
                &self.foundry_token_file,
                self.azure_client_secret.is_none(),
            ),
            azure_tenant: self.azure_tenant,
            azure_client_id: self.azure_client_id,
            azure_client_secret: self.azure_client_secret,
            azure_foundry_url: self
                .azure_foundry_url
                .clone()
                .filter(|url| !url.is_empty())
                .or_else(|| foundry_file_url(&self.foundry_token_file)),
            azure_api_version: self.azure_api_version,
            vertex_enabled: self.vertex,
            vertex_base_url: self.vertex_base_url,
            vertex_adc_path: self.vertex_adc_path,
            codex_token_file,
            byok_direct: self.byok_direct,
            auth_introspect_url: self.auth_introspect_url.filter(|url| !url.is_empty()),
        })
    }
}

fn default_foundry_token_file() -> String {
    format!(
        "{}/.cache/selfhost-server/foundry-tokens.json",
        std::env::var("HOME").unwrap_or_default()
    )
}

/// The `--azure-foundry-url` saved by `--foundry-login`, so later runs do
/// not need to repeat the flag.
fn foundry_file_url(token_file: &Option<String>) -> Option<String> {
    let path = token_file
        .clone()
        .unwrap_or_else(default_foundry_token_file);
    let file: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    file["foundry_url"].as_str().map(ToOwned::to_owned)
}

/// Loads the `--foundry-login` refresh grant. A service principal (when
/// configured) always wins; the grant is only used without one.
fn load_foundry_oauth(
    token_file: &Option<String>,
    no_principal: bool,
) -> Option<selfhost_server::config::DynamicAuth> {
    if !no_principal {
        return None;
    }
    let path = token_file
        .clone()
        .unwrap_or_else(default_foundry_token_file);
    let file = std::fs::read_to_string(&path).ok()?;
    let file: serde_json::Value = serde_json::from_str(&file).ok()?;
    Some(selfhost_server::config::DynamicAuth::AzureDeviceCode {
        token_url: file["token_url"].as_str()?.to_owned(),
        client_id: file["client_id"].as_str()?.to_owned(),
        scope: file["scope"].as_str()?.to_owned(),
        refresh_token: file["refresh_token"].as_str()?.to_owned(),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "selfhost_server=info".into()),
        )
        .init();

    // The persistent config sits underneath flags and env: expand it into
    // argv entries for anything neither provides.
    let config_path = std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .find_map(|pair| {
            pair[0]
                .strip_prefix("--config=")
                .map(ToOwned::to_owned)
                .or((pair[0] == "--config").then(|| pair[1].clone()))
        })
        .or_else(|| std::env::var(selfhost_server::config_file::CONFIG_PATH_ENV).ok())
        .unwrap_or_else(selfhost_server::config_file::default_path);
    let argv = std::env::args().collect::<Vec<_>>();
    let expanded = selfhost_server::config_file::expand(&config_path, &argv)
        .with_context(|| format!("invalid config file {config_path}"))?;
    let file_found = std::path::Path::new(&config_path).is_file();
    let args = Args::parse_from(expanded);
    if args.print_config {
        println!(
            "{} ({})",
            config_path,
            if file_found { "found" } else { "not present" }
        );
        return Ok(());
    }
    tracing::info!(path = %config_path, found = file_found, "server config");
    if args.doctor {
        doctor(&args, &config_path, file_found).await;
        return Ok(());
    }
    let bind_addr: SocketAddr = args
        .bind
        .parse()
        .context("failed to parse --bind address")?;
    let config = args.into_config().await?;
    let app = selfhost_server::router(config);

    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .with_context(|| format!("failed to bind {bind_addr}"))?;
    tracing::info!("self-hosted agent server listening on http://{bind_addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

/// One-stop diagnostics for "why doesn't agent mode work": which local LLM
/// backends answer, which provider integrations are configured, and whether
/// the server can take its port. Informational only — always exits 0.
async fn doctor(args: &Args, config_path: &str, file_found: bool) {
    println!("PRAW backend doctor (v{})", env!("CARGO_PKG_VERSION"));
    let config_path = config_path.to_owned();

    println!("\nLocal LLM backends:");
    let http = reqwest::Client::new();
    for target in detect::default_targets() {
        match detect::probe(&http, &target).await {
            Ok(found) => {
                let models = if found.models.is_empty() {
                    String::from("no models")
                } else {
                    format!(
                        "{} model(s): {}",
                        found.models.len(),
                        found.models.join(", ")
                    )
                };
                println!("  ok  {} at {} — {models}", found.backend, found.base_url);
            }
            Err(_) => {
                println!(
                    "  --  {} at {} — not responding",
                    target.backend, target.base_url
                );
            }
        }
    }
    if args.llm_base_url.is_some() {
        println!("  ok  custom endpoint via --llm-base-url (autodetect skipped)");
    }

    println!("\nProviders:");
    if let Some(provider) = &args.provider {
        println!("  ok  --provider {provider} (forced)");
    } else {
        println!("  --  --provider not forced; BYOK model-name routing decides per request");
    }
    for (name, key) in [
        ("openai", &args.openai_api_key),
        ("anthropic", &args.anthropic_api_key),
        ("google", &args.google_api_key),
        ("xai", &args.xai_api_key),
        ("zai", &args.zai_api_key),
        ("opencode", &args.opencode_api_key),
        ("openrouter", &args.openrouter_api_key),
        ("meta", &args.meta_api_key),
    ] {
        match key.as_deref().filter(|key| !key.is_empty()) {
            Some(key) => println!("  ok  {name} key set ({})", mask_key(key)),
            None => println!("  --  {name} key not set"),
        }
    }
    let azure_ready = args.azure_tenant.is_some()
        && args.azure_client_id.is_some()
        && args.azure_client_secret.is_some()
        && args.azure_foundry_url.is_some();
    println!(
        "  {}  azure-foundry {}",
        if azure_ready { "ok" } else { "--" },
        if azure_ready {
            format!(
                "tenant {} configured",
                args.azure_tenant.as_deref().unwrap_or_default()
            )
        } else {
            String::from(
                "needs --azure-tenant/--azure-client-id/--azure-client-secret/--azure-foundry-url",
            )
        }
    );
    if args.vertex {
        let adc = args
            .vertex_adc_path
            .clone()
            .unwrap_or_else(selfhost_server::config::default_adc_path);
        let exists = std::path::Path::new(&adc).is_file();
        println!(
            "  {}  vertex (application-default credentials at {adc})",
            if exists { "ok" } else { "!!" }
        );
    } else {
        println!("  --  vertex disabled (pass --vertex to enable)");
    }
    let codex_tokens = selfhost_server::codex::default_token_file();
    println!(
        "  {}  chatgpt/codex tokens at {codex_tokens}",
        if std::path::Path::new(&codex_tokens).is_file() {
            "ok"
        } else {
            "--"
        }
    );

    println!("\nServer:");
    println!(
        "  {}  config {}",
        if file_found { "ok" } else { "--" },
        config_path
    );
    println!("  --  bind address {}", args.bind);
    let bind_check: SocketAddr = args
        .bind
        .parse()
        .unwrap_or_else(|_| "127.0.0.1:8080".parse().expect("valid socket address"));
    match std::net::TcpListener::bind(bind_check) {
        Ok(_listener) => println!("  ok  port is free"),
        Err(error) => println!("  !!  port unavailable: {error}"),
    }
    println!(
        "  {}  client auth: {}",
        if args.api_key.is_some() { "ok" } else { "--" },
        if args.api_key.is_some() {
            String::from("--api-key required from clients")
        } else {
            String::from("open — any client token accepted (fine on loopback)")
        }
    );
    println!(
        "  {}  web search: {}",
        if args.web_search_url.is_some() {
            "ok"
        } else {
            "--"
        },
        args.web_search_url
            .as_deref()
            .unwrap_or("not configured (--web-search-url)")
    );
    println!(
        "  {}  transcription: {}",
        if args.transcribe_base_url.is_some() {
            "ok"
        } else {
            "--"
        },
        args.transcribe_base_url
            .as_deref()
            .unwrap_or("not configured (--transcribe-base-url)")
    );
}

/// Enough of a key to recognize it, not enough to leak it.
fn mask_key(key: &str) -> String {
    if key.len() <= 8 {
        String::from("…")
    } else {
        format!("…{}", &key[key.len() - 4..])
    }
}
