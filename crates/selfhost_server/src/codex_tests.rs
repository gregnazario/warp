use super::*;

#[test]
fn pkce_challenge_is_s256_of_verifier() {
    let (verifier, challenge) = pkce_pair();
    assert_eq!(verifier.len(), 64);
    let digest = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    assert_eq!(challenge, digest);
}

#[test]
fn account_id_is_extracted_from_id_token() {
    let claims = serde_json::json!({
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}
    });
    let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap());
    let id_token = format!("header.{payload}.signature");
    assert_eq!(
        account_id_from_id_token(&id_token).as_deref(),
        Some("acct-123")
    );
    assert_eq!(account_id_from_id_token("garbage"), None);
}

#[tokio::test]
async fn token_exchange_parses_reply_and_persists() {
    // The reply parser is exercised directly below; no HTTP stub needed.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let app = axum::Router::new().route(
        "/token",
        axum::routing::post(|| async {
            axum::Json(serde_json::json!({
                "access_token": "at",
                "refresh_token": "rt",
                "id_token": format!(
                    "h.{}.s",
                    URL_SAFE_NO_PAD.encode(serde_json::to_vec(&serde_json::json!({
                        "https://api.openai.com/auth": {"chatgpt_account_id": "acc"}
                    })).unwrap())
                ),
                "expires_in": 3600
            }))
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Point the module's fixed token URL at the stub via a hosts-free trick:
    // exercise the reply parser directly instead of the fixed URL.
    let reply = serde_json::json!({
        "access_token": "at",
        "refresh_token": "rt",
        "id_token": format!(
            "h.{}.s",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&serde_json::json!({
                "https://api.openai.com/auth": {"chatgpt_account_id": "acc"}
            })).unwrap())
        ),
        "expires_in": 3600
    });
    let tokens = tokens_from_reply(&reply.to_string(), "test").unwrap();
    assert_eq!(tokens.access_token, "at");
    assert_eq!(tokens.account_id.as_deref(), Some("acc"));
    assert!(!tokens.expired());

    let file = std::env::temp_dir().join("selfhost-codex-test-tokens.json");
    save_tokens(file.to_str().unwrap(), &tokens).unwrap();
    let loaded = load_tokens(file.to_str().unwrap()).unwrap();
    assert_eq!(loaded.refresh_token, "rt");
    std::fs::remove_file(&file).ok();
}

#[tokio::test]
async fn load_or_refresh_requires_a_prior_login() {
    let file = std::env::temp_dir().join("selfhost-codex-no-such-tokens.json");
    let error = load_or_refresh(&reqwest::Client::new(), file.to_str().unwrap())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("--codex-login"), "{error}");
}
