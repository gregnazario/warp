use super::*;
use crate::config::{Config, LlmSchema, ProviderKind};

async fn serve_models(body: &'static str) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(
            listener,
            axum::Router::new().route(
                "/v1/models",
                axum::routing::get(move || async move {
                    axum::response::Response::builder()
                        .header("content-type", "application/json")
                        .body(body.to_owned())
                        .unwrap()
                }),
            ),
        )
        .await
        .unwrap();
    });
    format!("http://{addr}/v1")
}

#[tokio::test]
async fn fetch_filters_non_chat_models() {
    let base = serve_models(
        r#"{"data":[{"id":"gpt-9"},{"id":"text-embedding-3"},{"id":"gpt-9-mini"},{"id":"whisper-1"}]}"#,
    )
    .await;
    let endpoint = CatalogEndpoint {
        base_url: "https://api.openai.com/v1",
        schema: LlmSchema::Openai,
        chat_prefixes: &["gpt", "o3"],
    };
    // Point the static base at the mock via the URL we control in the test:
    // fetch_models builds {base}/models, so use the mock base directly.
    let mut endpoint = endpoint;
    endpoint.base_url = Box::leak(base.into_boxed_str());
    let models = fetch_models(&reqwest::Client::new(), &endpoint, "sk").await;
    assert_eq!(models, vec!["gpt-9".to_owned(), "gpt-9-mini".to_owned()]);
}

#[tokio::test]
async fn fetch_returns_empty_on_error() {
    let endpoint = CatalogEndpoint {
        base_url: "http://127.0.0.1:1/v1",
        schema: LlmSchema::Openai,
        chat_prefixes: &[],
    };
    assert!(
        fetch_models(&reqwest::Client::new(), &endpoint, "sk")
            .await
            .is_empty()
    );
}

#[test]
fn curated_providers_have_endpoints() {
    for kind in [
        ProviderKind::Openai,
        ProviderKind::Anthropic,
        ProviderKind::Google,
        ProviderKind::Xai,
        ProviderKind::Zai,
        ProviderKind::Opencode,
    ] {
        assert!(endpoint_for(&kind).is_some(), "{kind:?} needs an endpoint");
    }
    for kind in [
        ProviderKind::OpenRouter,
        ProviderKind::Chatgpt,
        ProviderKind::AzureFoundry,
        ProviderKind::Vertex,
    ] {
        assert!(endpoint_for(&kind).is_none());
    }
}

#[test]
fn catalog_models_route_to_their_provider() {
    let config = Config {
        openai_api_key: Some("sk".to_owned()),
        provider_catalog: std::collections::HashMap::from([(
            "openai".to_owned(),
            vec!["gpt-9".to_owned()],
        )]),
        ..Default::default()
    };
    let resolved = config.resolve_llm("gpt-9", None, None);
    assert_eq!(resolved.base_url, "https://api.openai.com/v1");
    assert_eq!(resolved.api_key.as_deref(), Some("sk"));
}

#[test]
fn catalog_models_need_the_provider_key() {
    let config = Config {
        provider_catalog: std::collections::HashMap::from([(
            "openai".to_owned(),
            vec!["gpt-9".to_owned()],
        )]),
        ..Default::default()
    };
    // No key: falls through to the local backend instead of sending an
    // unauthenticated request upstream.
    let resolved = config.resolve_llm("gpt-9", None, None);
    assert_eq!(resolved.base_url, config.llm_base_url);
}

#[test]
fn opencode_catalog_covers_ambiguous_model_names() {
    // glm-5.2 exists on both Z.ai and OpenCode; whichever catalog was fetched
    // with a key claims it.
    let config = Config {
        opencode_api_key: Some("oc".to_owned()),
        provider_catalog: std::collections::HashMap::from([(
            "opencode".to_owned(),
            vec!["glm-5.2".to_owned(), "kimi-k2.7-code".to_owned()],
        )]),
        ..Default::default()
    };
    let resolved = config.resolve_llm("glm-5.2", None, None);
    assert_eq!(resolved.base_url, "https://opencode.ai/zen/v1");
    assert_eq!(resolved.api_key.as_deref(), Some("oc"));
}
