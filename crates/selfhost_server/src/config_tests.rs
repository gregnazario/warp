use warp_multi_agent_api::request::settings::ApiKeys;

use super::*;

fn api_keys(anthropic: &str, open_router: &str) -> ApiKeys {
    ApiKeys {
        anthropic: anthropic.to_owned(),
        open_router: open_router.to_owned(),
        ..Default::default()
    }
}

#[test]
fn byok_keys_route_by_default() {
    let config = Config::default();
    assert!(config.byok_direct, "BYOK routing must be on out of the box");

    let resolved = config.resolve_llm("claude-sonnet-4", None, Some(&api_keys("sk-ant", "")));
    assert_eq!(resolved.base_url, "https://api.anthropic.com/v1");
    assert_eq!(resolved.api_key.as_deref(), Some("sk-ant"));
}

#[test]
fn byok_routing_can_be_disabled() {
    let config = Config {
        byok_direct: false,
        ..Default::default()
    };
    let resolved = config.resolve_llm("claude-sonnet-4", None, Some(&api_keys("sk-ant", "")));
    assert_eq!(resolved.base_url, Config::default().llm_base_url);
}

#[test]
fn client_openrouter_key_routes_slash_models() {
    let config = Config::default();
    let keys = api_keys("", "sk-or");
    let resolved = config.resolve_llm("anthropic/claude-sonnet-4", None, Some(&keys));
    assert_eq!(resolved.base_url, "https://openrouter.ai/api/v1");
    assert_eq!(resolved.api_key.as_deref(), Some("sk-or"));
}

#[test]
fn server_openrouter_key_routes_slash_models() {
    let config = Config {
        openrouter_api_key: Some("sk-or".to_owned()),
        ..Default::default()
    };
    let resolved = config.resolve_llm("openai/gpt-4o", None, None);
    assert_eq!(resolved.base_url, "https://openrouter.ai/api/v1");
    assert_eq!(resolved.api_key.as_deref(), Some("sk-or"));
}

#[test]
fn custom_endstants_win_over_byok() {
    // A client-configured custom endpoint (router) takes precedence over
    // every server-side decision.
    use warp_multi_agent_api::request::settings::custom_model_providers::{
        CustomModel, CustomModelProvider,
    };
    let providers = warp_multi_agent_api::request::settings::CustomModelProviders {
        providers: vec![CustomModelProvider {
            base_url: "https://my-router.example/v1".to_owned(),
            api_key: "rk-1".to_owned(),
            models: vec![CustomModel {
                config_key: "cfg-1".to_owned(),
                slug: "some/model".to_owned(),
                ..Default::default()
            }],
            ..Default::default()
        }],
    };
    let config = Config::default();
    let resolved = config.resolve_llm("cfg-1", Some(&providers), Some(&api_keys("sk-ant", "")));
    assert_eq!(resolved.base_url, "https://my-router.example/v1");
    assert_eq!(resolved.model, "some/model");
}

#[test]
fn meta_key_routes_muse_models() {
    let config = Config {
        meta_api_key: Some("meta-key".to_owned()),
        ..Default::default()
    };
    let resolved = config.resolve_llm("muse-spark-1.3", None, None);
    assert_eq!(resolved.base_url, "https://api.meta.ai/v1");
    assert_eq!(resolved.api_key.as_deref(), Some("meta-key"));
}

#[test]
fn foundry_device_grant_routes_without_a_principal() {
    let config = Config {
        provider: Some(crate::config::ProviderKind::AzureFoundry),
        azure_foundry_url: Some("https://my-resource.services.ai.azure.com/models".to_owned()),
        foundry_oauth: Some(crate::config::DynamicAuth::AzureDeviceCode {
            token_url: "https://login.microsoftonline.com/organizations/oauth2/v2.0/token"
                .to_owned(),
            client_id: "cli-client".to_owned(),
            scope: "https://cognitiveservices.azure.com/.default".to_owned(),
            refresh_token: "rt".to_owned(),
        }),
        ..Default::default()
    };
    let resolved = config.resolve_llm("gpt-5", None, None);
    assert_eq!(
        resolved.base_url,
        "https://my-resource.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"
    );
    assert!(matches!(
        resolved.dynamic_auth,
        Some(crate::config::DynamicAuth::AzureDeviceCode { .. })
    ));
}
