use std::sync::{Arc, Mutex};

use warp_multi_agent_api as api;

use super::*;

#[tokio::test]
async fn openai_stream_parses_text_tool_calls_and_usage() {
    // Stand up a stub OpenAI-compatible endpoint.
    let sse_body = "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n\
                    data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n\
                    data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call1\",\"function\":{\"name\":\"run_shell_command\",\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}}]}}]}\n\n\
                    data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
                    data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":4}}\n\n\
                    data: [DONE]\n\n";
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(move || async move {
            (
                [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                sse_body,
            )
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let llm = ResolvedLlm::test_default(format!("http://{addr}/v1"));
    let mut texts = Vec::new();
    let mut tool_calls = Vec::new();
    let usage = stream_completion(
        &reqwest::Client::new(),
        &llm,
        "system",
        &[LlmMessage::User {
            text: "hi".to_owned(),
            images: Vec::new(),
        }],
        &[],
        |delta| {
            match delta {
                Delta::Text(text) => texts.push(text),
                Delta::ToolCall {
                    id,
                    name,
                    arguments,
                } => tool_calls.push((id, name, arguments)),
            }
            Ok(())
        },
    )
    .await
    .unwrap();

    assert_eq!(texts.join(""), "Hello");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].0.as_deref(), Some("call1"));
    assert_eq!(tool_calls[0].1.as_deref(), Some("run_shell_command"));
    assert_eq!(tool_calls[0].2, "{\"command\":\"ls\"}");
    assert_eq!(usage.input_tokens, 11);
    assert_eq!(usage.output_tokens, 4);
}

#[tokio::test]
async fn provider_errors_are_surfaced() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(|| async { (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let llm = ResolvedLlm::test_default(format!("http://{addr}/v1"));
    let result = stream_completion(
        &reqwest::Client::new(),
        &llm,
        "system",
        &[LlmMessage::User {
            text: "hi".to_owned(),
            images: Vec::new(),
        }],
        &[],
        |_| Ok(()),
    )
    .await;
    let error = result.unwrap_err().to_string();
    assert!(error.contains("500"), "unexpected error: {error}");
    assert!(error.contains("boom"), "unexpected error: {error}");
}

#[test]
fn config_resolves_custom_endpoint_models() {
    let mut config = crate::config::Config::test_default();
    config.llm_base_url = "http://default:1234/v1".to_owned();
    config.llm_model = Some("forced".to_owned());
    config.provider = None;
    let mut custom = api::request::settings::CustomModelProviders::default();
    custom.providers.push(api::request::settings::custom_model_providers::CustomModelProvider {
        base_url: "http://localhost:8081".to_owned(),
        api_key: "secret".to_owned(),
        schema: api::request::settings::custom_model_providers::CustomEndpointSchema::AnthropicMessages
            as i32,
        models: vec![api::request::settings::custom_model_providers::CustomModel {
            slug: "my-model".to_owned(),
            config_key: "my-config-key".to_owned(),
            reasoning_effort: String::new(),
        }],
    });

    let resolved = config.resolve_llm("my-config-key", Some(&custom), None);
    assert_eq!(resolved.base_url, "http://localhost:8081");
    assert_eq!(resolved.api_key.as_deref(), Some("secret"));
    assert_eq!(resolved.schema, crate::config::LlmSchema::Anthropic);
    assert_eq!(resolved.model, "my-model");

    let resolved = config.resolve_llm("claude-4-5-sonnet", Some(&custom), None);
    assert_eq!(resolved.base_url, "http://default:1234/v1");
    assert_eq!(resolved.model, "forced");
}

#[tokio::test]
async fn openai_user_images_are_sent_as_data_uris() {
    let captured: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_in_handler = captured.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(move |body: String| {
            let captured = captured_in_handler.clone();
            async move {
                if let Ok(value) = serde_json::from_str(&body) {
                    captured.lock().unwrap().push(value);
                }
                (
                    [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                    "data: {\"choices\":[{\"delta\":{}}]}\n\ndata: [DONE]\n\n".to_owned(),
                )
            }
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let llm = ResolvedLlm {
        model: "vision-model".to_owned(),
        ..ResolvedLlm::test_default(format!("http://{addr}/v1"))
    };
    stream_completion(
        &reqwest::Client::new(),
        &llm,
        "system",
        &[LlmMessage::User {
            text: "what is this".to_owned(),
            images: vec![super::super::llm::LlmImage {
                data: vec![1, 2, 3],
                mime_type: "image/png".to_owned(),
            }],
        }],
        &[],
        |_| Ok(()),
    )
    .await
    .unwrap();

    let body = captured.lock().unwrap()[0].clone();
    let content = &body["messages"][1]["content"];
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "image_url");
    let url = content[1]["image_url"]["url"].as_str().unwrap();
    assert!(url.starts_with("data:image/png;base64,"), "{url}");
    // base64([1,2,3]) = AQID
    assert!(url.ends_with("AQID"), "{url}");
}

#[tokio::test]
async fn anthropic_system_prompt_is_cacheable() {
    let captured: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_in_handler = captured.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().route(
        "/v1/messages",
        axum::routing::post(move |body: String| {
            let captured = captured_in_handler.clone();
            async move {
                if let Ok(value) = serde_json::from_str(&body) {
                    captured.lock().unwrap().push(value);
                }
                (
                    [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                    "data: {\"type\":\"message_stop\"}\n\n".to_owned(),
                )
            }
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let llm = ResolvedLlm {
        model: "claude".to_owned(),
        schema: crate::config::LlmSchema::Anthropic,
        ..ResolvedLlm::test_default(format!("http://{addr}/v1"))
    };
    stream_completion(
        &reqwest::Client::new(),
        &llm,
        "the system prompt",
        &[LlmMessage::User {
            text: "hi".to_owned(),
            images: Vec::new(),
        }],
        &[],
        |_| Ok(()),
    )
    .await
    .unwrap();

    let body = captured.lock().unwrap()[0].clone();
    assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    assert_eq!(body["system"][0]["text"], "the system prompt");
}

#[test]
fn byok_direct_routes_by_model_prefix_and_available_key() {
    let mut config = crate::config::Config::test_default();
    config.byok_direct = true;
    config.llm_base_url = "http://default:1234/v1".to_owned();
    let keys = api::request::settings::ApiKeys {
        anthropic: "sk-ant-key".to_owned(),
        openai: "sk-oai-key".to_owned(),
        google: "g-key".to_owned(),
        open_router: "or-key".to_owned(),
        ..Default::default()
    };

    let resolved = config.resolve_llm("claude-sonnet-4-5", None, Some(&keys));
    assert_eq!(resolved.schema, crate::config::LlmSchema::Anthropic);
    assert_eq!(resolved.base_url, "https://api.anthropic.com/v1");
    assert_eq!(resolved.api_key.as_deref(), Some("sk-ant-key"));
    assert_eq!(resolved.model, "claude-sonnet-4-5");

    let resolved = config.resolve_llm("gpt-5.2", None, Some(&keys));
    assert_eq!(resolved.base_url, "https://api.openai.com/v1");
    assert_eq!(resolved.api_key.as_deref(), Some("sk-oai-key"));

    let resolved = config.resolve_llm("gemini-3-pro", None, Some(&keys));
    assert!(resolved.base_url.contains("generativelanguage"));
    assert_eq!(resolved.api_key.as_deref(), Some("g-key"));

    let resolved = config.resolve_llm("meta/llama-4", None, Some(&keys));
    assert_eq!(resolved.base_url, "https://openrouter.ai/api/v1");
    assert_eq!(resolved.api_key.as_deref(), Some("or-key"));

    // Without the flag (or without keys), the server default applies.
    let mut disabled = config.clone();
    disabled.byok_direct = false;
    let resolved = disabled.resolve_llm("claude-sonnet-4-5", None, Some(&keys));
    assert_eq!(resolved.base_url, "http://default:1234/v1");

    let no_keys = api::request::settings::ApiKeys {
        anthropic: "  ".to_owned(),
        ..Default::default()
    };
    let resolved = config.resolve_llm("claude-sonnet-4-5", None, Some(&no_keys));
    assert_eq!(resolved.base_url, "http://default:1234/v1");
}

#[test]
fn explicit_provider_routes_with_keys_and_oauth() {
    use crate::config::{DynamicAuth, ProviderKind};

    // Gateway providers with static keys.
    let mut config = crate::config::Config::test_default();
    config.provider = Some(ProviderKind::Zai);
    config.zai_api_key = Some("zai-key".to_owned());
    let resolved = config.resolve_llm("glm-4.7", None, None);
    assert_eq!(resolved.base_url, "https://api.z.ai/api/coding/paas/v4");
    assert_eq!(resolved.api_key.as_deref(), Some("zai-key"));
    assert_eq!(resolved.model, "glm-4.7");

    config.provider = Some(ProviderKind::Opencode);
    config.opencode_api_key = Some("oc-key".to_owned());
    let resolved = config.resolve_llm("grok-code", None, None);
    assert_eq!(resolved.base_url, "https://opencode.ai/zen/v1");
    assert_eq!(resolved.api_key.as_deref(), Some("oc-key"));

    // Azure AI Foundry resolves Entra client-credentials OAuth.
    config.provider = Some(ProviderKind::AzureFoundry);
    config.azure_tenant = Some("tenant".to_owned());
    config.azure_client_id = Some("id".to_owned());
    config.azure_client_secret = Some("secret".to_owned());
    config.azure_foundry_url = Some("https://res.services.ai.azure.com/models".to_owned());
    let resolved = config.resolve_llm("gpt-5.2", None, None);
    assert_eq!(
        resolved.base_url,
        "https://res.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"
    );
    let azure_auth = match resolved.dynamic_auth {
        Some(DynamicAuth::AzureClientCredentials { client_id, .. }) => Some(client_id),
        _ => None,
    };
    assert_eq!(azure_auth.as_deref(), Some("id"));

    // Vertex requires the flag and an ADC file; without them it falls through.
    config.provider = Some(ProviderKind::Vertex);
    config.vertex_enabled = false;
    let resolved = config.resolve_llm("gemini-3-pro", None, None);
    assert_eq!(resolved.base_url, "");

    // Provider without credentials falls through to the local backend.
    config.provider = Some(ProviderKind::Openai);
    config.openai_api_key = None;
    let resolved = config.resolve_llm("gpt-5.2", None, None);
    assert_eq!(resolved.base_url, "");
}

#[test]
fn grok_oauth_token_from_client_routes_to_xai() {
    let mut config = crate::config::Config::test_default();
    config.byok_direct = true;
    let keys = api::request::settings::ApiKeys {
        grok_oauth_access_token: "grok-oauth-token".to_owned(),
        ..Default::default()
    };
    let resolved = config.resolve_llm("grok-4-fast", None, Some(&keys));
    assert_eq!(resolved.base_url, "https://api.x.ai/v1");
    assert_eq!(resolved.api_key.as_deref(), Some("grok-oauth-token"));

    // Without the prefix it does not route (the token only works on xAI).
    let resolved = config.resolve_llm("claude-sonnet-4-5", None, Some(&keys));
    assert_eq!(resolved.base_url, "");
}

#[tokio::test]
async fn chatgpt_stream_builds_responses_request_and_parses_events() {
    let captured: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_in_handler = captured.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let sse = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\"}\n\n\
               data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call1\",\"name\":\"run_shell_command\"}}\n\n\
               data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"command\\\":\\\"ls\\\"}\"}\n\n\
               data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":3}}}\n\n";
    let app = axum::Router::new().route(
        "/backend-api/codex/responses",
        axum::routing::post(move |headers: axum::http::HeaderMap, body: String| {
            let captured = captured_in_handler.clone();
            async move {
                if let Ok(value) = serde_json::from_str(&body) {
                    captured.lock().unwrap().push(value);
                }
                assert!(
                    headers
                        .get("chatgpt-account-id")
                        .is_some_and(|id| id == "acct")
                );
                (
                    [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                    sse.to_owned(),
                )
            }
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let llm = ResolvedLlm {
        base_url: format!("http://{addr}/backend-api/codex"),
        schema: crate::config::LlmSchema::Chatgpt,
        api_key: Some("access".to_owned()),
        account_id: Some("acct".to_owned()),
        ..ResolvedLlm::test_default(String::new())
    };
    let mut texts = Vec::new();
    let mut tool_calls = Vec::new();
    let usage = stream_completion(
        &reqwest::Client::new(),
        &llm,
        "system",
        &[LlmMessage::User {
            text: "hi".to_owned(),
            images: Vec::new(),
        }],
        &[],
        |delta| {
            match delta {
                Delta::Text(text) => texts.push(text),
                Delta::ToolCall {
                    id,
                    name,
                    arguments,
                } => tool_calls.push((id, name, arguments)),
            }
            Ok(())
        },
    )
    .await
    .unwrap();

    assert_eq!(texts.join(""), "Hel");
    assert_eq!(tool_calls.len(), 2);
    assert_eq!(tool_calls[0].0.as_deref(), Some("call1"));
    assert_eq!(tool_calls[0].1.as_deref(), Some("run_shell_command"));
    assert_eq!(tool_calls[1].2, "{\"command\":\"ls\"}");
    assert_eq!(usage.input_tokens, 7);
    assert_eq!(usage.output_tokens, 3);

    let body = captured.lock().unwrap()[0].clone();
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["instructions"], "system");
    assert_eq!(body["store"], false);
    assert_eq!(body["input"][0]["type"], "message");
    assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
    assert_eq!(body["input"][0]["content"][0]["text"], "hi");
}

#[test]
fn chatgpt_provider_resolves_codex_backend_and_default_catalog() {
    let mut config = crate::config::Config::test_default();
    config.provider = Some(crate::config::ProviderKind::Chatgpt);
    config.codex_token_file = "/tmp/codex.json".to_owned();

    let resolved = config.resolve_llm("gpt-5-codex", None, None);
    assert_eq!(resolved.base_url, "https://chatgpt.com/backend-api/codex");
    assert_eq!(resolved.schema, crate::config::LlmSchema::Chatgpt);
    assert!(resolved.api_key.is_none());
    assert_eq!(resolved.model, "gpt-5-codex");
}

#[test]
fn native_keys_route_zai_and_opencode_models_with_headers() {
    let mut config = crate::config::Config::test_default();
    config.zai_api_key = Some("zai-key".to_owned());
    config.opencode_api_key = Some("oc-key".to_owned());
    // xAI key present too: `grok-code` must still go to OpenCode, `grok-4` to xAI.
    config.xai_api_key = Some("xai-key".to_owned());

    let resolved = config.resolve_llm("glm-4.7", None, None);
    assert_eq!(resolved.base_url, "https://api.z.ai/api/coding/paas/v4");
    assert_eq!(resolved.api_key.as_deref(), Some("zai-key"));
    assert!(resolved.headers.is_empty());

    let resolved = config.resolve_llm("grok-code", None, None);
    assert_eq!(resolved.base_url, "https://opencode.ai/zen/v1");
    assert_eq!(resolved.api_key.as_deref(), Some("oc-key"));
    assert!(
        resolved
            .headers
            .iter()
            .any(|(name, value)| name == "https-referer" && value == "https://opencode.ai/")
    );
    assert!(resolved.headers.iter().any(|(name, _)| name == "x-title"));

    let resolved = config.resolve_llm("code-supernova", None, None);
    assert_eq!(resolved.base_url, "https://opencode.ai/zen/v1");

    let resolved = config.resolve_llm("grok-4", None, None);
    assert_eq!(resolved.base_url, "https://api.x.ai/v1");
    assert_eq!(resolved.api_key.as_deref(), Some("xai-key"));

    // Native keys are also honored when the client carries BYOK keys, since
    // server-side keys take precedence.
    let client_keys = api::request::settings::ApiKeys {
        anthropic: "client-ant".to_owned(),
        ..Default::default()
    };
    let resolved = config.resolve_llm("glm-4.7", None, Some(&client_keys));
    assert_eq!(resolved.api_key.as_deref(), Some("zai-key"));

    // No matching prefix falls back to the local backend.
    let resolved = config.resolve_llm("mistral-large", None, None);
    assert_eq!(resolved.base_url, "");
}

#[tokio::test]
async fn provider_headers_are_sent_on_llm_requests() {
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen_in_handler = seen.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().route(
        "/chat/completions",
        axum::routing::post(move |headers: axum::http::HeaderMap| {
            let seen = seen_in_handler.clone();
            async move {
                for name in ["https-referer", "x-title"] {
                    if let Some(value) = headers.get(name) {
                        seen.lock().unwrap().push(format!("{name}: {value:?}"));
                    }
                }
                (
                    [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                    "data: {\"choices\":[{\"delta\":{}}]}\n\ndata: [DONE]\n\n".to_owned(),
                )
            }
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let llm = ResolvedLlm {
        headers: vec![
            (
                "https-referer".to_owned(),
                "https://opencode.ai/".to_owned(),
            ),
            ("x-title".to_owned(), "opencode".to_owned()),
        ],
        ..ResolvedLlm::test_default(format!("http://{addr}"))
    };
    stream_completion(
        &reqwest::Client::new(),
        &llm,
        "system",
        &[LlmMessage::User {
            text: "hi".to_owned(),
            images: Vec::new(),
        }],
        &[],
        |_| Ok(()),
    )
    .await
    .unwrap();

    let seen = seen.lock().unwrap();
    assert!(
        seen.iter().any(|line| line.contains("opencode.ai/")),
        "{seen:?}"
    );
    assert!(seen.iter().any(|line| line.contains("x-title")), "{seen:?}");
}
