use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use base64::prelude::BASE64_URL_SAFE;
use futures::stream::Stream;
use instant::Instant;
use prost::Message as _;
use serde_json::Value;
use warp_multi_agent_api as api;

use crate::config::Config;
use crate::metrics::MetricsHandle;

mod conversation;
pub(crate) mod llm;
mod tools;

pub use llm::{LlmMessage, ToolCallOut, Usage};

/// Shared server state handed to every handler.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub http: reqwest::Client,
    /// OAuth access tokens keyed by provider, cached until near expiry.
    pub token_cache: Arc<std::sync::Mutex<HashMap<String, (String, Instant)>>>,
    pub metrics: MetricsHandle,
    /// Model catalog, re-probed from the LLM endpoint so models pulled after
    /// startup show up in the client's picker without a backend restart.
    pub model_cache: Arc<std::sync::Mutex<ModelCache>>,
}

/// The served model list and when it was last re-probed.
#[derive(Default)]
pub struct ModelCache {
    pub models: Vec<String>,
    pub refreshed_at: Option<Instant>,
}

pub fn router(config: Config) -> axum::Router {
    // Seed the catalog with the startup configuration; an explicit
    // `--llm-model` pin holds until a live probe finds models.
    let seeded_models = if config.llm_models.is_empty() {
        config
            .llm_model
            .clone()
            .map(|model| vec![model])
            .unwrap_or_default()
    } else {
        config.llm_models.clone()
    };
    let state = AppState {
        model_cache: Arc::new(std::sync::Mutex::new(ModelCache {
            models: seeded_models,
            refreshed_at: Some(Instant::now()),
        })),
        config: Arc::new(config),
        http: reqwest::Client::new(),
        token_cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
        metrics: MetricsHandle::new(),
    };
    // The client joins its server-root URL with paths via string
    // concatenation, so a root with a trailing slash produces double slashes.
    // Register both spellings.
    axum::Router::new()
        .route("/healthz", axum::routing::get(health))
        .route("/ai/multi-agent", axum::routing::post(multi_agent))
        .route("//ai/multi-agent", axum::routing::post(multi_agent))
        .route(
            "/ai/passive-suggestions",
            axum::routing::post(passive_suggestions),
        )
        .route(
            "//ai/passive-suggestions",
            axum::routing::post(passive_suggestions),
        )
        .route(
            "/graphql/v2",
            axum::routing::post(crate::graphql::graphql).get(graphql_websocket),
        )
        .route(
            "//graphql/v2",
            axum::routing::post(crate::graphql::graphql).get(graphql_websocket),
        )
        .route(
            "/ai/transcribe",
            axum::routing::post(crate::transcribe::transcribe),
        )
        .route(
            "//ai/transcribe",
            axum::routing::post(crate::transcribe::transcribe),
        )
        .route(
            "/ai/relevant_files",
            axum::routing::post(crate::relevant_files::relevant_files),
        )
        .route(
            "//ai/relevant_files",
            axum::routing::post(crate::relevant_files::relevant_files),
        )
        .route(
            "/api/v1/agent/connected-self-hosted-workers",
            axum::routing::get(connected_self_hosted_workers),
        )
        .route(
            "//api/v1/agent/connected-self-hosted-workers",
            axum::routing::get(connected_self_hosted_workers),
        )
        .route("/api/v1/agent/runs", axum::routing::get(agent_runs))
        .route("//api/v1/agent/runs", axum::routing::get(agent_runs))
        .route(
            "/api/v1/mcp/factory",
            axum::routing::post(mcp_factory).get(mcp_factory_stream_unsupported),
        )
        .route(
            "//api/v1/mcp/factory",
            axum::routing::post(mcp_factory).get(mcp_factory_stream_unsupported),
        )
        .route("/metrics", axum::routing::get(metrics))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

/// Minimal graphql-ws listener for the client's RTC connection.
///
/// Warp's realtime socket drives Warp Drive sync and cloud-agent updates,
/// none of which exist self-hosted. Accepting the upgrade and answering the
/// protocol handshake keeps the client's listener connected and quiet instead
/// of retrying; every subscription is rejected with a graphql-ws `error`
/// message so callers see an empty result set rather than a hang.
async fn graphql_websocket(upgrade: axum::extract::WebSocketUpgrade) -> Response {
    upgrade.on_upgrade(handle_graphql_websocket)
}

async fn handle_graphql_websocket(mut socket: axum::extract::ws::WebSocket) {
    use axum::extract::ws::Message;

    async fn send(
        socket: &mut axum::extract::ws::WebSocket,
        value: Value,
    ) -> Result<(), axum::Error> {
        socket.send(Message::Text(value.to_string().into())).await
    }

    while let Some(Ok(message)) = socket.recv().await {
        let Message::Text(text) = message else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        match value["type"].as_str() {
            Some("connection_init") => {
                if send(&mut socket, serde_json::json!({"type": "connection_ack"}))
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Some("ping") => {
                if send(&mut socket, serde_json::json!({"type": "pong"}))
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Some("subscribe") => {
                let id = value["id"].clone();
                if send(
                    &mut socket,
                    serde_json::json!({
                        "type": "error",
                        "id": id,
                        "payload": [{"message":
                            "Subscriptions are not supported by the self-hosted agent server."}],
                    }),
                )
                .await
                .is_err()
                {
                    break;
                }
            }
            _ => {}
        }
    }
}

/// Minimal MCP streamable-HTTP responder for the client's bundled "factory"
/// MCP server. It answers the JSON-RPC handshake and advertises no tools, so
/// the client spawns the server successfully with an empty toolset instead of
/// failing to connect.
async fn mcp_factory(State(state): State<AppState>, headers: HeaderMap, body: String) -> Response {
    if !check_auth(&state, &headers).await {
        return unauthorized();
    }
    state.metrics.mcp_request();
    let Ok(request) = serde_json::from_str::<Value>(&body) else {
        return (StatusCode::BAD_REQUEST, "undecodable JSON-RPC").into_response();
    };
    let id = request["id"].clone();
    let Some(method) = request["method"].as_str() else {
        // Notifications get no response in JSON-RPC.
        return StatusCode::ACCEPTED.into_response();
    };
    let result = match method {
        "initialize" => serde_json::json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "selfhost-server", "version": env!("CARGO_PKG_VERSION")},
        }),
        "tools/list" => serde_json::json!({"tools": []}),
        _ => serde_json::json!({}),
    };
    let payload = if id.is_null() {
        serde_json::json!({"jsonrpc": "2.0", "result": result})
    } else {
        serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result})
    };
    axum::Json(payload).into_response()
}

async fn mcp_factory_stream_unsupported() -> Response {
    // Streamable-HTTP MCP servers may decline the server-to-client SSE stream.
    (StatusCode::METHOD_NOT_ALLOWED, "SSE stream not supported").into_response()
}

/// Prometheus-format metrics for the self-hosted server.
async fn metrics(State(state): State<AppState>) -> Response {
    state.metrics.render().into_response()
}

/// The client polls this for ambient agent runs; a self-hosted setup has
/// none, so the history is empty.
async fn agent_runs(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !check_auth(&state, &headers).await {
        return unauthorized();
    }
    axum::Json(serde_json::json!({ "runs": [] })).into_response()
}

/// The client polls this for cloud agents that can run on this machine; a
/// self-hosted setup has none.
async fn connected_self_hosted_workers(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    if !check_auth(&state, &headers).await {
        return unauthorized();
    }
    axum::Json(serde_json::json!({ "workers": [] })).into_response()
}

fn unauthorized() -> Response {
    (StatusCode::UNAUTHORIZED, "invalid or missing API key").into_response()
}

/// Extracts the client's bearer token, if any.
pub(crate) fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
}

/// Verifies the client's bearer token.
///
/// With `auth_introspect_url` configured, the token is POSTed to that URL and
/// any 2xx response accepts it — this is how an external auth service (SSO,
/// an auth gateway, Foundry-style brokers) takes over authentication.
/// Otherwise the static `--api-key` applies, and the Warp client's `wk-`
/// prefix is tolerated. With neither configured, all clients are trusted.
pub(crate) async fn check_auth(state: &AppState, headers: &HeaderMap) -> bool {
    let token = bearer_token(headers);
    if let Some(introspect_url) = &state.config.auth_introspect_url {
        let Some(token) = token else {
            return false;
        };
        return state
            .http
            .post(introspect_url)
            .json(&serde_json::json!({ "token": token }))
            .send()
            .await
            .map(|response| response.status().is_success())
            .unwrap_or(false);
    }

    let Some(expected) = &state.config.api_key else {
        return true;
    };
    let Some(token) = token else {
        return false;
    };
    token == expected || token.strip_prefix("wk-") == Some(expected.as_str())
}

async fn multi_agent(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    if !check_auth(&state, &headers).await {
        return unauthorized();
    }
    state.metrics.agent_request();
    let Ok(request) = api::Request::decode(&body[..]) else {
        return (StatusCode::BAD_REQUEST, "undecodable request").into_response();
    };
    let stream = handle_request(state, request);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn passive_suggestions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    if !check_auth(&state, &headers).await {
        return unauthorized();
    }
    let Ok(request) = api::Request::decode(&body[..]) else {
        return (StatusCode::BAD_REQUEST, "undecodable request").into_response();
    };
    let conversation_id = conversation_id(&request);
    let events: Vec<api::ResponseEvent> = vec![
        response_event_init(&conversation_id),
        // An empty actions event marks the stream as acted-upon so the client
        // does not retry it.
        client_actions_event(vec![]),
        finished_done_event(None, false),
    ];
    Sse::new(futures::stream::iter(events.into_iter().map(|event| {
        Ok::<_, std::convert::Infallible>(encode_event(event))
    })))
    .keep_alive(KeepAlive::default())
    .into_response()
}

fn conversation_id(request: &api::Request) -> String {
    let from_metadata = request
        .metadata
        .as_ref()
        .map(|metadata| metadata.conversation_id.clone())
        .unwrap_or_default();
    if from_metadata.is_empty() {
        uuid::Uuid::new_v4().to_string()
    } else {
        from_metadata
    }
}

fn response_event_init(conversation_id: &str) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::Init(
            api::response_event::StreamInit {
                conversation_id: conversation_id.to_owned(),
                request_id: uuid::Uuid::new_v4().to_string(),
                run_id: uuid::Uuid::new_v4().to_string(),
            },
        )),
    }
}

fn client_actions_event(actions: Vec<api::client_action::Action>) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: actions
                    .into_iter()
                    .map(|action| api::ClientAction {
                        action: Some(action),
                    })
                    .collect(),
            },
        )),
    }
}

fn finished_done_event(usage: Option<Usage>, done: bool) -> api::ResponseEvent {
    use api::response_event::StreamFinished;
    use api::response_event::stream_finished::{Done, Reason, TokenUsage};
    let reason = if done {
        Reason::Done(Done {})
    } else {
        Reason::Other(api::response_event::stream_finished::Other {})
    };
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::Finished(StreamFinished {
            reason: Some(reason),
            token_usage: usage
                .map(|usage| {
                    vec![TokenUsage {
                        model_id: String::new(),
                        total_input: usage.input_tokens as u32,
                        output: usage.output_tokens as u32,
                        ..Default::default()
                    }]
                })
                .unwrap_or_default(),
            ..Default::default()
        })),
    }
}

fn finished_error_event(message: String) -> api::ResponseEvent {
    use api::response_event::StreamFinished;
    use api::response_event::stream_finished::{InternalError, Reason};
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::Finished(StreamFinished {
            reason: Some(Reason::InternalError(InternalError { message })),
            ..Default::default()
        })),
    }
}

/// Generates the response-event stream for one multi-agent request.
fn handle_request(
    state: AppState,
    request: api::Request,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> + Send + 'static {
    async_stream::stream! {
        let conversation_id = conversation_id(&request);
        yield Ok(encode_event(response_event_init(&conversation_id)));

        let task_context = request.task_context.clone();
        let target_task_id = task_context
            .as_ref()
            .and_then(|context| context.tasks.last().map(|task| task.id.clone()))
            .filter(|id| !id.is_empty());
        let new_conversation = target_task_id.is_none();
        let task_id = target_task_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let supported_tools = request
            .settings
            .as_ref()
            .map(|settings| settings.supported_tools.clone())
            .unwrap_or_default();
        let supported_tool_registry = tools::tools_for_client(&supported_tools);
        let tool_defs: Vec<_> = supported_tool_registry
            .iter()
            .map(|tool| tool.def.clone())
            .collect();

        let system_prompt =
            conversation::system_prompt(state.config.system_prompt.as_deref(), &request);
        let llm_messages = conversation::translate(&request);

        let requested_model = request
            .settings
            .as_ref()
            .and_then(|settings| settings.model_config.as_ref())
            .map(|model_config| model_config.base.clone())
            .filter(|base| !base.is_empty())
            .unwrap_or_else(|| "default".to_owned());
        let custom_providers = request
            .settings
            .as_ref()
            .and_then(|settings| settings.custom_model_providers.as_ref());
        let byok_api_keys = request
            .settings
            .as_ref()
            .and_then(|settings| settings.api_keys.as_ref());
        let llm = state
            .config
            .resolve_llm(&requested_model, custom_providers, byok_api_keys);

        // The raw query text from this request's input: used for the task
        // description on new conversations, and recorded into the client's
        // task state if it is not already there.
        let input_query = request.input.as_ref().and_then(|input| {
            let Some(api::request::input::Type::UserInputs(user_inputs)) = &input.r#type else {
                return None;
            };
            user_inputs.inputs.iter().rev().find_map(|entry| match &entry.input {
                Some(api::request::input::user_inputs::user_input::Input::UserQuery(query)) => {
                    (!query.query.is_empty()).then(|| query.query.clone())
                }
                _ => None,
            })
        });

        // A new conversation needs its task and the user's query message.
        let mut setup_actions = Vec::new();
        setup_actions.push(api::client_action::Action::BeginTransaction(
            api::client_action::BeginTransaction {},
        ));
        if new_conversation {
            let description = input_query
                .as_deref()
                .map(first_line)
                .unwrap_or_else(|| "Agent task".to_owned());
            setup_actions.push(api::client_action::Action::CreateTask(
                api::client_action::CreateTask {
                    task: Some(api::Task {
                        id: task_id.clone(),
                        description,
                        ..Default::default()
                    }),
                },
            ));
        }
        if let Some(query) = &input_query
            && !user_query_recorded(task_context.as_ref(), query)
        {
            setup_actions.push(add_message_action(
                &task_id,
                user_query_message(&task_id, query),
            ));
        }
        yield Ok(encode_event(client_actions_event(setup_actions)));

        let mut emitted_actions = false;
        let worker_system_prompt = system_prompt;
        let worker_messages = llm_messages;

        // Context-window management: when the conversation nears the budget,
        // summarize the older prefix into a client-visible summary and run the
        // turn from a compacted view.
        let mut convo = worker_messages;
        let budget = conversation::effective_context_budget(
            state.config.context_window_tokens,
            request.settings.as_ref(),
        );
        if budget > 0
            && conversation::estimate_tokens(&worker_system_prompt, &convo)
                > budget - budget / 5
            && let Some(target_task) = task_context
                .as_ref()
                .and_then(|context| context.tasks.last())
            && let Some((move_action, compacted)) =
                summarize_prefix(&state, &llm, target_task, budget).await
        {
            emitted_actions = true;
            yield Ok(encode_event(client_actions_event(vec![move_action])));
            convo = compacted;
        }

        // Run the LLM on a worker task, streaming actions back over a channel
        // so they can be yielded as they arrive. The worker loops while the
        // model calls the server-executed web-search tool, feeding results
        // back into the conversation before finishing.
        enum LlmUpdate {
            Actions(Vec<api::client_action::Action>),
            Done(Result<llm::Usage, anyhow::Error>),
        }

        let web_search_allowed = web_search_enabled(&state.config, request.settings.as_ref());
        let worker_mcp_context = request.mcp_context.clone();
        let token_cache_metrics = state.metrics.clone();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<LlmUpdate>();
        let http = state.http.clone();
        let token_cache = state.token_cache.clone();
        let web_search_url = state.config.web_search_url.clone();
        let codex_token_file = state.config.codex_token_file.clone();
        let worker = tokio::spawn(async move {
            // OAuth-backed providers fetch (a cached) bearer token first.
            let mut llm = llm;
            if let Some(dynamic_auth) = llm.dynamic_auth.clone() {
                match crate::oauth::resolve_token(&http, &dynamic_auth, &token_cache).await {
                    Ok(token) => llm.api_key = Some(token),
                    Err(error) => {
                        let _ = tx.send(LlmUpdate::Done(Err(error)));
                        return;
                    }
                }
            }
            if llm.schema == crate::config::LlmSchema::Chatgpt {
                match crate::codex::load_or_refresh(&http, &codex_token_file).await {
                    Ok(tokens) => {
                        llm.api_key = Some(tokens.access_token);
                        llm.account_id = tokens.account_id;
                    }
                    Err(error) => {
                        let _ = tx.send(LlmUpdate::Done(Err(error)));
                        return;
                    }
                }
            }
            let latest_usage = llm::Usage::default();
            let mut assistant_open = false;
            let mut assistant_message_id = uuid::Uuid::new_v4().to_string();
            let mut client_tool_calls: Vec<ToolCallOut> = Vec::new();
            let mut round_tools = tool_defs.clone();
            if web_search_allowed {
                round_tools.push(crate::websearch::tool_def());
            }

            let mut result = Ok(latest_usage);
            for round in 0..MAX_WEB_SEARCH_ROUNDS {
                let allow_web = web_search_allowed && round + 1 < MAX_WEB_SEARCH_ROUNDS;
                let tools_for_round: Vec<llm::ToolDef> = if allow_web {
                    round_tools.clone()
                } else {
                    tool_defs.clone()
                };
                let mut server_search_calls: Vec<ToolCallOut> = Vec::new();
                result = llm::stream_completion(
                    &http,
                    &llm,
                    &worker_system_prompt,
                    &convo,
                    &tools_for_round,
                    |delta| {
                        match delta {
                            llm::Delta::Text(text) => {
                                let mut batch = Vec::new();
                                if !assistant_open {
                                    assistant_open = true;
                                    batch.push(add_message_action(
                                        &task_id,
                                        assistant_output_message(
                                            &task_id,
                                            &assistant_message_id,
                                            "",
                                        ),
                                    ));
                                }
                                batch.push(api::client_action::Action::AppendToMessageContent(
                                    api::client_action::AppendToMessageContent {
                                        task_id: task_id.clone(),
                                        message: Some(assistant_output_message(
                                            &task_id,
                                            &assistant_message_id,
                                            &text,
                                        )),
                                        mask: Some(prost_types::FieldMask {
                                            paths: vec!["agent_output.text".to_owned()],
                                        }),
                                    },
                                ));
                                let _ = tx.send(LlmUpdate::Actions(batch));
                            }
                            llm::Delta::ToolCall { id, name, arguments } => {
                                if name.as_deref() == Some(crate::websearch::WEB_SEARCH_TOOL_NAME) {
                                    merge_tool_call(&mut server_search_calls, id, name, arguments);
                                } else {
                                    merge_tool_call(&mut client_tool_calls, id, name, arguments);
                                }
                            }
                        }
                        Ok(())
                    },
                )
                .await;
                if result.is_err() {
                    break;
                }
                if server_search_calls.is_empty() {
                    break;
                }

                // Execute the searches server-side and feed the results back
                // into the conversation for the next round.
                let mut result_messages = Vec::new();
                for call in &server_search_calls {
                    let query = web_search_query(call);
                    let (ui_message, llm_text) =
                        match crate::websearch::search(&http, web_search_url.as_deref().unwrap_or_default(), &query)
                            .await
                        {
                            Ok(results) => (
                                web_search_message(&task_id, &query, Some(&results)),
                                crate::websearch::render_for_llm(&query, &results),
                            ),
                            Err(error) => (
                                web_search_message(&task_id, &query, None),
                                format!("Web search failed: {error:#}"),
                            ),
                        };
                    let _ = tx.send(LlmUpdate::Actions(vec![add_message_action(&task_id, ui_message)]));
                    result_messages.push(LlmMessage::ToolResult {
                        tool_call_id: call.id.clone(),
                        text: llm_text,
                    });
                }
                convo.push(LlmMessage::Assistant {
                    text: String::new(),
                    tool_calls: server_search_calls,
                });
                convo.extend(result_messages);
                // The next round streams into a fresh assistant message.
                assistant_open = false;
                assistant_message_id = uuid::Uuid::new_v4().to_string();
            }

            if let Err(error) = &result {
                tracing::error!("LLM completion failed: {error:#}");
            }
            if result.is_ok() {
                token_cache_metrics
                    .agent_tool_calls(client_tool_calls.len());
                let mut final_actions = tool_call_actions(
                    &supported_tool_registry,
                    &task_id,
                    &client_tool_calls,
                    worker_mcp_context.as_ref(),
                );
                final_actions.push(add_message_action(
                    &task_id,
                    model_used_message(&task_id, &llm.model),
                ));
                let _ = tx.send(LlmUpdate::Actions(final_actions));
            }
            let _ = tx.send(LlmUpdate::Done(result));
        });

        let mut usage = None;
        let failure = None;
        while let Some(update) = rx.recv().await {
            match update {
                LlmUpdate::Actions(actions) if !actions.is_empty() => {
                    emitted_actions = true;
                    yield Ok(encode_event(client_actions_event(actions)));
                }
                LlmUpdate::Actions(_) => {}
                LlmUpdate::Done(Ok(done_usage)) => {
                    usage = Some(done_usage);
                    state
                        .metrics
                        .usage(done_usage.input_tokens, done_usage.output_tokens);
                }
                LlmUpdate::Done(Err(_)) => state.metrics.agent_llm_error(),
            }
        }
        let _ = worker.await;

        if let Some(error) = failure {
            if emitted_actions {
                yield Ok(encode_event(client_actions_event(vec![
                    api::client_action::Action::RollbackTransaction(
                        api::client_action::RollbackTransaction {},
                    ),
                ])));
            }
            if is_context_window_error(&error) {
                yield Ok(encode_event(finished_context_window_exceeded_event()));
            } else {
                yield Ok(encode_event(finished_error_event(format!(
                    "self-hosted LLM call failed: {error:#}"
                ))));
            }
        } else {
            yield Ok(encode_event(client_actions_event(vec![
                api::client_action::Action::CommitTransaction(
                    api::client_action::CommitTransaction {},
                ),
            ])));
            yield Ok(encode_event(finished_done_event(usage, true)));
        }
    }
}

/// Builds `AddMessagesToTask` actions for the model's completed tool calls.
/// Calls to tools the client does not support become explanatory output
/// messages instead.
fn tool_call_actions(
    registry: &[tools::SupportedTool],
    task_id: &str,
    pending_tool_calls: &[ToolCallOut],
    mcp_context: Option<&api::request::McpContext>,
) -> Vec<api::client_action::Action> {
    let mut actions = Vec::new();
    for call in pending_tool_calls {
        let arguments = match &call.arguments {
            serde_json::Value::String(raw) => {
                serde_json::from_str(raw).unwrap_or_else(|_| serde_json::json!({}))
            }
            other => other.clone(),
        };
        let Some(tool) = registry.iter().find(|tool| tool.def.name == call.name) else {
            actions.push(add_message_action(
                task_id,
                assistant_output_message(
                    task_id,
                    &uuid::Uuid::new_v4().to_string(),
                    &format!(
                        "The model attempted to call the unsupported tool '{}'.",
                        call.name
                    ),
                ),
            ));
            continue;
        };
        let mut tool_call = match (tool.build)(uuid::Uuid::new_v4().to_string(), &arguments) {
            Ok(tool_call) => tool_call,
            Err(error) => {
                actions.push(add_message_action(
                    task_id,
                    assistant_output_message(
                        task_id,
                        &uuid::Uuid::new_v4().to_string(),
                        &format!(
                            "The model called '{}' with invalid arguments: {error:#}",
                            call.name
                        ),
                    ),
                ));
                continue;
            }
        };
        resolve_mcp_server_id(&mut tool_call, call, mcp_context);
        actions.push(add_message_action(
            task_id,
            api::Message {
                id: uuid::Uuid::new_v4().to_string(),
                task_id: task_id.to_owned(),
                message: Some(api::message::Message::ToolCall(tool_call)),
                ..Default::default()
            },
        ));
    }
    actions
}

/// Attributes an MCP tool call to the server in the client's MCP context that
/// provides the tool, so the client knows where to route it.
fn resolve_mcp_server_id(
    tool_call: &mut api::message::ToolCall,
    call: &ToolCallOut,
    mcp_context: Option<&api::request::McpContext>,
) {
    let Some(api::message::tool_call::Tool::CallMcpTool(mcp_call)) = tool_call.tool.as_mut() else {
        return;
    };
    if !mcp_call.server_id.is_empty() {
        return;
    }
    let empty = Vec::new();
    let servers = mcp_context
        .map(|context| &context.servers)
        .unwrap_or(&empty);
    if let Some(server) = servers
        .iter()
        .find(|server| server.tools.iter().any(|tool| tool.name == call.name))
    {
        mcp_call.server_id = server.id.clone();
    } else if let Some(server) = servers.first() {
        // Fall back to the only/first server so the call still has a target.
        mcp_call.server_id = server.id.clone();
    }
}

fn first_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or_default();
    let mut truncated: String = line.chars().take(80).collect();
    if truncated.is_empty() {
        truncated = "Agent task".to_owned();
    }
    truncated
}

fn user_query_recorded(task_context: Option<&api::request::TaskContext>, query: &str) -> bool {
    !task_context
        .map(|context| {
            context.tasks.iter().any(|task| {
                task.messages.iter().any(|message| {
                    matches!(
                        &message.message,
                        Some(api::message::Message::UserQuery(user_query))
                            if user_query.query == query
                    )
                })
            })
        })
        .unwrap_or(false)
}

fn user_query_message(task_id: &str, query: &str) -> api::Message {
    api::Message {
        id: uuid::Uuid::new_v4().to_string(),
        task_id: task_id.to_owned(),
        message: Some(api::message::Message::UserQuery(api::message::UserQuery {
            query: query.to_owned(),
            ..Default::default()
        })),
        ..Default::default()
    }
}

fn assistant_output_message(task_id: &str, message_id: &str, text: &str) -> api::Message {
    api::Message {
        id: message_id.to_owned(),
        task_id: task_id.to_owned(),
        message: Some(api::message::Message::AgentOutput(
            api::message::AgentOutput {
                text: text.to_owned(),
            },
        )),
        ..Default::default()
    }
}

fn add_message_action(task_id: &str, message: api::Message) -> api::client_action::Action {
    api::client_action::Action::AddMessagesToTask(api::client_action::AddMessagesToTask {
        task_id: task_id.to_owned(),
        messages: vec![message],
    })
}

fn merge_tool_call(
    pending: &mut Vec<ToolCallOut>,
    id: Option<String>,
    name: Option<String>,
    arguments: String,
) {
    if let Some(name) = name {
        let mut call = ToolCallOut {
            id: id.unwrap_or_default(),
            name,
            arguments: serde_json::Value::Null,
        };
        append_arguments(&mut call, &arguments);
        pending.push(call);
    } else if let Some(last) = pending.last_mut() {
        append_arguments(last, &arguments);
    }
}

fn append_arguments(call: &mut ToolCallOut, arguments: &str) {
    if arguments.is_empty() {
        return;
    }
    let current = match &call.arguments {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    call.arguments = serde_json::Value::String(format!("{current}{arguments}"));
}

fn encode_event(event: api::ResponseEvent) -> Event {
    let mut bytes = Vec::with_capacity(event.encoded_len());
    event
        .encode(&mut bytes)
        .expect("ResponseEvent always encodes");
    Event::default().data(BASE64_URL_SAFE.encode(bytes))
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

/// Maximum server-side web-search rounds per request before finishing.
const MAX_WEB_SEARCH_ROUNDS: usize = 3;

/// Prompt used to compress older conversation history.
const SUMMARIZATION_PROMPT: &str = "You compress coding-agent conversations so work can continue \
in a fresh context. Summarize: the user's goal, what has been done so far, the key files, \
commands and their outcomes, the current state, and any next steps. Be factual and compact; \
plain prose and short bullets.";

/// Summarizes the older prefix of the target task's history, returning the
/// client action that moves it into a summary subtask plus the compacted LLM
/// view to run the turn from.
async fn summarize_prefix(
    state: &AppState,
    llm: &crate::config::ResolvedLlm,
    task: &api::Task,
    budget: u64,
) -> Option<(api::client_action::Action, Vec<LlmMessage>)> {
    let cut = conversation::compaction_cut(&task.messages, budget)?;
    let (prefix, suffix) = task.messages.split_at(cut);
    let to_summarize = conversation::render_messages_for_summary(prefix);
    let to_summarize = truncate_chars(&to_summarize, 60_000);

    let mut summary_text = String::new();
    llm::stream_completion(
        &state.http,
        llm,
        SUMMARIZATION_PROMPT,
        &[LlmMessage::User {
            text: to_summarize,
            images: Vec::new(),
        }],
        &[],
        |delta| {
            if let llm::Delta::Text(text) = delta {
                summary_text.push_str(&text);
            }
            Ok(())
        },
    )
    .await
    .ok()?;
    let summary = summary_text.trim();
    if summary.is_empty() {
        return None;
    }

    let token_count = conversation::estimate_tokens(
        "",
        &[LlmMessage::User {
            text: summary.to_owned(),
            images: Vec::new(),
        }],
    ) as i32;
    let move_action = api::client_action::Action::MoveMessagesToNewTask(
        api::client_action::MoveMessagesToNewTask {
            source_task_id: task.id.clone(),
            new_task: Some(api::Task {
                id: uuid::Uuid::new_v4().to_string(),
                description: "Earlier conversation".to_owned(),
                dependencies: Some(api::task::Dependencies {
                    parent_task_id: task.id.clone(),
                }),
                ..Default::default()
            }),
            first_message_id: prefix.first()?.id.clone(),
            last_message_id: prefix.last()?.id.clone(),
            expected_message_count: prefix.len() as u32,
            replacement_messages: vec![api::Message {
                id: uuid::Uuid::new_v4().to_string(),
                task_id: task.id.clone(),
                message: Some(api::message::Message::Summarization(
                    api::message::Summarization {
                        finished_duration: None,
                        summary_type: Some(
                            api::message::summarization::SummaryType::ConversationSummary(
                                api::message::summarization::ConversationSummary {
                                    summary: summary.to_owned(),
                                    token_count,
                                },
                            ),
                        ),
                    },
                )),
                ..Default::default()
            }],
        },
    );
    Some((move_action, conversation::compacted_view(summary, suffix)))
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((index, _)) => format!("{}\n[...truncated...]", &text[..index]),
        None => text.to_owned(),
    }
}

/// Whether the server offers its web-search tool for this request.
fn web_search_enabled(
    config: &crate::config::Config,
    settings: Option<&api::request::Settings>,
) -> bool {
    config.web_search_url.is_some() && settings.is_some_and(|settings| settings.web_search_enabled)
}

/// Extracts the query from the model's web_search tool-call arguments.
fn web_search_query(call: &ToolCallOut) -> String {
    let arguments = match &call.arguments {
        serde_json::Value::String(raw) => {
            serde_json::from_str::<serde_json::Value>(raw).unwrap_or(serde_json::Value::Null)
        }
        other => other.clone(),
    };
    arguments["query"].as_str().unwrap_or_default().to_owned()
}

/// The client-visible web-search message: results on success, an error status
/// otherwise.
fn web_search_message(
    task_id: &str,
    query: &str,
    results: Option<&[crate::websearch::SearchResult]>,
) -> api::Message {
    use api::message::web_search::status::{self};
    let status = match results {
        Some(results) => status::Type::Success(status::Success {
            query: query.to_owned(),
            pages: results
                .iter()
                .map(|result| status::success::SearchedPage {
                    url: result.url.clone(),
                    title: result.title.clone(),
                })
                .collect(),
        }),
        None => status::Type::Error(()),
    };
    api::Message {
        id: uuid::Uuid::new_v4().to_string(),
        task_id: task_id.to_owned(),
        message: Some(api::message::Message::WebSearch(api::message::WebSearch {
            status: Some(api::message::web_search::Status {
                r#type: Some(status),
            }),
        })),
        ..Default::default()
    }
}

/// Tells the client which model served this turn.
fn model_used_message(task_id: &str, model: &str) -> api::Message {
    api::Message {
        id: uuid::Uuid::new_v4().to_string(),
        task_id: task_id.to_owned(),
        message: Some(api::message::Message::ModelUsed(api::message::ModelUsed {
            model_id: model.to_owned(),
            model_display_name: model.to_owned(),
            is_fallback: false,
            ..Default::default()
        })),
        ..Default::default()
    }
}

/// Heuristically detects provider errors caused by an over-long prompt, which
/// the client surfaces as a context-window condition.
fn is_context_window_error(error: &anyhow::Error) -> bool {
    let text = format!("{error:#}").to_ascii_lowercase();
    [
        "context length",
        "context window",
        "maximum context",
        "input length exceeds",
        "prompt is too long",
        "too many tokens",
        "reduce the length",
    ]
    .iter()
    .any(|needle| text.contains(needle))
}

fn finished_context_window_exceeded_event() -> api::ResponseEvent {
    use api::response_event::StreamFinished;
    use api::response_event::stream_finished::{ContextWindowExceeded, Reason};
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::Finished(StreamFinished {
            reason: Some(Reason::ContextWindowExceeded(ContextWindowExceeded {})),
            ..Default::default()
        })),
    }
}
