use std::sync::Mutex;

use base64::Engine as _;
use prost::Message as _;
use serde_json::Value;

use super::*;

/// A captured LLM request body plus a canned OpenAI-style SSE reply.
#[derive(Clone)]
struct StubLlm {
    bodies: Arc<Mutex<Vec<serde_json::Value>>>,
    replies: Arc<Mutex<Vec<String>>>,
}

fn sse_stream(events: &[serde_json::Value]) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str(&format!("data: {}\n\n", event));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

async fn spawn_stub_llm(stub: StubLlm) -> std::net::SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(move |body: String| {
            let stub = stub.clone();
            async move {
                if let Ok(value) = serde_json::from_str(&body) {
                    stub.bodies.lock().unwrap().push(value);
                }
                let reply = {
                    let mut queue = stub.replies.lock().unwrap();
                    if queue.is_empty() {
                        sse_stream(&[])
                    } else {
                        queue.remove(0)
                    }
                };
                (
                    [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                    reply,
                )
            }
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

fn test_config(llm_addr: std::net::SocketAddr) -> crate::config::Config {
    crate::config::Config {
        llm_base_url: format!("http://{llm_addr}/v1"),
        ..crate::config::Config::test_default()
    }
}

fn post_request(
    addr: std::net::SocketAddr,
    path: &str,
    request: &api::Request,
    bearer: Option<&str>,
) -> reqwest::RequestBuilder {
    let client = reqwest::Client::new();
    let mut builder = client
        .post(format!("http://{addr}{path}"))
        .body(request.encode_to_vec());
    if let Some(bearer) = bearer {
        builder = builder.bearer_auth(bearer);
    }
    builder
}

/// Reads an SSE body into decoded `ResponseEvent`s.
fn decode_events(body: &str) -> Vec<api::ResponseEvent> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| {
            let bytes = BASE64_URL_SAFE.decode(data.trim()).unwrap();
            api::ResponseEvent::decode(bytes.as_slice()).unwrap()
        })
        .collect()
}

fn actions_of(event: &api::ResponseEvent) -> Vec<api::client_action::Action> {
    match &event.r#type {
        Some(api::response_event::Type::ClientActions(actions)) => actions
            .actions
            .iter()
            .filter_map(|action| action.action.clone())
            .collect(),
        _ => Vec::new(),
    }
}

fn user_query_request(query: &str) -> api::Request {
    api::Request {
        settings: Some(api::request::Settings {
            model_config: Some(api::request::settings::ModelConfig {
                base: "my-model".to_owned(),
                ..Default::default()
            }),
            ..Default::default()
        }),
        input: Some(api::request::Input {
            r#type: Some(api::request::input::Type::UserInputs(
                api::request::input::UserInputs {
                    inputs: vec![api::request::input::user_inputs::UserInput {
                        input: Some(
                            api::request::input::user_inputs::user_input::Input::UserQuery(
                                api::request::input::UserQuery {
                                    query: query.to_owned(),
                                    ..Default::default()
                                },
                            ),
                        ),
                    }],
                },
            )),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn full_agent_round_trip_streams_valid_events() {
    // Round 1 replies with a tool call; round 2 with plain text.
    let replies = vec![
        sse_stream(&[
            serde_json::json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call1","function":{"name":"run_shell_command","arguments":"{\"command\":\"ls -la\"}"}}]}}]}),
            serde_json::json!({"choices":[{"finish_reason":"tool_calls"}]}),
        ]),
        sse_stream(&[
            serde_json::json!({"choices":[{"delta":{"content":"Look "}}]}),
            serde_json::json!({"choices":[{"delta":{"content":"around."}}]}),
            serde_json::json!({"choices":[{"finish_reason":"stop"}]}),
        ]),
    ];
    let stub = StubLlm {
        bodies: Arc::new(Mutex::new(Vec::new())),
        replies: Arc::new(Mutex::new(replies)),
    };
    let llm_addr = spawn_stub_llm(stub.clone()).await;
    let app = router(test_config(llm_addr));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // --- Round 1: the model answers with a tool call. ---
    let response = post_request(
        addr,
        "/ai/multi-agent",
        &user_query_request("list files"),
        None,
    )
    .send()
    .await
    .unwrap();
    assert!(response.status().is_success());
    let events = decode_events(&response.text().await.unwrap());

    assert!(
        matches!(
            &events[0].r#type,
            Some(api::response_event::Type::Init(init))
                if !init.conversation_id.is_empty()
        ),
        "first event must be Init: {:?}",
        events[0]
    );

    let all_actions: Vec<_> = events.iter().flat_map(actions_of).collect();
    assert!(
        all_actions
            .iter()
            .any(|action| matches!(action, api::client_action::Action::CreateTask(_))),
        "expected CreateTask in {all_actions:?}"
    );

    assert!(
        all_actions.iter().any(|action| matches!(
            action,
            api::client_action::Action::AddMessagesToTask(add)
                if add.messages.iter().any(|message| matches!(
                    &message.message,
                    Some(api::message::Message::ToolCall(call))
                        if matches!(
                            &call.tool,
                            Some(api::message::tool_call::Tool::RunShellCommand(run))
                                if run.command == "ls -la"
                        )
                ))
        )),
        "expected a run_shell_command ToolCall action in {all_actions:?}"
    );
    assert!(matches!(
        &all_actions.last().unwrap(),
        api::client_action::Action::CommitTransaction(_)
    ));
    // Round 1 must not emit plain assistant text alongside the tool call.
    assert!(
        !all_actions.iter().any(|action| matches!(
            action,
            api::client_action::Action::AppendToMessageContent(_)
        )),
        "unexpected text appends in round 1: {all_actions:?}"
    );
    assert!(matches!(
        &events.last().unwrap().r#type,
        Some(api::response_event::Type::Finished(finished))
            if matches!(&finished.reason, Some(api::response_event::stream_finished::Reason::Done(_)))
    ));

    // The LLM saw the system prompt and the user query.
    let first_llm_request = {
        let bodies = stub.bodies.lock().unwrap();
        bodies[0].clone()
    };
    assert_eq!(first_llm_request["model"], "my-model");
    assert_eq!(first_llm_request["messages"][0]["role"], "system");
    assert!(
        first_llm_request["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|message| message["role"] == "user"
                && message["content"]
                    .as_str()
                    .is_some_and(|content| content.contains("list files")))
    );
    assert!(!first_llm_request["tools"].as_array().unwrap().is_empty());

    // --- Round 2: the client reports the tool result; the model wraps up. ---
    let mut request = user_query_request("list files");
    let task_id = "task-1".to_owned();
    request.task_context = Some(api::request::TaskContext {
        tasks: vec![api::Task {
            id: task_id.clone(),
            messages: vec![
                api::Message {
                    id: "m1".to_owned(),
                    task_id: task_id.clone(),
                    message: Some(api::message::Message::UserQuery(api::message::UserQuery {
                        query: "list files".to_owned(),
                        ..Default::default()
                    })),
                    ..Default::default()
                },
                api::Message {
                    id: "m2".to_owned(),
                    task_id: task_id.clone(),
                    message: Some(api::message::Message::ToolCall(api::message::ToolCall {
                        tool_call_id: "call1".to_owned(),
                        tool: Some(api::message::tool_call::Tool::RunShellCommand(
                            api::message::tool_call::RunShellCommand {
                                command: "ls -la".to_owned(),
                                ..Default::default()
                            },
                        )),
                    })),
                    ..Default::default()
                },
                api::Message {
                    id: "m3".to_owned(),
                    task_id: task_id.clone(),
                    message: Some(api::message::Message::ToolCallResult(
                        api::message::ToolCallResult {
                            tool_call_id: "call1".to_owned(),
                            result: Some(api::message::tool_call_result::Result::RunShellCommand(
                                api::RunShellCommandResult {
                                    result: Some(
                                        api::run_shell_command_result::Result::CommandFinished(
                                            api::ShellCommandFinished {
                                                output: "file-a.txt".to_owned(),
                                                exit_code: 0,
                                                ..Default::default()
                                            },
                                        ),
                                    ),
                                    ..Default::default()
                                },
                            )),
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }],
    });
    request.metadata = Some(api::request::Metadata {
        conversation_id: "conv-1".to_owned(),
        ..Default::default()
    });

    let response = post_request(addr, "/ai/multi-agent", &request, None)
        .send()
        .await
        .unwrap();
    let events = decode_events(&response.text().await.unwrap());

    // Continuing a conversation must not create a second task.
    let all_actions: Vec<_> = events.iter().flat_map(actions_of).collect();
    assert!(
        !all_actions
            .iter()
            .any(|action| matches!(action, api::client_action::Action::CreateTask(_))),
        "unexpected CreateTask on continuation: {all_actions:?}"
    );
    assert!(matches!(
        &events[0].r#type,
        Some(api::response_event::Type::Init(init)) if init.conversation_id == "conv-1"
    ));
    let appended_text: String = all_actions
        .iter()
        .filter_map(|action| match action {
            api::client_action::Action::AppendToMessageContent(append) => match &append.message {
                Some(message) => match &message.message {
                    Some(api::message::Message::AgentOutput(output)) => Some(output.text.clone()),
                    _ => None,
                },
                None => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(appended_text, "Look around.");

    // The LLM request replayed the full history including the tool result.
    let bodies = stub.bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 2);
    let second_llm_request = &bodies[1];
    let tool_messages: Vec<_> = second_llm_request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["role"] == "tool")
        .collect();
    assert_eq!(tool_messages.len(), 1);
    assert_eq!(tool_messages[0]["tool_call_id"], "call1");
    assert!(
        tool_messages[0]["content"]
            .as_str()
            .is_some_and(|content| content.contains("file-a.txt")),
        "tool result output missing: {tool_messages:?}"
    );
}

#[tokio::test]
async fn passive_suggestions_finish_without_actions() {
    let stub = StubLlm {
        bodies: Arc::new(Mutex::new(Vec::new())),
        replies: Arc::new(Mutex::new(Vec::new())),
    };
    let llm_addr = spawn_stub_llm(stub).await;
    let app = router(test_config(llm_addr));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let request = api::Request {
        input: Some(api::request::Input {
            r#type: Some(api::request::input::Type::GeneratePassiveSuggestions(
                api::request::input::GeneratePassiveSuggestions::default(),
            )),
            ..Default::default()
        }),
        ..Default::default()
    };
    let response = post_request(addr, "/ai/passive-suggestions", &request, None)
        .send()
        .await
        .unwrap();
    let events = decode_events(&response.text().await.unwrap());
    assert_eq!(events.len(), 3);
    assert!(events.iter().all(|event| {
        !matches!(&event.r#type, Some(api::response_event::Type::ClientActions(actions)) if !actions.actions.is_empty())
    }));
}

#[tokio::test]
async fn api_key_auth_is_enforced() {
    let stub = StubLlm {
        bodies: Arc::new(Mutex::new(Vec::new())),
        replies: Arc::new(Mutex::new(Vec::new())),
    };
    let llm_addr = spawn_stub_llm(stub).await;
    let mut config = test_config(llm_addr);
    config.api_key = Some("sekrit".to_owned());
    let app = router(config);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let request = user_query_request("hi");
    let denied = post_request(addr, "/ai/multi-agent", &request, None)
        .send()
        .await
        .unwrap();
    assert_eq!(denied.status(), reqwest::StatusCode::UNAUTHORIZED);

    // The Warp client prefixes API keys with `wk-`.
    let accepted = post_request(addr, "/ai/multi-agent", &request, Some("wk-sekrit"))
        .send()
        .await
        .unwrap();
    assert!(accepted.status().is_success());
}

#[tokio::test]
async fn graphql_serves_user_record_and_degrades_other_queries() {
    let stub = StubLlm {
        bodies: Arc::new(Mutex::new(Vec::new())),
        replies: Arc::new(Mutex::new(Vec::new())),
    };
    let llm_addr = spawn_stub_llm(stub).await;
    let app = router(test_config(llm_addr));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let client = reqwest::Client::new();
    let get_user = client
        .post(format!("http://{addr}/graphql/v2"))
        .body(r#"{"query":"{ user { globalSkills } }"}"#)
        .send()
        .await
        .unwrap();
    assert!(get_user.status().is_success());
    let body: serde_json::Value = get_user.json().await.unwrap();
    assert_eq!(body["data"]["user"]["__typename"], "UserOutput");

    let other = client
        .post(format!("http://{addr}/graphql/v2"))
        .body(r#"{"query":"mutation { generateApiKey { apiKey { secret } } }"}"#)
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = other.json().await.unwrap();
    assert!(body["data"].is_null());
    assert!(
        body["errors"][0]["message"]
            .as_str()
            .is_some_and(|message| message.contains("not supported"))
    );
}

fn long_history_request() -> api::Request {
    let mut request = user_query_request("continue the work");
    let mut messages = Vec::new();
    for index in 0..3 {
        let content = format!(
            "Round {index}: we investigated the flaky test suite and discovered a race \
             in the scheduler queue; multiple fixes were attempted and reverted before \
             landing the current retry strategy, which so far looks stable."
        );
        messages.push(api::Message {
            id: format!("m{}", index * 2),
            task_id: "task-1".to_owned(),
            message: Some(api::message::Message::UserQuery(api::message::UserQuery {
                query: content.clone(),
                ..Default::default()
            })),
            ..Default::default()
        });
        messages.push(api::Message {
            id: format!("m{}", index * 2 + 1),
            task_id: "task-1".to_owned(),
            message: Some(api::message::Message::AgentOutput(
                api::message::AgentOutput { text: content },
            )),
            ..Default::default()
        });
    }
    request.task_context = Some(api::request::TaskContext {
        tasks: vec![api::Task {
            id: "task-1".to_owned(),
            messages,
            ..Default::default()
        }],
    });
    request.metadata = Some(api::request::Metadata {
        conversation_id: "conv-2".to_owned(),
        ..Default::default()
    });
    request
}

#[tokio::test]
async fn compacts_old_history_into_a_summary() {
    let replies = vec![
        sse_stream(&[
            serde_json::json!({"choices":[{"delta":{"content":"Summary of earlier work."}}]}),
        ]),
        sse_stream(&[serde_json::json!({"choices":[{"delta":{"content":"Continuing."}}]})]),
    ];
    let stub = StubLlm {
        bodies: Arc::new(Mutex::new(Vec::new())),
        replies: Arc::new(Mutex::new(replies)),
    };
    let llm_addr = spawn_stub_llm(stub.clone()).await;
    let mut config = test_config(llm_addr);
    // Tiny budget so the long history above trips compaction.
    config.context_window_tokens = 400;
    let app = router(config);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let response = post_request(addr, "/ai/multi-agent", &long_history_request(), None)
        .send()
        .await
        .unwrap();
    let events = decode_events(&response.text().await.unwrap());
    let all_actions: Vec<_> = events.iter().flat_map(actions_of).collect();

    let move_action = all_actions
        .iter()
        .find_map(|action| match action {
            api::client_action::Action::MoveMessagesToNewTask(move_messages) => Some(move_messages),
            _ => None,
        })
        .expect("expected a MoveMessagesToNewTask action");
    assert_eq!(move_action.source_task_id, "task-1");
    assert_eq!(move_action.first_message_id, "m0");
    assert!(move_action.expected_message_count >= 1 && move_action.expected_message_count <= 6);
    let replacement = move_action
        .replacement_messages
        .iter()
        .find_map(|message| match &message.message {
            Some(api::message::Message::Summarization(summarization)) => Some(summarization),
            _ => None,
        })
        .expect("expected a Summarization replacement message");
    match &replacement.summary_type {
        Some(api::message::summarization::SummaryType::ConversationSummary(summary)) => {
            assert_eq!(summary.summary, "Summary of earlier work.");
        }
        other => panic!("unexpected summary type: {other:?}"),
    }
    // The moved messages went into a subtask of the source task.
    assert_eq!(
        move_action
            .new_task
            .as_ref()
            .unwrap()
            .dependencies
            .as_ref()
            .unwrap()
            .parent_task_id,
        "task-1"
    );

    // The first LLM call was the summarization; the second ran on the
    // compacted view containing the summary, not the old text.
    let bodies = stub.bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 2);
    assert!(
        bodies[0].to_string().contains("Round 0:"),
        "summarization call should see the old history"
    );
    let main_messages = bodies[1]["messages"].as_array().unwrap();
    assert!(
        serde_json::to_string(&main_messages)
            .unwrap()
            .contains("Summary of earlier work."),
        "main call should carry the summary"
    );
    assert!(
        !serde_json::to_string(&main_messages)
            .unwrap()
            .contains("Round 0:"),
        "main call should not carry the compacted-away text"
    );
}

#[tokio::test]
async fn web_search_runs_server_side_and_feeds_the_model() {
    let replies = vec![
        sse_stream(&[
            serde_json::json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"ws1","function":{"name":"web_search","arguments":"{\"query\":\"rust ownership\"}"}}]}}]}),
        ]),
        sse_stream(&[
            serde_json::json!({"choices":[{"delta":{"content":"Rust ownership explained."}}]}),
        ]),
    ];
    let stub = StubLlm {
        bodies: Arc::new(Mutex::new(Vec::new())),
        replies: Arc::new(Mutex::new(replies)),
    };
    let llm_addr = spawn_stub_llm(stub.clone()).await;

    let search_hits = Arc::new(Mutex::new(Vec::<String>::new()));
    let search_hits_in_handler = search_hits.clone();
    let search_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let search_addr = search_listener.local_addr().unwrap();
    let search_app = axum::Router::new().route(
        "/search",
        axum::routing::get(move |raw_query: axum::extract::RawQuery| {
            let search_hits = search_hits_in_handler.clone();
            async move {
                if let Some(raw) = raw_query.0.as_deref() {
                    search_hits.lock().unwrap().push(raw.to_owned());
                }
                axum::Json(serde_json::json!({
                    "results": [{"title": "Ownership", "url": "https://doc.rust/lang", "content": "Rust ownership rules."}]
                }))
            }
        }),
    );
    tokio::spawn(async move { axum::serve(search_listener, search_app).await.unwrap() });

    let mut config = test_config(llm_addr);
    config.web_search_url = Some(format!("http://{search_addr}/search"));
    let app = router(config);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let mut request = user_query_request("explain rust ownership");
    if let Some(settings) = request.settings.as_mut() {
        settings.web_search_enabled = true;
    }

    let response = post_request(addr, "/ai/multi-agent", &request, None)
        .send()
        .await
        .unwrap();
    let events = decode_events(&response.text().await.unwrap());
    let all_actions: Vec<_> = events.iter().flat_map(actions_of).collect();

    // The client sees a WebSearch message, not a tool call to execute.
    let web_search_message = all_actions
        .iter()
        .find_map(|action| match action {
            api::client_action::Action::AddMessagesToTask(add) => {
                add.messages
                    .iter()
                    .find_map(|message| match &message.message {
                        Some(api::message::Message::WebSearch(search)) => Some(search),
                        _ => None,
                    })
            }
            _ => None,
        })
        .expect("expected a WebSearch message action");
    match &web_search_message.status.as_ref().unwrap().r#type {
        Some(api::message::web_search::status::Type::Success(success)) => {
            assert_eq!(success.query, "rust ownership");
            assert_eq!(success.pages[0].title, "Ownership");
        }
        other => panic!("unexpected web search status: {other:?}"),
    }
    assert!(
        !all_actions.iter().any(|action| matches!(
            action,
            api::client_action::Action::AddMessagesToTask(add)
                if add.messages.iter().any(|message| matches!(
                    &message.message,
                    Some(api::message::Message::ToolCall(_))
                ))
        )),
        "web_search must not leak to the client as a tool call"
    );

    // The search endpoint was queried, and the follow-up LLM round received
    // the results as a tool message.
    assert!(
        search_hits
            .lock()
            .unwrap()
            .first()
            .is_some_and(|raw| raw.starts_with("q=rust+ownership")),
        "search hits: {:?}",
        search_hits.lock().unwrap()
    );
    let bodies = stub.bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 2);
    let tool_messages: Vec<&Value> = bodies[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["role"] == "tool")
        .collect();
    assert_eq!(tool_messages.len(), 1);
    assert!(
        tool_messages[0]["content"]
            .as_str()
            .is_some_and(|content| content.contains("Rust ownership rules.")),
        "follow-up call should carry the search results"
    );
    assert!(matches!(
        &events.last().unwrap().r#type,
        Some(api::response_event::Type::Finished(finished))
            if matches!(&finished.reason, Some(api::response_event::stream_finished::Reason::Done(_)))
    ));
}

#[tokio::test]
async fn relevant_files_ranks_the_outline() {
    let stub = StubLlm {
        bodies: Arc::new(Mutex::new(Vec::new())),
        replies: Arc::new(Mutex::new(Vec::new())),
    };
    let llm_addr = spawn_stub_llm(stub).await;
    let app = router(test_config(llm_addr));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let response = reqwest::Client::new()
        .post(format!("http://{addr}/ai/relevant_files"))
        .json(&serde_json::json!({
            "query": "voice recorder session",
            "files": [
                {"path": "src/voice/session.rs", "symbols": "VoiceSession start stop"},
                {"path": "src/payments/mod.rs", "symbols": "Gateway charge"},
            ],
        }))
        .send()
        .await
        .unwrap();
    let reply: Value = response.json().await.unwrap();
    assert_eq!(
        reply["relevant_file_paths"],
        serde_json::json!(["src/voice/session.rs"])
    );
}

#[test]
fn model_catalog_lists_all_configured_models() {
    let config = crate::config::Config {
        llm_models: vec!["big-model".to_owned(), "fast-model".to_owned()],
        ..crate::config::Config::test_default()
    };
    let response = crate::graphql::get_user_response(&config);
    let agent_mode = &response["data"]["user"]["user"]["llms"]["agentMode"];
    assert_eq!(agent_mode["defaultId"], "big-model");
    let choices = agent_mode["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 2);
    assert_eq!(choices[0]["id"], "big-model");
    assert_eq!(choices[1]["id"], "fast-model");
    // Vision is advertised so the client sends screenshots when configured.
    assert_eq!(choices[0]["visionSupported"], true);
}

#[tokio::test]
async fn auth_introspection_uses_the_external_auth_service() {
    let accept = Arc::new(Mutex::new(true));
    let accept_in_handler = accept.clone();
    let auth_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let auth_addr = auth_listener.local_addr().unwrap();
    let auth_app = axum::Router::new().route(
        "/introspect",
        axum::routing::post(move |body: String| {
            let accept = accept_in_handler.clone();
            async move {
                // The external auth service sees the token and decides.
                assert!(body.contains("\"token\""));
                if *accept.lock().unwrap() {
                    axum::http::StatusCode::OK
                } else {
                    axum::http::StatusCode::UNAUTHORIZED
                }
            }
        }),
    );
    tokio::spawn(async move { axum::serve(auth_listener, auth_app).await.unwrap() });

    let stub = StubLlm {
        bodies: Arc::new(Mutex::new(Vec::new())),
        replies: Arc::new(Mutex::new(Vec::new())),
    };
    let llm_addr = spawn_stub_llm(stub).await;
    let mut config = test_config(llm_addr);
    config.auth_introspect_url = Some(format!("http://{auth_addr}/introspect"));
    let app = router(config);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let request = user_query_request("hi");
    let client = reqwest::Client::new();
    let url = format!("http://{addr}/ai/multi-agent");
    let body = request.encode_to_vec();

    // Accepted by the external auth service.
    let response = client
        .post(&url)
        .header("Authorization", "Bearer user-token-from-foundry")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());

    // Rejected by the external auth service.
    *accept.lock().unwrap() = false;
    let response = client
        .post(&url)
        .header("Authorization", "Bearer user-token-from-foundry")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);

    // No token at all is rejected without consulting the auth service.
    let response = client.post(&url).body("").send().await.unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn metrics_endpoint_reports_counters() {
    let replies = vec![sse_stream(&[
        serde_json::json!({"choices":[{"delta":{"content":"ok"}}]}),
        serde_json::json!({"choices":[{"finish_reason":"stop"}]}),
    ])];
    let stub = StubLlm {
        bodies: Arc::new(Mutex::new(Vec::new())),
        replies: Arc::new(Mutex::new(replies)),
    };
    let llm_addr = spawn_stub_llm(stub.clone()).await;
    let app = router(test_config(llm_addr));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let response = post_request(addr, "/ai/multi-agent", &user_query_request("hi"), None)
        .send()
        .await
        .unwrap();
    let _ = response.text().await;

    let client = reqwest::Client::new();
    let metrics = client
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        metrics.contains("selfhost_agent_requests_total 1"),
        "{metrics}"
    );
    assert!(
        metrics.contains("selfhost_llm_output_tokens_total"),
        "token counters should exist: {metrics}"
    );
}
