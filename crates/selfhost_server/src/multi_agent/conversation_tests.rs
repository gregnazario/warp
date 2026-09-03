use warp_multi_agent_api::request::input::user_inputs::user_input::Input as UserInputKind;

use super::*;

fn request_with_input(query: &str, context: Option<api::InputContext>) -> api::Request {
    api::Request {
        input: Some(api::request::Input {
            context,
            r#type: Some(api::request::input::Type::UserInputs(
                api::request::input::UserInputs {
                    inputs: vec![api::request::input::user_inputs::UserInput {
                        input: Some(UserInputKind::UserQuery(api::request::input::UserQuery {
                            query: query.to_owned(),
                            referenced_attachments: Default::default(),
                            mode: None,
                            intended_agent: api::AgentType::Unknown.into(),
                        })),
                    }],
                },
            )),
        }),
        ..Default::default()
    }
}

#[test]
fn first_request_translates_to_a_single_user_message() {
    let request = request_with_input("list the files here", None);
    let messages = translate(&request);
    assert_eq!(messages.len(), 1);
    match &messages[0] {
        LlmMessage::User { text, images } => {
            assert_eq!(text, "list the files here");
            assert!(images.is_empty());
        }
        other => panic!("expected user message, got {other:?}"),
    }
}

#[test]
fn system_prompt_includes_environment_context() {
    let mut request = request_with_input("hi", None);
    if let Some(input) = request.input.as_mut() {
        input.context = Some(api::InputContext {
            directory: Some(api::input_context::Directory {
                pwd: "/repo".to_owned(),
                home: "/home/u".to_owned(),
                pwd_file_symbols_indexed: false,
            }),
            operating_system: Some(api::input_context::OperatingSystem {
                platform: "MacOS".to_owned(),
                distribution: String::new(),
            }),
            ..Default::default()
        });
    }
    let prompt = system_prompt(None, &request);
    assert!(prompt.contains("Working directory: /repo"));
    assert!(prompt.contains("OS: MacOS"));
}

#[test]
fn tool_call_history_becomes_assistant_and_tool_result_messages() {
    let request = api::Request {
        task_context: Some(api::request::TaskContext {
            tasks: vec![api::Task {
                id: "task1".to_owned(),
                messages: vec![
                    api::Message {
                        id: "m1".to_owned(),
                        message: Some(api::message::Message::UserQuery(api::message::UserQuery {
                            query: "list the files".to_owned(),
                            ..Default::default()
                        })),
                        ..Default::default()
                    },
                    api::Message {
                        id: "m2".to_owned(),
                        message: Some(api::message::Message::ToolCall(api::message::ToolCall {
                            tool_call_id: "call1".to_owned(),
                            tool: Some(api::message::tool_call::Tool::RunShellCommand(
                                api::message::tool_call::RunShellCommand {
                                    command: "ls".to_owned(),
                                    ..Default::default()
                                },
                            )),
                        })),
                        ..Default::default()
                    },
                    api::Message {
                        id: "m3".to_owned(),
                        message: Some(api::message::Message::ToolCallResult(
                            api::message::ToolCallResult {
                                tool_call_id: "call1".to_owned(),
                                result: Some(api::message::tool_call_result::Result::RunShellCommand(
                                    api::RunShellCommandResult {
                                        result: Some(api::run_shell_command_result::Result::CommandFinished(
                                            api::ShellCommandFinished {
                                                output: "a.txt\nb.txt".to_owned(),
                                                exit_code: 0,
                                                ..Default::default()
                                            },
                                        )),
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
        }),
        input: Some(api::request::Input {
            r#type: Some(api::request::input::Type::UserInputs(
                api::request::input::UserInputs {
                    inputs: vec![api::request::input::user_inputs::UserInput {
                        input: Some(UserInputKind::ToolCallResult(
                            api::request::input::ToolCallResult {
                                tool_call_id: "call1".to_owned(),
                                result: Some(api::request::input::tool_call_result::Result::RunShellCommand(
                                    api::RunShellCommandResult {
                                        result: Some(api::run_shell_command_result::Result::CommandFinished(
                                            api::ShellCommandFinished {
                                                output: "a.txt\nb.txt".to_owned(),
                                                exit_code: 0,
                                                ..Default::default()
                                            },
                                        )),
                                        ..Default::default()
                                    },
                                )),
                            },
                        )),
                    }],
                },
            )),
            ..Default::default()
        }),
        ..Default::default()
    };

    let messages = translate(&request);
    // User query, assistant tool call, tool result — the input-side duplicate
    // of the tool result must not add a second one.
    assert_eq!(messages.len(), 3, "messages: {messages:?}");
    match &messages[1] {
        LlmMessage::Assistant { tool_calls, .. } => {
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].id, "call1");
            assert_eq!(tool_calls[0].name, "run_shell_command");
            assert_eq!(tool_calls[0].arguments["command"], "ls");
        }
        other => panic!("expected assistant tool call, got {other:?}"),
    }
    match &messages[2] {
        LlmMessage::ToolResult { tool_call_id, text } => {
            assert_eq!(tool_call_id, "call1");
            assert!(text.contains("a.txt"));
        }
        other => panic!("expected tool result, got {other:?}"),
    }
}

#[test]
fn unrecorded_input_tool_result_is_appended() {
    let request = api::Request {
        input: Some(api::request::Input {
            r#type: Some(api::request::input::Type::UserInputs(
                api::request::input::UserInputs {
                    inputs: vec![api::request::input::user_inputs::UserInput {
                        input: Some(UserInputKind::ToolCallResult(
                            api::request::input::ToolCallResult {
                                tool_call_id: "call9".to_owned(),
                                result: None,
                            },
                        )),
                    }],
                },
            )),
            ..Default::default()
        }),
        ..Default::default()
    };
    let messages = translate(&request);
    assert_eq!(messages.len(), 1);
    match &messages[0] {
        LlmMessage::ToolResult { tool_call_id, .. } => assert_eq!(tool_call_id, "call9"),
        other => panic!("expected tool result, got {other:?}"),
    }
}

#[test]
fn repeated_user_query_in_input_is_not_duplicated() {
    let request = request_with_input("same query", None);
    // Both the input and the task state contain the query, as they will once
    // the server has echoed it into a message.
    let mut request = request;
    if let Some(task_context) = request.task_context.as_mut() {
        task_context.tasks.push(api::Task {
            id: "task1".to_owned(),
            messages: vec![api::Message {
                id: "m1".to_owned(),
                message: Some(api::message::Message::UserQuery(api::message::UserQuery {
                    query: "same query".to_owned(),
                    ..Default::default()
                })),
                ..Default::default()
            }],
            ..Default::default()
        });
    }
    let messages = translate(&request);
    assert_eq!(messages.len(), 1);
}

#[test]
fn input_context_images_attach_to_the_latest_user_message() {
    let mut request = request_with_input("what is in this screenshot", None);
    if let Some(input) = request.input.as_mut() {
        input.context = Some(api::InputContext {
            images: vec![api::input_context::Image {
                data: vec![1, 2, 3],
                mime_type: "image/png".to_owned(),
            }],
            ..Default::default()
        });
    }
    let messages = translate(&request);
    match messages.last() {
        Some(LlmMessage::User { text, images }) => {
            assert_eq!(text, "what is in this screenshot");
            assert_eq!(images.len(), 1);
            assert_eq!(images[0].data, vec![1, 2, 3]);
            assert_eq!(images[0].mime_type, "image/png");
        }
        other => panic!("expected user message with images, got {other:?}"),
    }
}

#[test]
fn images_without_a_user_query_get_a_synthetic_message() {
    let request = api::Request {
        input: Some(api::request::Input {
            context: Some(api::InputContext {
                images: vec![api::input_context::Image {
                    data: vec![9],
                    mime_type: "image/jpeg".to_owned(),
                }],
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let messages = translate(&request);
    assert_eq!(messages.len(), 1);
    match &messages[0] {
        LlmMessage::User { images, .. } => assert_eq!(images[0].mime_type, "image/jpeg"),
        other => panic!("expected user message, got {other:?}"),
    }
}
