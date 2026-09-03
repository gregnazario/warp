use warp_multi_agent_api as api;

use super::llm::{LlmImage, LlmMessage};

/// Translates a multi-agent `Request` into provider-neutral LLM messages.
///
/// The request's task state is the source of truth for prior history; input
/// entries that are not yet reflected in the task state (the client may or may
/// not have recorded them optimistically) are appended idempotently.
pub fn translate(request: &api::Request) -> Vec<LlmMessage> {
    let mut messages = Vec::new();
    if let Some(task_context) = &request.task_context {
        for task in &task_context.tasks {
            for message in &task.messages {
                translate_message(message, &mut messages);
            }
        }
    }

    if let Some(input) = &request.input {
        append_pending_inputs(input, &mut messages);
        attach_input_images(input, &mut messages);
    }

    messages
}

/// Attaches the images from the input's context (e.g. screenshots) to the
/// latest user message, or to a synthetic one when the input carried no query.
fn attach_input_images(input: &api::request::Input, messages: &mut Vec<LlmMessage>) {
    let Some(context) = input.context.as_ref() else {
        return;
    };
    if context.images.is_empty() {
        return;
    }
    let images: Vec<LlmImage> = context
        .images
        .iter()
        .filter(|image| !image.data.is_empty())
        .map(|image| LlmImage {
            data: image.data.clone(),
            mime_type: if image.mime_type.is_empty() {
                "image/png".to_owned()
            } else {
                image.mime_type.clone()
            },
        })
        .collect();
    if images.is_empty() {
        return;
    }
    if let Some(LlmMessage::User {
        images: existing, ..
    }) = messages.last_mut()
    {
        existing.extend(images);
    } else {
        messages.push(LlmMessage::User {
            text: "The user attached the images above.".to_owned(),
            images,
        });
    }
}

fn translate_message(message: &api::Message, out: &mut Vec<LlmMessage>) {
    use api::message::Message as MessageKind;
    let Some(kind) = message.message.as_ref() else {
        return;
    };
    match kind {
        MessageKind::UserQuery(user_query) => {
            out.push(LlmMessage::User {
                text: render_user_query(&user_query.query, user_query.context.as_ref()),
                images: Vec::new(),
            });
        }
        MessageKind::SystemQuery(system_query) => {
            if let Some(text) = render_system_query(system_query) {
                out.push(LlmMessage::User {
                    text: format!("[system] {text}"),
                    images: Vec::new(),
                });
            }
        }
        MessageKind::AgentOutput(output) => {
            out.push(LlmMessage::Assistant {
                text: output.text.clone(),
                tool_calls: Vec::new(),
            });
        }
        MessageKind::AgentReasoning(_)
        | MessageKind::Summarization(_)
        | MessageKind::UpdateTodos(_)
        | MessageKind::ModelUsed(_)
        | MessageKind::ServerEvent(_)
        | MessageKind::DebugOutput(_)
        | MessageKind::ArtifactEvent(_)
        | MessageKind::CodeReview(_)
        | MessageKind::UpdateReviewComments(_)
        | MessageKind::WebSearch(_)
        | MessageKind::WebFetch(_)
        | MessageKind::MessagesReceivedFromAgents(_)
        | MessageKind::EventsFromAgents(_)
        | MessageKind::PassiveSuggestionResult(_)
        | MessageKind::OrchestrationConfigSnapshot(_) => {}
        MessageKind::InvokeSkill(invoke) => {
            let (name, instructions) = invoke
                .skill
                .as_ref()
                .map(|skill| {
                    let name = skill
                        .descriptor
                        .as_ref()
                        .map(|descriptor| descriptor.name.clone())
                        .unwrap_or_default();
                    let content = skill
                        .content
                        .as_ref()
                        .map(|content| content.content.clone())
                        .unwrap_or_default();
                    (name, content)
                })
                .unwrap_or_default();
            let mut text = format!("[system] The user invoked the skill '{name}'.");
            if !instructions.is_empty() {
                text.push_str(&format!(" Follow its instructions:\n{instructions}"));
            }
            if let Some(query) = &invoke.user_query {
                text.push_str("\n\n");
                text.push_str(&render_user_query(&query.query, query.context.as_ref()));
            }
            out.push(LlmMessage::User {
                text,
                images: Vec::new(),
            });
        }
        MessageKind::ToolCall(tool_call) => {
            if let Some(tool) = tool_call.tool.as_ref()
                && let Some((name, arguments)) = tool_call_to_json(tool)
            {
                out.push(LlmMessage::Assistant {
                    text: String::new(),
                    tool_calls: vec![super::llm::ToolCallOut {
                        id: tool_call.tool_call_id.clone(),
                        name,
                        arguments,
                    }],
                });
            }
        }
        MessageKind::ToolCallResult(result) => {
            out.push(LlmMessage::ToolResult {
                tool_call_id: result.tool_call_id.clone(),
                text: summarize_tool_call_result(result),
            });
        }
    }
}

fn render_user_query(query: &str, context: Option<&api::InputContext>) -> String {
    let mut text = query.to_owned();
    if let Some(rendered) = context.and_then(render_input_context_summary) {
        text.push_str("\n\n---\nUser context:\n");
        text.push_str(&rendered);
    }
    text
}

fn render_system_query(system_query: &api::message::SystemQuery) -> Option<String> {
    use api::message::system_query::Type as SystemQueryKind;
    match system_query.r#type.as_ref()? {
        SystemQueryKind::AutoCodeDiff(query) => Some(query.query.clone()),
        SystemQueryKind::ResumeConversation(_) => {
            Some("The user resumed this conversation.".to_owned())
        }
        SystemQueryKind::CreateNewProject(query) => Some(query.query.clone()),
        SystemQueryKind::CloneRepository(query) => {
            Some(format!("Clone the repository at {}.", query.url))
        }
        SystemQueryKind::SummarizeConversation(query) => {
            let instructions = if query.prompt.is_empty() {
                String::new()
            } else {
                format!(" Focus on: {}", query.prompt)
            };
            Some(format!("Summarize this conversation.{instructions}"))
        }
        SystemQueryKind::GeneratePassiveSuggestions(_)
        | SystemQueryKind::FetchReviewComments(_)
        | SystemQueryKind::HandoffRehydration(_) => None,
    }
}

fn append_pending_inputs(input: &api::request::Input, messages: &mut Vec<LlmMessage>) {
    let Some(api::request::input::Type::UserInputs(user_inputs)) = input.r#type.as_ref() else {
        return;
    };
    let input_context = input.context.as_ref();
    for entry in &user_inputs.inputs {
        match entry.input.as_ref() {
            Some(api::request::input::user_inputs::user_input::Input::ToolCallResult(result)) => {
                let already_recorded = messages.iter().any(|message| {
                    matches!(
                        message,
                        LlmMessage::ToolResult { tool_call_id, .. }
                            if *tool_call_id == result.tool_call_id
                    )
                });
                if !already_recorded {
                    let text = result
                        .result
                        .as_ref()
                        .and_then(to_message_side_result)
                        .map(|kind| summarize_result_kind(&kind))
                        .unwrap_or_else(|| "(result not summarized)".to_owned());
                    messages.push(LlmMessage::ToolResult {
                        tool_call_id: result.tool_call_id.clone(),
                        text,
                    });
                }
            }
            Some(api::request::input::user_inputs::user_input::Input::UserQuery(user_query)) => {
                let query = render_user_query(&user_query.query, input_context);
                let already_recorded = messages.iter().any(
                    |message| matches!(message, LlmMessage::User { text, .. } if *text == query),
                );
                if !already_recorded {
                    messages.push(LlmMessage::User {
                        text: query,
                        images: Vec::new(),
                    });
                }
            }
            Some(
                api::request::input::user_inputs::user_input::Input::CliAgentUserQuery(_)
                | api::request::input::user_inputs::user_input::Input::MessagesReceivedFromAgents(_)
                | api::request::input::user_inputs::user_input::Input::EventsFromAgents(_)
                | api::request::input::user_inputs::user_input::Input::PassiveSuggestionResult(_)
                | api::request::input::user_inputs::user_input::Input::OrchestrationConfigUpdate(_)
                | api::request::input::user_inputs::user_input::Input::ConversationHandoff(_),
            ) => {}
            None => {}
        }
    }
}

/// Maps a proto tool call to its tool name and JSON arguments.
#[allow(deprecated)] // legacy FileGlob variants must stay exhaustively matched
fn tool_call_to_json(tool: &api::message::tool_call::Tool) -> Option<(String, serde_json::Value)> {
    use api::message::tool_call::Tool as ToolKind;
    let value = match tool {
        ToolKind::RunShellCommand(call) => serde_json::json!({"command": call.command}),
        ToolKind::ReadFiles(call) => {
            serde_json::json!({"files": call.files.iter().map(|file| serde_json::json!({"name": file.name})).collect::<Vec<_>>()})
        }
        ToolKind::Grep(call) => {
            serde_json::json!({"queries": call.queries, "path": call.path})
        }
        ToolKind::FileGlobV2(call) => {
            serde_json::json!({"patterns": call.patterns, "search_dir": call.search_dir})
        }
        ToolKind::ApplyFileDiffs(call) => serde_json::json!({
            "summary": call.summary,
            "diffs": call.diffs.iter().map(|diff| serde_json::json!({
                "file_path": diff.file_path,
                "search": diff.search,
                "replace": diff.replace,
            })).collect::<Vec<_>>(),
            "new_files": call.new_files.iter().map(|file| serde_json::json!({
                "file_path": file.file_path,
                "content": file.content,
            })).collect::<Vec<_>>(),
        }),
        ToolKind::WriteToLongRunningShellCommand(call) => serde_json::json!({
            "input": String::from_utf8_lossy(&call.input),
            "command_id": call.command_id,
        }),
        ToolKind::ReadShellCommandOutput(call) => {
            serde_json::json!({"command_id": call.command_id})
        }
        ToolKind::AskUserQuestion(call) => serde_json::json!({
            "questions": call.questions.iter().map(|question| serde_json::json!({
                "question": question.question,
                "options": match &question.question_type {
                    Some(api::ask_user_question::question::QuestionType::MultipleChoice(multiple_choice)) => {
                        multiple_choice.options.iter().map(|option| option.label.clone()).collect::<Vec<_>>()
                    }
                    _ => Vec::<String>::new(),
                },
            })).collect::<Vec<_>>(),
        }),
        ToolKind::SuggestPrompt(call) => match &call.display_mode {
            Some(api::message::tool_call::suggest_prompt::DisplayMode::PromptChip(chip)) => {
                serde_json::json!({"prompt": chip.prompt, "label": chip.label})
            }
            Some(api::message::tool_call::suggest_prompt::DisplayMode::InlineQueryBanner(
                banner,
            )) => {
                serde_json::json!({"prompt": banner.query})
            }
            None => return None,
        },
        ToolKind::CallMcpTool(call) => serde_json::json!({
            "name": call.name,
            "args": call.args
                .as_ref()
                .map(super::tools::prost_struct_to_json)
                .unwrap_or_else(|| serde_json::json!({})),
        }),
        ToolKind::ReadMcpResource(call) => serde_json::json!({"uri": call.uri}),

        ToolKind::Server(_)
        | ToolKind::SearchCodebase(_)
        | ToolKind::SuggestPlan(_)
        | ToolKind::SuggestCreatePlan(_)
        | ToolKind::FileGlob(_)
        | ToolKind::SuggestNewConversation(_)
        | ToolKind::OpenCodeReview(_)
        | ToolKind::InitProject(_)
        | ToolKind::Subagent(_)
        | ToolKind::ReadDocuments(_)
        | ToolKind::EditDocuments(_)
        | ToolKind::CreateDocuments(_)
        | ToolKind::UseComputer(_)
        | ToolKind::InsertReviewComments(_)
        | ToolKind::ReadSkill(_)
        | ToolKind::RequestComputerUse(_)
        | ToolKind::FetchConversation(_)
        | ToolKind::SendMessageToAgent(_)
        | ToolKind::TransferShellCommandControlToUser(_)
        | ToolKind::UploadFileArtifact(_)
        | ToolKind::RunAgents(_)
        | ToolKind::WaitForEvents(_)
        | ToolKind::StartRecording(_)
        | ToolKind::StopRecording(_) => return None,
    };
    Some((tool_name(tool), value))
}

/// The `run_shell_command`-style tool name for a proto tool call.
#[allow(deprecated)] // legacy FileGlob variants must stay exhaustively matched
fn tool_name(tool: &api::message::tool_call::Tool) -> String {
    let name = match tool {
        api::message::tool_call::Tool::RunShellCommand(_) => "run_shell_command",
        api::message::tool_call::Tool::ReadFiles(_) => "read_files",
        api::message::tool_call::Tool::Grep(_) => "grep",
        api::message::tool_call::Tool::FileGlobV2(_) => "file_glob",
        api::message::tool_call::Tool::ApplyFileDiffs(_) => "apply_file_diffs",
        api::message::tool_call::Tool::WriteToLongRunningShellCommand(_) => {
            "write_to_long_running_command"
        }
        api::message::tool_call::Tool::ReadShellCommandOutput(_) => "read_command_output",
        api::message::tool_call::Tool::AskUserQuestion(_) => "ask_user_question",
        api::message::tool_call::Tool::SuggestPrompt(_) => "suggest_prompt",
        api::message::tool_call::Tool::CallMcpTool(_) => "call_mcp_tool",
        api::message::tool_call::Tool::ReadMcpResource(_) => "read_mcp_resource",
        api::message::tool_call::Tool::Server(_) => "server",
        api::message::tool_call::Tool::SearchCodebase(_) => "search_codebase",
        api::message::tool_call::Tool::SuggestPlan(_) => "suggest_plan",
        api::message::tool_call::Tool::SuggestCreatePlan(_) => "suggest_create_plan",
        api::message::tool_call::Tool::FileGlob(_) => "file_glob",
        api::message::tool_call::Tool::SuggestNewConversation(_) => "suggest_new_conversation",
        api::message::tool_call::Tool::OpenCodeReview(_) => "open_code_review",
        api::message::tool_call::Tool::InitProject(_) => "init_project",
        api::message::tool_call::Tool::Subagent(_) => "subagent",
        api::message::tool_call::Tool::ReadDocuments(_) => "read_documents",
        api::message::tool_call::Tool::EditDocuments(_) => "edit_documents",
        api::message::tool_call::Tool::CreateDocuments(_) => "create_documents",
        api::message::tool_call::Tool::UseComputer(_) => "use_computer",
        api::message::tool_call::Tool::InsertReviewComments(_) => "insert_review_comments",
        api::message::tool_call::Tool::ReadSkill(_) => "read_skill",
        api::message::tool_call::Tool::RequestComputerUse(_) => "request_computer_use",
        api::message::tool_call::Tool::FetchConversation(_) => "fetch_conversation",
        api::message::tool_call::Tool::SendMessageToAgent(_) => "send_message_to_agent",
        api::message::tool_call::Tool::TransferShellCommandControlToUser(_) => {
            "transfer_shell_command_control_to_user"
        }
        api::message::tool_call::Tool::UploadFileArtifact(_) => "upload_file_artifact",
        api::message::tool_call::Tool::RunAgents(_) => "run_agents",
        api::message::tool_call::Tool::WaitForEvents(_) => "wait_for_events",
        api::message::tool_call::Tool::StartRecording(_) => "start_recording",
        api::message::tool_call::Tool::StopRecording(_) => "stop_recording",
    };
    name.to_owned()
}

/// Renders a tool result as compact text for the LLM.
fn summarize_tool_call_result(result: &api::message::ToolCallResult) -> String {
    let Some(kind) = result.result.as_ref() else {
        return "(no result recorded)".to_owned();
    };
    summarize_result_kind(kind)
}

/// Maps an input-side tool result onto its message-side equivalent so both
/// paths share one summarizer. The two proto types wrap the same inner result
/// messages, so variants move across unchanged; uncommon variants return None.
fn to_message_side_result(
    result: &api::request::input::tool_call_result::Result,
) -> Option<api::message::tool_call_result::Result> {
    use api::message::tool_call_result::Result as Out;
    use api::request::input::tool_call_result::Result as In;
    Some(match result {
        In::RunShellCommand(inner) => Out::RunShellCommand(inner.clone()),
        In::ReadFiles(inner) => Out::ReadFiles(inner.clone()),
        In::ApplyFileDiffs(inner) => Out::ApplyFileDiffs(inner.clone()),
        In::Grep(inner) => Out::Grep(inner.clone()),
        In::FileGlobV2(inner) => Out::FileGlobV2(inner.clone()),
        In::WriteToLongRunningShellCommand(inner) => {
            Out::WriteToLongRunningShellCommand(inner.clone())
        }
        In::ReadShellCommandOutput(inner) => Out::ReadShellCommandOutput(inner.clone()),
        In::AskUserQuestion(inner) => Out::AskUserQuestion(inner.clone()),
        In::CallMcpTool(inner) => Out::CallMcpTool(inner.clone()),
        In::ReadMcpResource(inner) => Out::ReadMcpResource(inner.clone()),
        In::SuggestPrompt(inner) => Out::SuggestPrompt(*inner),
        _ => return None,
    })
}

/// Renders one message-side tool result kind as compact text for the LLM.
#[allow(deprecated)] // legacy FileGlob variants must stay exhaustively matched
fn summarize_result_kind(result: &api::message::tool_call_result::Result) -> String {
    use api::message::tool_call_result::Result as ResultKind;
    match result {
        ResultKind::RunShellCommand(run) => match run.result.as_ref() {
            Some(api::run_shell_command_result::Result::CommandFinished(finished)) => {
                format!(
                    "Command finished with exit code {}:\n{}",
                    finished.exit_code, finished.output
                )
            }
            Some(api::run_shell_command_result::Result::LongRunningCommandSnapshot(snapshot)) => {
                format!(
                    "Command is long-running. Latest output:\n{}",
                    snapshot.output
                )
            }
            Some(api::run_shell_command_result::Result::PermissionDenied(_)) => {
                "The user denied permission to run this command.".to_owned()
            }
            None => "(empty command result)".to_owned(),
        },
        ResultKind::ReadFiles(read) => match read.result.as_ref() {
            Some(api::read_files_result::Result::TextFilesSuccess(success)) => success
                .files
                .iter()
                .map(|file| format!("--- {} ---\n{}", file.file_path, file.content))
                .collect::<Vec<_>>()
                .join("\n\n"),
            Some(api::read_files_result::Result::AnyFilesSuccess(success)) => success
                .files
                .iter()
                .map(|file| match file.content.as_ref() {
                    Some(api::any_file_content::Content::TextContent(text)) => {
                        format!("--- {} ---\n{}", text.file_path, text.content)
                    }
                    Some(api::any_file_content::Content::BinaryContent(binary)) => {
                        format!(
                            "--- {} --- ({} bytes of data)",
                            binary.file_path,
                            binary.data.len()
                        )
                    }
                    None => "(empty file)".to_owned(),
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
            Some(api::read_files_result::Result::Error(error)) => {
                format!("Failed to read files: {}", error.message)
            }
            None => "(no files read)".to_owned(),
        },
        ResultKind::Grep(grep) => match grep.result.as_ref() {
            Some(api::grep_result::Result::Success(success)) => success
                .matched_files
                .iter()
                .map(|file| {
                    let lines = file
                        .matched_lines
                        .iter()
                        .map(|line| line.line_number.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("{}: lines [{lines}]", file.file_path)
                })
                .collect::<Vec<_>>()
                .join("\n"),
            Some(api::grep_result::Result::Error(error)) => {
                format!("Grep failed: {}", error.message)
            }
            None => "(no matches)".to_owned(),
        },
        ResultKind::FileGlobV2(glob) => match glob.result.as_ref() {
            Some(api::file_glob_v2_result::Result::Success(success)) => success
                .matched_files
                .iter()
                .map(|match_| match_.file_path.clone())
                .collect::<Vec<_>>()
                .join("\n"),
            Some(api::file_glob_v2_result::Result::Error(error)) => {
                format!("File glob failed: {}", error.message)
            }
            None => "(no matches)".to_owned(),
        },
        ResultKind::ApplyFileDiffs(apply) => match apply.result.as_ref() {
            Some(api::apply_file_diffs_result::Result::Success(success)) => {
                let mut files = success
                    .updated_files_v2
                    .iter()
                    .filter_map(|updated| updated.file.as_ref().map(|f| f.file_path.clone()))
                    .collect::<Vec<_>>();
                files.extend(
                    success
                        .deleted_files
                        .iter()
                        .map(|deleted| format!("{} (deleted)", deleted.file_path)),
                );
                format!("Applied changes to: {}", files.join(", "))
            }
            Some(api::apply_file_diffs_result::Result::Error(error)) => {
                format!("Failed to apply diffs: {}", error.message)
            }
            None => "(no changes applied)".to_owned(),
        },
        ResultKind::WriteToLongRunningShellCommand(write) => match write.result.as_ref() {
            Some(api::write_to_long_running_shell_command_result::Result::CommandFinished(
                finished,
            )) => format!(
                "Command finished with exit code {}:\n{}",
                finished.exit_code, finished.output
            ),
            Some(
                api::write_to_long_running_shell_command_result::Result::LongRunningCommandSnapshot(
                    snapshot,
                ),
            ) => format!("Command still running. Latest output:\n{}", snapshot.output),
            Some(api::write_to_long_running_shell_command_result::Result::Error(_)) => {
                "The write failed because the command could not be found.".to_owned()
            }
            None => "(no result)".to_owned(),
        },
        ResultKind::ReadShellCommandOutput(read) => match read.result.as_ref() {
            Some(api::read_shell_command_output_result::Result::CommandFinished(finished)) => {
                format!(
                    "Command finished with exit code {}:\n{}",
                    finished.exit_code, finished.output
                )
            }
            Some(api::read_shell_command_output_result::Result::LongRunningCommandSnapshot(
                snapshot,
            )) => format!("Command still running. Latest output:\n{}", snapshot.output),
            Some(api::read_shell_command_output_result::Result::Error(_)) => {
                "The command could not be found.".to_owned()
            }
            None => "(no result)".to_owned(),
        },
        ResultKind::AskUserQuestion(answer) => match answer.result.as_ref() {
            Some(api::ask_user_question_result::Result::Success(success)) => success
                .answers
                .iter()
                .map(|item| match item.answer.as_ref() {
                    Some(api::ask_user_question_result::answer_item::Answer::MultipleChoice(
                        choice,
                    )) => {
                        let mut answer = choice.selected_options.join(", ");
                        if !choice.other_text.is_empty() {
                            if !answer.is_empty() {
                                answer.push_str(", ");
                            }
                            answer.push_str(&choice.other_text);
                        }
                        format!("{}: {}", item.question_id, answer)
                    }
                    Some(api::ask_user_question_result::answer_item::Answer::Skipped(_)) => {
                        format!("{}: (skipped)", item.question_id)
                    }
                    None => format!("{}: (no answer)", item.question_id),
                })
                .collect::<Vec<_>>()
                .join("\n"),
            Some(api::ask_user_question_result::Result::Error(error)) => {
                format!("Questioning the user failed: {}", error.message)
            }
            None => "(no answers)".to_owned(),
        },
        ResultKind::CallMcpTool(call) => match call.result.as_ref() {
            Some(api::call_mcp_tool_result::Result::Success(success)) => success
                .results
                .iter()
                .map(|result| match result.result.as_ref() {
                    Some(api::call_mcp_tool_result::success::result::Result::Text(text)) => {
                        text.text.clone()
                    }
                    Some(api::call_mcp_tool_result::success::result::Result::Image(image)) => {
                        format!(
                            "({} bytes of {} image data)",
                            image.data.len(),
                            image.mime_type
                        )
                    }
                    Some(api::call_mcp_tool_result::success::result::Result::Resource(
                        resource,
                    )) => match resource.content_type.as_ref() {
                        Some(api::mcp_resource_content::ContentType::Text(text)) => {
                            text.content.clone()
                        }
                        Some(api::mcp_resource_content::ContentType::Binary(binary)) => format!(
                            "({} bytes of {} binary data)",
                            binary.data.len(),
                            binary.mime_type
                        ),
                        None => "(empty resource)".to_owned(),
                    },
                    None => "(empty)".to_owned(),
                })
                .collect::<Vec<_>>()
                .join("\n"),
            Some(api::call_mcp_tool_result::Result::Error(error)) => {
                format!("MCP tool call failed: {}", error.message)
            }
            None => "(no result)".to_owned(),
        },
        ResultKind::ReadMcpResource(read) => match read.result.as_ref() {
            Some(api::read_mcp_resource_result::Result::Success(success)) => success
                .contents
                .iter()
                .map(|content| match content.content_type.as_ref() {
                    Some(api::mcp_resource_content::ContentType::Text(text)) => {
                        format!("{}:\n{}", content.uri, text.content)
                    }
                    Some(api::mcp_resource_content::ContentType::Binary(binary)) => format!(
                        "{}: ({} bytes of {} binary data)",
                        content.uri,
                        binary.data.len(),
                        binary.mime_type
                    ),
                    None => "(empty resource)".to_owned(),
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
            Some(api::read_mcp_resource_result::Result::Error(error)) => {
                format!("Failed to read MCP resource: {}", error.message)
            }
            None => "(no contents)".to_owned(),
        },
        ResultKind::SuggestPrompt(prompt) => match prompt.result.as_ref() {
            Some(api::suggest_prompt_result::Result::Accepted(_)) => {
                "The user accepted the suggested prompt.".to_owned()
            }
            Some(api::suggest_prompt_result::Result::Rejected(_)) => {
                "The user rejected the suggested prompt.".to_owned()
            }
            None => "(no result)".to_owned(),
        },

        ResultKind::Server(_)
        | ResultKind::SearchCodebase(_)
        | ResultKind::SuggestPlan(_)
        | ResultKind::SuggestCreatePlan(_)
        | ResultKind::FileGlob(_)
        | ResultKind::Cancel(_)
        | ResultKind::SuggestNewConversation(_)
        | ResultKind::OpenCodeReview(_)
        | ResultKind::InitProject(_)
        | ResultKind::Subagent(_)
        | ResultKind::ReadDocuments(_)
        | ResultKind::EditDocuments(_)
        | ResultKind::CreateDocuments(_)
        | ResultKind::UseComputer(_)
        | ResultKind::InsertReviewComments(_)
        | ResultKind::ReadSkill(_)
        | ResultKind::RequestComputerUseResult(_)
        | ResultKind::FetchConversation(_)
        | ResultKind::SendMessageToAgent(_)
        | ResultKind::TransferShellCommandControlToUser(_)
        | ResultKind::UploadFileArtifact(_)
        | ResultKind::RunAgentsResult(_)
        | ResultKind::WaitForEvents(_)
        | ResultKind::StartRecording(_)
        | ResultKind::StopRecording(_) => {
            "(result not summarized by the self-hosted server)".to_owned()
        }
    }
}

/// Builds the system prompt, combining the server default (or override) with
/// the request's client context.
pub fn system_prompt(config_override: Option<&str>, request: &api::Request) -> String {
    let mut prompt = config_override.unwrap_or(DEFAULT_SYSTEM_PROMPT).to_owned();
    if let Some(context) = request
        .input
        .as_ref()
        .and_then(|input| input.context.as_ref())
        && let Some(summary) = render_input_context_summary(context)
    {
        prompt.push_str("\n\n# Environment\n");
        prompt.push_str(&summary);
    }
    prompt
}

fn render_input_context_summary(context: &api::InputContext) -> Option<String> {
    let mut lines = Vec::new();
    if let Some(directory) = &context.directory {
        if !directory.pwd.is_empty() {
            lines.push(format!("Working directory: {}", directory.pwd));
        }
        if !directory.home.is_empty() {
            lines.push(format!("Home directory: {}", directory.home));
        }
    }
    if let Some(os) = &context.operating_system {
        let mut os_line = os.platform.clone();
        if !os.distribution.is_empty() {
            os_line.push_str(&format!(" ({})", os.distribution));
        }
        if !os_line.is_empty() {
            lines.push(format!("OS: {os_line}"));
        }
    }
    if let Some(shell) = &context.shell
        && !shell.name.is_empty()
    {
        lines.push(format!("Shell: {}", shell.name));
    }
    if let Some(git) = &context.git {
        if !git.branch.is_empty() {
            lines.push(format!("Git branch: {}", git.branch));
        } else if !git.head.is_empty() {
            lines.push(format!("Git HEAD: {}", git.head));
        }
    }
    for codebase in &context.codebases {
        lines.push(format!("Codebase: {} ({})", codebase.name, codebase.path));
    }
    for rules in &context.project_rules {
        for rule_file in &rules.active_rule_files {
            lines.push(format!(
                "Project rules from {}:\n{}",
                rule_file.file_path, rule_file.content
            ));
        }
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

/// The default agent prompt. Warp's full prompt lives server-side; this is a
/// compact equivalent for a self-hosted deployment.
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a software engineering agent working in the user's terminal. You can \
run shell commands, read and edit files, and ask the user questions using the \
provided tools. Prefer read-only exploration before making changes, keep the \
user informed with concise explanations, and stop to answer questions or hand \
control back to the user when a task is complete. When editing files, use the \
apply_file_diffs tool with exact search strings that appear in the file.";

#[cfg(test)]
#[path = "conversation_tests.rs"]
mod tests;

/// Renders task messages as plain text for the summarization prompt.
#[allow(deprecated)] // legacy FileGlob variants must stay exhaustively matched
pub fn render_messages_for_summary(messages: &[api::Message]) -> String {
    let mut text = String::new();
    for message in messages {
        match &message.message {
            Some(api::message::Message::UserQuery(query)) => {
                text.push_str(&format!("user: {}\n", query.query));
            }
            Some(api::message::Message::SystemQuery(query)) => {
                if let Some(rendered) = render_system_query(query) {
                    text.push_str(&format!("system: {rendered}\n"));
                }
            }
            Some(api::message::Message::AgentOutput(output)) if !output.text.is_empty() => {
                text.push_str(&format!("assistant: {}\n", output.text));
            }
            Some(api::message::Message::ToolCall(call)) => {
                if let Some((name, arguments)) = call.tool.as_ref().and_then(tool_call_to_json) {
                    text.push_str(&format!("assistant called {name} with {arguments}\n"));
                }
            }
            Some(api::message::Message::ToolCallResult(result)) => {
                text.push_str(&format!(
                    "tool result: {}\n",
                    summarize_result(&result.result)
                ));
            }
            _ => {}
        }
    }
    text
}

/// Summarizes a message's tool result, if any.
fn summarize_result(result: &Option<api::message::tool_call_result::Result>) -> String {
    result
        .as_ref()
        .map(summarize_result_kind)
        .unwrap_or_else(|| "(no result)".to_owned())
}

/// Rough token estimate (~4 chars per token) over the provider-neutral view.
pub fn estimate_tokens(system_prompt: &str, messages: &[LlmMessage]) -> u64 {
    let chars: usize = system_prompt.len()
        + messages
            .iter()
            .map(|message| {
                let text = match message {
                    LlmMessage::User { text, .. } => text.len(),
                    LlmMessage::Assistant { text, tool_calls } => {
                        text.len()
                            + tool_calls
                                .iter()
                                .map(|call| call.arguments.to_string().len() + call.name.len())
                                .sum::<usize>()
                    }
                    LlmMessage::ToolResult { text, .. } => text.len(),
                };
                text + 8
            })
            .sum::<usize>();
    (chars / 4) as u64
}

/// The context-window budget for a request, in estimated tokens.
///
/// The client's per-model limit wins when it sets one; otherwise the
/// server-configured budget applies; `0` for both disables compaction.
pub fn effective_context_budget(configured: u64, settings: Option<&api::request::Settings>) -> u64 {
    let client_limit = settings
        .and_then(|settings| settings.model_config.as_ref())
        .map(|model_config| model_config.base_model_context_window_limit)
        .unwrap_or(0);
    if client_limit > 0 {
        u64::from(client_limit)
    } else if configured > 0 {
        configured
    } else {
        AUTO_CONTEXT_BUDGET
    }
}

/// Budget used when neither the client nor the operator configured one.
pub const AUTO_CONTEXT_BUDGET: u64 = 96_000;

/// Picks the compaction cut over a task's messages: everything before `cut`
/// gets summarized, everything from `cut` on stays verbatim.
///
/// Returns `None` when there is nothing worth compacting (the prefix would be
/// too small, or messages lack the ids needed to move them client-side).
pub fn compaction_cut(messages: &[api::Message], budget: u64) -> Option<usize> {
    if messages.len() < 4 {
        return None;
    }
    // Keep a suffix worth roughly a third of the budget, and never keep more
    // than 24 messages.
    let mut kept_estimate = 0u64;
    let mut cut = messages.len();
    while cut > 0 && (kept_estimate < budget / 3 || messages.len() - cut < 2) {
        cut -= 1;
        kept_estimate += estimate_tokens("", &messages_for_estimate(&messages[cut..cut + 1]));
        if messages.len() - cut >= 24 {
            break;
        }
    }
    // A kept message must not open with an unanswered tool result, and every
    // moved message needs an id so the client can move it.
    while cut < messages.len()
        && matches!(
            &messages[cut].message,
            Some(api::message::Message::ToolCallResult(_))
        )
    {
        cut += 1;
    }
    if cut == 0 || messages[..cut].iter().any(|message| message.id.is_empty()) {
        return None;
    }
    let prefix_estimate = estimate_tokens("", &messages_for_estimate(&messages[..cut]));
    (prefix_estimate >= budget / 5).then_some(cut)
}

fn messages_for_estimate(messages: &[api::Message]) -> Vec<LlmMessage> {
    let mut out = Vec::new();
    for message in messages {
        translate_message(message, &mut out);
    }
    out
}

/// Builds the user-visible view of a compacted conversation: a note carrying
/// the summary followed by the retained suffix.
pub fn compacted_view(summary: &str, suffix: &[api::Message]) -> Vec<LlmMessage> {
    let mut view = vec![LlmMessage::User {
        text: format!(
            "[system] Earlier parts of this conversation were summarized to fit the \
             model's context window. Summary of what happened so far:\n\n{summary}\n\n\
             Continue working from this state."
        ),
        images: Vec::new(),
    }];
    let mut suffix_messages = Vec::new();
    for message in suffix {
        translate_message(message, &mut suffix_messages);
    }
    view.extend(suffix_messages);
    view
}
