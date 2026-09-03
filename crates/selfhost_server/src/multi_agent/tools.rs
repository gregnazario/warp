use std::collections::BTreeMap;

use anyhow::{Context as _, Result, bail};
use serde_json::{Value, json};
use warp_multi_agent_api as api;

use super::llm::ToolDef;

/// A tool the server can offer: its definition for the LLM plus how to build
/// the Warp `ToolCall` message from the model's JSON arguments.
#[derive(Clone)]
pub struct SupportedTool {
    pub tool_type: api::ToolType,
    pub def: ToolDef,
    pub build: fn(tool_call_id: String, arguments: &Value) -> Result<api::message::ToolCall>,
}

macro_rules! string_arg {
    ($arguments:expr, $field:literal) => {
        $arguments
            .get($field)
            .and_then(Value::as_str)
            .context(concat!("missing string argument '", $field, "'"))?
    };
}

/// The tools the self-hosted server can drive, all executed by the client.
pub fn all_tools() -> Vec<SupportedTool> {
    use api::message::tool_call as tc;
    use warp_multi_agent_api::ToolType;

    vec![
        SupportedTool {
            tool_type: ToolType::RunShellCommand,
            def: ToolDef {
                name: "run_shell_command",
                description: "Run a shell command in the user's terminal. The user is asked for \
                              confirmation for commands that are not read-only.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to run.",
                        }
                    },
                    "required": ["command"],
                }),
            },
            build: |tool_call_id, arguments| {
                Ok(api::message::ToolCall {
                    tool_call_id,
                    tool: Some(tc::Tool::RunShellCommand(tc::RunShellCommand {
                        command: string_arg!(arguments, "command").to_owned(),
                        ..Default::default()
                    })),
                })
            },
        },
        SupportedTool {
            tool_type: ToolType::ReadFiles,
            def: ToolDef {
                name: "read_files",
                description: "Read the contents of files at the given paths (relative to the \
                              working directory).",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Path of the file to read."},
                                },
                                "required": ["name"],
                            },
                        }
                    },
                    "required": ["files"],
                }),
            },
            build: |tool_call_id, arguments| {
                let files = arguments["files"]
                    .as_array()
                    .context("missing array argument 'files'")?;
                Ok(api::message::ToolCall {
                    tool_call_id,
                    tool: Some(tc::Tool::ReadFiles(tc::ReadFiles {
                        files: files
                            .iter()
                            .filter_map(|file| {
                                file["name"].as_str().map(|name| tc::read_files::File {
                                    name: name.to_owned(),
                                    line_ranges: Vec::new(),
                                })
                            })
                            .collect(),
                    })),
                })
            },
        },
        SupportedTool {
            tool_type: ToolType::SearchCodebase,
            def: ToolDef {
                name: "search_codebase",
                description: "Search for code relevant to a natural-language query across the                               codebase the user has open. Returns the most relevant files.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for, in natural language.",
                        },
                        "path_filters": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional path prefixes to restrict the search to.",
                        },
                    },
                    "required": ["query"],
                }),
            },
            build: |tool_call_id, arguments| {
                let path_filters = arguments["path_filters"]
                    .as_array()
                    .map(|filters| {
                        filters
                            .iter()
                            .filter_map(|filter| filter.as_str().map(ToOwned::to_owned))
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(api::message::ToolCall {
                    tool_call_id,
                    tool: Some(tc::Tool::SearchCodebase(tc::SearchCodebase {
                        query: string_arg!(arguments, "query").to_owned(),
                        path_filters,
                        codebase_path: String::new(),
                    })),
                })
            },
        },
        SupportedTool {
            tool_type: ToolType::Grep,
            def: ToolDef {
                name: "grep",
                description: "Search file contents for the given patterns, returning matching \
                              files and line numbers.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Search patterns.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional file or directory (relative) to search in.",
                        },
                    },
                    "required": ["queries"],
                }),
            },
            build: |tool_call_id, arguments| {
                let queries = arguments["queries"]
                    .as_array()
                    .context("missing array argument 'queries'")?
                    .iter()
                    .filter_map(|query| query.as_str().map(ToOwned::to_owned))
                    .collect();
                Ok(api::message::ToolCall {
                    tool_call_id,
                    tool: Some(tc::Tool::Grep(tc::Grep {
                        queries,
                        path: arguments["path"].as_str().unwrap_or_default().to_owned(),
                    })),
                })
            },
        },
        SupportedTool {
            tool_type: ToolType::FileGlobV2,
            def: ToolDef {
                name: "file_glob",
                description: "Find files whose names match the given glob patterns (supports ?, \
                              *, []).",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Glob patterns to match file names against.",
                        },
                        "search_dir": {
                            "type": "string",
                            "description": "Optional directory (relative) to search in.",
                        },
                        "max_matches": {
                            "type": "integer",
                            "description": "Optional maximum number of matches (0 = no limit).",
                        },
                    },
                    "required": ["patterns"],
                }),
            },
            build: |tool_call_id, arguments| {
                let patterns = arguments["patterns"]
                    .as_array()
                    .context("missing array argument 'patterns'")?
                    .iter()
                    .filter_map(|pattern| pattern.as_str().map(ToOwned::to_owned))
                    .collect();
                Ok(api::message::ToolCall {
                    tool_call_id,
                    tool: Some(tc::Tool::FileGlobV2(tc::FileGlobV2 {
                        patterns,
                        search_dir: arguments["search_dir"]
                            .as_str()
                            .unwrap_or_default()
                            .to_owned(),
                        max_matches: arguments["max_matches"].as_i64().unwrap_or(0) as i32,
                        max_depth: 0,
                        min_depth: 0,
                    })),
                })
            },
        },
        SupportedTool {
            tool_type: ToolType::ApplyFileDiffs,
            def: ToolDef {
                name: "apply_file_diffs",
                description: "Edit files by applying exact string replacements, create new \
                              files, or delete files. Each replacement's `search` must appear \
                              exactly once in the file.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "One-line summary of the change."},
                        "diffs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file_path": {"type": "string"},
                                    "search": {"type": "string", "description": "Exact content to replace."},
                                    "replace": {"type": "string", "description": "Replacement content."},
                                },
                                "required": ["file_path", "search", "replace"],
                            },
                        },
                        "new_files": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file_path": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["file_path", "content"],
                            },
                        },
                    },
                }),
            },
            build: |tool_call_id, arguments| {
                let diffs: Vec<tc::apply_file_diffs::FileDiff> = arguments["diffs"]
                    .as_array()
                    .map(|diffs| {
                        diffs
                            .iter()
                            .filter_map(|diff| {
                                Some(tc::apply_file_diffs::FileDiff {
                                    file_path: diff["file_path"].as_str()?.to_owned(),
                                    search: diff["search"].as_str().unwrap_or_default().to_owned(),
                                    replace: diff["replace"]
                                        .as_str()
                                        .unwrap_or_default()
                                        .to_owned(),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let new_files: Vec<tc::apply_file_diffs::NewFile> = arguments["new_files"]
                    .as_array()
                    .map(|files| {
                        files
                            .iter()
                            .filter_map(|file| {
                                Some(tc::apply_file_diffs::NewFile {
                                    file_path: file["file_path"].as_str()?.to_owned(),
                                    content: file["content"]
                                        .as_str()
                                        .unwrap_or_default()
                                        .to_owned(),
                                    allow_overwrite: true,
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if diffs.is_empty() && new_files.is_empty() {
                    bail!("apply_file_diffs requires at least one entry in 'diffs' or 'new_files'");
                }
                Ok(api::message::ToolCall {
                    tool_call_id,
                    tool: Some(tc::Tool::ApplyFileDiffs(tc::ApplyFileDiffs {
                        summary: arguments["summary"].as_str().unwrap_or_default().to_owned(),
                        diffs,
                        new_files,
                        deleted_files: Vec::new(),
                        v4a_updates: Vec::new(),
                    })),
                })
            },
        },
        SupportedTool {
            tool_type: ToolType::WriteToLongRunningShellCommand,
            def: ToolDef {
                name: "write_to_long_running_command",
                description: "Write input to a long-running command's PTY (e.g. answer a \
                              prompt).",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Text to write to the command."},
                        "mode": {
                            "type": "string",
                            "enum": ["raw", "line", "block"],
                            "description": "How to write the input. Defaults to 'line'.",
                        },
                        "command_id": {
                            "type": "string",
                            "description": "Optional ID of the command to write to.",
                        },
                    },
                    "required": ["input"],
                }),
            },
            build: |tool_call_id, arguments| {
                let mode = match arguments["mode"].as_str() {
                    Some("raw") => tc::write_to_long_running_shell_command::mode::Mode::Raw(()),
                    Some("block") => tc::write_to_long_running_shell_command::mode::Mode::Block(()),
                    _ => tc::write_to_long_running_shell_command::mode::Mode::Line(()),
                };
                Ok(api::message::ToolCall {
                    tool_call_id,
                    tool: Some(tc::Tool::WriteToLongRunningShellCommand(
                        tc::WriteToLongRunningShellCommand {
                            input: string_arg!(arguments, "input").as_bytes().to_vec(),
                            mode: Some(tc::write_to_long_running_shell_command::Mode {
                                mode: Some(mode),
                            }),
                            command_id: arguments["command_id"]
                                .as_str()
                                .unwrap_or_default()
                                .to_owned(),
                        },
                    )),
                })
            },
        },
        SupportedTool {
            tool_type: ToolType::ReadShellCommandOutput,
            def: ToolDef {
                name: "read_command_output",
                description: "Read the output of a long-running command, waiting until it \
                              completes.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command_id": {
                            "type": "string",
                            "description": "ID of the command to read.",
                        }
                    },
                    "required": ["command_id"],
                }),
            },
            build: |tool_call_id, arguments| {
                Ok(api::message::ToolCall {
                    tool_call_id,
                    tool: Some(tc::Tool::ReadShellCommandOutput(
                        tc::ReadShellCommandOutput {
                            command_id: string_arg!(arguments, "command_id").to_owned(),
                            delay: Some(tc::read_shell_command_output::Delay::OnCompletion(())),
                        },
                    )),
                })
            },
        },
        SupportedTool {
            tool_type: ToolType::AskUserQuestion,
            def: ToolDef {
                name: "ask_user_question",
                description: "Ask the user one or multiple-choice questions and wait for their \
                              answers.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "options": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Selectable answers.",
                                    },
                                },
                                "required": ["question", "options"],
                            },
                        }
                    },
                    "required": ["questions"],
                }),
            },
            build: |tool_call_id, arguments| {
                let questions = arguments["questions"]
                    .as_array()
                    .context("missing array argument 'questions'")?;
                let mut built = Vec::new();
                for (index, question) in questions.iter().enumerate() {
                    let options: Vec<api::ask_user_question::Option> = question["options"]
                        .as_array()
                        .map(|options| {
                            options
                                .iter()
                                .filter_map(|option| {
                                    option.as_str().map(|label| api::ask_user_question::Option {
                                        label: label.to_owned(),
                                    })
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    built.push(api::ask_user_question::Question {
                        question_id: format!("q{index}"),
                        question: question["question"]
                            .as_str()
                            .context("question is missing its 'question' text")?
                            .to_owned(),
                        question_type: Some(
                            api::ask_user_question::question::QuestionType::MultipleChoice(
                                api::ask_user_question::MultipleChoice {
                                    options,
                                    recommended_option_index: 0,
                                    is_multiselect: false,
                                    supports_other: true,
                                },
                            ),
                        ),
                    });
                }
                Ok(api::message::ToolCall {
                    tool_call_id,
                    tool: Some(tc::Tool::AskUserQuestion(api::AskUserQuestion {
                        questions: built,
                    })),
                })
            },
        },
        SupportedTool {
            tool_type: ToolType::SuggestPrompt,
            def: ToolDef {
                name: "suggest_prompt",
                description: "Suggest a follow-up prompt the user may want to run next.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "The suggested prompt."},
                        "label": {"type": "string", "description": "Optional short label."},
                    },
                    "required": ["prompt"],
                }),
            },
            build: |tool_call_id, arguments| {
                Ok(api::message::ToolCall {
                    tool_call_id,
                    tool: Some(tc::Tool::SuggestPrompt(tc::SuggestPrompt {
                        display_mode: Some(tc::suggest_prompt::DisplayMode::PromptChip(
                            tc::suggest_prompt::PromptChip {
                                prompt: string_arg!(arguments, "prompt").to_owned(),
                                label: arguments["label"].as_str().unwrap_or_default().to_owned(),
                            },
                        )),
                        is_trigger_irrelevant: false,
                    })),
                })
            },
        },
        SupportedTool {
            tool_type: ToolType::CallMcpTool,
            def: ToolDef {
                name: "call_mcp_tool",
                description: "Call a tool provided by a connected MCP server.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the MCP tool."},
                        "args": {"type": "object", "description": "Named arguments for the tool."},
                    },
                    "required": ["name", "args"],
                }),
            },
            build: |tool_call_id, arguments| {
                Ok(api::message::ToolCall {
                    tool_call_id,
                    tool: Some(tc::Tool::CallMcpTool(tc::CallMcpTool {
                        name: string_arg!(arguments, "name").to_owned(),
                        args: Some(json_to_prost_struct(&arguments["args"])),
                        // The handler resolves this from the request's MCP
                        // context when the model omits it.
                        server_id: String::new(),
                    })),
                })
            },
        },
        SupportedTool {
            tool_type: ToolType::ReadMcpResource,
            def: ToolDef {
                name: "read_mcp_resource",
                description: "Read a resource provided by a connected MCP server.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "uri": {"type": "string", "description": "URI of the resource to read."},
                    },
                    "required": ["uri"],
                }),
            },
            build: |tool_call_id, arguments| {
                Ok(api::message::ToolCall {
                    tool_call_id,
                    tool: Some(tc::Tool::ReadMcpResource(tc::ReadMcpResource {
                        uri: string_arg!(arguments, "uri").to_owned(),
                        server_id: String::new(),
                    })),
                })
            },
        },
    ]
}

/// Returns the tools the client declared support for (all of them when the
/// client declared none), in a stable order.
pub fn tools_for_client(supported_tools: &[i32]) -> Vec<SupportedTool> {
    let supported: std::collections::HashSet<i32> = supported_tools.iter().copied().collect();
    ALL_TOOLS
        .get_or_init(all_tools)
        .iter()
        .filter(|tool| supported_tools.is_empty() || supported.contains(&(tool.tool_type as i32)))
        .cloned()
        .collect()
}

static ALL_TOOLS: std::sync::OnceLock<Vec<SupportedTool>> = std::sync::OnceLock::new();

/// Converts a JSON object into a protobuf `Struct` (for MCP tool arguments).
pub fn json_to_prost_struct(value: &Value) -> prost_types::Struct {
    let mut fields = BTreeMap::new();
    if let Value::Object(map) = value {
        for (key, value) in map {
            fields.insert(key.clone(), json_to_prost_value(value));
        }
    }
    prost_types::Struct { fields }
}

/// Converts a protobuf `Struct` (e.g. MCP tool arguments) into JSON.
pub fn prost_struct_to_json(value: &prost_types::Struct) -> Value {
    let mut map = serde_json::Map::new();
    for (key, value) in &value.fields {
        map.insert(key.clone(), prost_value_to_json(value));
    }
    Value::Object(map)
}

fn prost_value_to_json(value: &prost_types::Value) -> Value {
    use prost_types::value::Kind;
    match value.kind.as_ref() {
        Some(Kind::NullValue(_)) | None => Value::Null,
        Some(Kind::BoolValue(b)) => Value::Bool(*b),
        Some(Kind::NumberValue(n)) => serde_json::Number::from_f64(*n)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Some(Kind::StringValue(s)) => Value::String(s.clone()),
        Some(Kind::ListValue(list)) => {
            Value::Array(list.values.iter().map(prost_value_to_json).collect())
        }
        Some(Kind::StructValue(inner)) => prost_struct_to_json(inner),
    }
}

fn json_to_prost_value(value: &Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match value {
        Value::Null => Kind::NullValue(0),
        Value::Bool(b) => Kind::BoolValue(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Kind::NumberValue(i as f64)
            } else {
                Kind::NumberValue(n.as_f64().unwrap_or_default())
            }
        }
        Value::String(s) => Kind::StringValue(s.clone()),
        Value::Array(items) => Kind::ListValue(prost_types::ListValue {
            values: items.iter().map(json_to_prost_value).collect(),
        }),
        Value::Object(_) => Kind::StructValue(json_to_prost_struct(value)),
    };
    prost_types::Value { kind: Some(kind) }
}

#[cfg(test)]
#[path = "tools_tests.rs"]
mod tests;
