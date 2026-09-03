use warp_multi_agent_api::ToolType;

use super::*;

fn tool_by_name(name: &str) -> SupportedTool {
    tools_for_client(&[])
        .into_iter()
        .find(|tool| tool.def.name == name)
        .unwrap_or_else(|| panic!("tool {name} not found"))
}

#[test]
fn tools_for_client_filters_by_supported_tools() {
    let supported = vec![ToolType::RunShellCommand as i32, ToolType::Grep as i32];
    let tools = tools_for_client(&supported);
    assert_eq!(tools.len(), 2);
    let names: Vec<&str> = tools.iter().map(|tool| tool.def.name).collect();
    assert!(names.contains(&"run_shell_command"));
    assert!(names.contains(&"grep"));

    // An empty supported list means the server may use any tool.
    assert_eq!(tools_for_client(&[]).len(), all_tools().len());
}

#[test]
fn run_shell_command_builds_from_json() {
    let tool = tool_by_name("run_shell_command");
    let call = (tool.build)(
        "id1".to_owned(),
        &serde_json::json!({"command": "cargo test"}),
    )
    .unwrap();
    assert_eq!(call.tool_call_id, "id1");
    match call.tool {
        Some(api::message::tool_call::Tool::RunShellCommand(run)) => {
            assert_eq!(run.command, "cargo test");
        }
        other => panic!("unexpected tool: {other:?}"),
    }
}

#[test]
fn apply_file_diffs_builds_diffs_and_new_files() {
    let tool = tool_by_name("apply_file_diffs");
    let call = (tool.build)(
        "id2".to_owned(),
        &serde_json::json!({
            "summary": "add greeting",
            "diffs": [{"file_path": "a.txt", "search": "hi", "replace": "hello"}],
            "new_files": [{"file_path": "new.txt", "content": "content"}],
        }),
    )
    .unwrap();
    match call.tool {
        Some(api::message::tool_call::Tool::ApplyFileDiffs(apply)) => {
            assert_eq!(apply.diffs.len(), 1);
            assert_eq!(apply.diffs[0].file_path, "a.txt");
            assert_eq!(apply.new_files.len(), 1);
            assert_eq!(apply.new_files[0].content, "content");
        }
        other => panic!("unexpected tool: {other:?}"),
    }
}

#[test]
fn apply_file_diffs_requires_changes() {
    let tool = tool_by_name("apply_file_diffs");
    assert!((tool.build)("id3".to_owned(), &serde_json::json!({})).is_err());
}

#[test]
fn json_struct_conversion_round_trips() {
    let value = serde_json::json!({
        "string": "s",
        "number": 3,
        "float": 1.5,
        "bool": true,
        "nested": {"a": [1, "two", null]},
    });
    let proto = json_to_prost_struct(&value);
    let back = prost_struct_to_json(&proto);
    assert_eq!(back["string"], "s");
    assert_eq!(back["number"].as_f64(), Some(3.0));
    assert_eq!(back["float"], 1.5);
    assert_eq!(back["bool"], true);
    assert_eq!(back["nested"]["a"][1], "two");
    assert!(back["nested"]["a"][2].is_null());
}

#[test]
fn grep_and_file_glob_build_from_json() {
    let grep = tool_by_name("grep");
    let call = (grep.build)(
        "id4".to_owned(),
        &serde_json::json!({"queries": ["foo", "bar"], "path": "src"}),
    )
    .unwrap();
    match call.tool {
        Some(api::message::tool_call::Tool::Grep(g)) => {
            assert_eq!(g.queries, vec!["foo".to_owned(), "bar".to_owned()]);
            assert_eq!(g.path, "src");
        }
        other => panic!("unexpected tool: {other:?}"),
    }

    let glob = tool_by_name("file_glob");
    let call = (glob.build)(
        "id5".to_owned(),
        &serde_json::json!({"patterns": ["*.rs"], "search_dir": "crates"}),
    )
    .unwrap();
    match call.tool {
        Some(api::message::tool_call::Tool::FileGlobV2(g)) => {
            assert_eq!(g.patterns, vec!["*.rs".to_owned()]);
            assert_eq!(g.search_dir, "crates");
        }
        other => panic!("unexpected tool: {other:?}"),
    }
}
