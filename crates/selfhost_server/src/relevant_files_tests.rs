use super::*;

#[test]
fn ranks_by_query_token_overlap_preferring_paths() {
    let files = [
        FileContext {
            path: "src/editor_window.rs".to_owned(),
            symbols: "Editor::new Editor::handle_input".to_owned(),
        },
        FileContext {
            path: "src/payment_gateway.rs".to_owned(),
            symbols: "Gateway::charge Receipt".to_owned(),
        },
        FileContext {
            path: "docs/style.md".to_owned(),
            symbols: String::new(),
        },
    ];

    let query_tokens = tokenize("handle input in the editor window");
    let mut scored: Vec<(f64, &str)> = files
        .iter()
        .map(|file| (score_file(&query_tokens, file), file.path.as_str()))
        .collect();
    scored.retain(|(score, _)| *score > 0.0);
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Only the editor file matches the query; the payment gateway and docs
    // files match nothing.
    assert_eq!(scored.len(), 1);
    assert_eq!(scored[0].1, "src/editor_window.rs");
}

#[test]
fn compound_tokens_split_camel_and_snake_case() {
    let tokens = tokenize("parseConfigFile parse_config");
    assert!(tokens.contains(&"parse".to_owned()));
    assert!(tokens.contains(&"config".to_owned()));
    assert!(tokens.contains(&"file".to_owned()));
}

#[test]
fn unmatched_files_are_dropped() {
    let files = [FileContext {
        path: "a/b.rs".to_owned(),
        symbols: String::new(),
    }];
    let query_tokens = tokenize("nothing related here");
    assert_eq!(score_file(&query_tokens, &files[0]), 0.0);
}

#[test]
fn debug_scores() {
    let files = vec![
        FileContext {
            path: "crates/editor/src/editor.rs".to_owned(),
            symbols: "Editor::new Editor::handle_input".to_owned(),
        },
        FileContext {
            path: "crates/editor/src/input/mod.rs".to_owned(),
            symbols: "InputState KeyEvent".to_owned(),
        },
    ];
    let query_tokens = tokenize("handle input in the editor");
    println!("query: {query_tokens:?}");
    for file in &files {
        println!(
            "{}: path_tokens={:?} symbol_tokens={:?} score={}",
            file.path,
            tokenize(&file.path.replace(['/', '_', '-', '.'], " ")),
            tokenize(&file.symbols.replace(['(', ')', ',', '<', '>'], " ")),
            score_file(&query_tokens, file)
        );
    }
}
