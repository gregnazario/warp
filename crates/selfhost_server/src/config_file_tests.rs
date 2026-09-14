use super::*;

fn write_config(dir: &std::path::Path, body: &str) -> String {
    let path = dir.join("server.toml");
    std::fs::write(&path, body).unwrap();
    path.to_str().unwrap().to_owned()
}

#[test]
fn file_entries_fill_unset_flags() {
    let dir = std::env::temp_dir().join(format!("praw-cfg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = write_config(
        &dir,
        "meta_api_key = \"mk\"\nprovider = \"meta\"\nbyok_direct = false\nllm_models = [\"a\", \"b\"]\nvertex = true\n",
    );

    let argv = vec![
        "selfhost_server".to_owned(),
        "--bind".to_owned(),
        "127.0.0.1:9".to_owned(),
    ];
    let expanded = expand(&path, &argv).unwrap();
    assert!(expanded.contains(&"--meta-api-key".to_owned()));
    assert!(expanded.contains(&"mk".to_owned()));
    assert!(expanded.contains(&"--provider".to_owned()));
    // byok_direct is a Set-flag with a value.
    let position = expanded
        .iter()
        .position(|arg| arg == "--byok-direct")
        .unwrap();
    assert_eq!(expanded[position + 1], "false");
    // String lists join into the comma form the flag expects.
    let position = expanded
        .iter()
        .position(|arg| arg == "--llm-models")
        .unwrap();
    assert_eq!(expanded[position + 1], "a,b");
    // Booleans flagged true expand to the bare flag.
    assert!(expanded.contains(&"--vertex".to_owned()));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn command_line_and_env_win_over_the_file() {
    let dir = std::env::temp_dir().join(format!("praw-cfg-env-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = write_config(
        &dir,
        "meta_api_key = \"from-file\"\nopenai_api_key = \"from-file\"\n",
    );

    let argv = vec![
        "selfhost_server".to_owned(),
        "--meta-api-key".to_owned(),
        "from-flag".to_owned(),
    ];
    let expanded = expand_with(&path, &argv, |name| name == "SELFHOST_OPENAI_API_KEY").unwrap();
    // Flag value present exactly once — the file's copy is not injected.
    assert_eq!(
        expanded.iter().filter(|arg| **arg == "from-flag").count(),
        1
    );
    assert!(!expanded.contains(&"from-file".to_owned()));
    // Env-covered flags are not injected either.
    assert!(!expanded.contains(&"--openai-api-key".to_owned()));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn missing_file_is_transparent() {
    let argv = vec!["selfhost_server".to_owned()];
    let expanded = expand("/nonexistent/praw-server.toml", &argv).unwrap();
    assert_eq!(expanded, argv);
}
