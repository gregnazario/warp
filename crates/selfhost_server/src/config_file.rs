//! The persistent server config: a TOML file that lives beside the UI's data
//! so the app-started backend and manual runs share one configuration.
//! Precedence: command-line flags > environment variables > this file >
//! built-in defaults, implemented by expanding file entries into argv for
//! flags the user did not pass and no environment variable provides.

use std::collections::HashMap;

use anyhow::{Context as _, Result};

/// Side by side with the UI's data (warp.sqlite): on macOS the app-support
/// directory, elsewhere XDG config.
pub fn default_path() -> String {
    if cfg!(target_os = "macos") {
        format!(
            "{}/Library/Application Support/dev.parw.PRAW/server.toml",
            std::env::var("HOME").unwrap_or_default()
        )
    } else {
        format!(
            "{}/praw/server.toml",
            match std::env::var("XDG_CONFIG_HOME") {
                Ok(dir) if !dir.is_empty() => dir,
                _ => format!("{}/.config", std::env::var("HOME").unwrap_or_default()),
            }
        )
    }
}

/// The env variable that names a config file explicitly.
pub const CONFIG_PATH_ENV: &str = "PRAW_SERVER_CONFIG";

/// Env variables that stand in for flags; a file entry never overrides one.
const FLAG_ENV: &[(&str, &str)] = &[
    ("llm-api-key", "SELFHOST_LLM_API_KEY"),
    ("api-key", "SELFHOST_API_KEY"),
    ("transcribe-base-url", "SELFHOST_TRANSCRIBE_BASE_URL"),
    ("transcribe-api-key", "SELFHOST_TRANSCRIBE_API_KEY"),
    ("web-search-url", "SELFHOST_WEB_SEARCH_URL"),
    ("auth-introspect-url", "SELFHOST_AUTH_INTROSPECT_URL"),
    ("openai-api-key", "SELFHOST_OPENAI_API_KEY"),
    ("anthropic-api-key", "SELFHOST_ANTHROPIC_API_KEY"),
    ("google-api-key", "SELFHOST_GOOGLE_API_KEY"),
    ("xai-api-key", "SELFHOST_XAI_API_KEY"),
    ("zai-api-key", "SELFHOST_ZAI_API_KEY"),
    ("opencode-api-key", "SELFHOST_OPENCODE_API_KEY"),
    ("openrouter-api-key", "SELFHOST_OPENROUTER_API_KEY"),
    ("meta-api-key", "SELFHOST_META_API_KEY"),
];

/// Flag names that take no value (store_true style).
const VALUELESS_FLAGS: &[&str] = &["vertex", "codex-login", "foundry-login", "vertex-login"];

/// Parses the file's flat table into flag-name -> value strings.
fn file_entries(body: &str) -> Result<HashMap<String, String>> {
    let parsed: toml::Table = body
        .parse()
        .with_context(|| "failed to parse the server config file")?;
    let mut entries = HashMap::new();
    for (key, value) in parsed {
        let flag = key.replace('_', "-");
        let rendered = match value {
            toml::Value::String(text) => text,
            toml::Value::Integer(int) => int.to_string(),
            toml::Value::Float(float) => float.to_string(),
            toml::Value::Boolean(flag) => flag.to_string(),
            toml::Value::Array(items) => items
                .into_iter()
                .filter_map(|item| match item {
                    toml::Value::String(text) => Some(text),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(","),
            other => {
                anyhow::bail!("config key '{key}' must be scalar or a string list, got {other:?}")
            }
        };
        if !rendered.is_empty() {
            entries.insert(flag, rendered);
        }
    }
    Ok(entries)
}

/// The flags already provided on this command line (both `--flag value` and
/// `--flag=value` spellings, plus their values).
fn passed_flags(argv: &[String]) -> HashMap<String, Option<String>> {
    let mut passed = HashMap::new();
    let mut iter = argv.iter().peekable();
    while let Some(arg) = iter.next() {
        let Some(flag) = arg.strip_prefix("--") else {
            continue;
        };
        if let Some((flag, value)) = flag.split_once('=') {
            passed.insert(flag.to_owned(), Some(value.to_owned()));
        } else if VALUELESS_FLAGS.contains(&flag) {
            passed.insert(flag.to_owned(), None);
        } else if let Some(next) = iter.peek() {
            if !next.starts_with("--") {
                passed.insert(flag.to_owned(), Some(iter.next().unwrap().clone()));
            } else {
                passed.insert(flag.to_owned(), None);
            }
        }
    }
    passed
}

/// Expands the config file into extra argv entries: one `--flag value` pair
/// per file entry the command line does not set and no env variable provides.
pub fn expand(path: &str, argv: &[String]) -> Result<Vec<String>> {
    expand_with(path, argv, |name| std::env::var_os(name).is_some())
}

fn expand_with(path: &str, argv: &[String], env_set: impl Fn(&str) -> bool) -> Result<Vec<String>> {
    let body = match std::fs::read_to_string(path) {
        Ok(body) => body,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(argv.to_vec());
        }
        Err(error) => {
            return Err(anyhow::Error::new(error).context(format!("failed to read {path}")));
        }
    };
    let entries = file_entries(&body)?;
    let passed = passed_flags(argv);
    let mut expanded = argv.to_vec();
    for (flag, value) in entries {
        if passed.contains_key(&flag) {
            continue;
        }
        if FLAG_ENV
            .iter()
            .any(|(flag_name, env)| *flag_name == flag && env_set(env))
        {
            continue;
        }
        if VALUELESS_FLAGS.contains(&flag.as_str()) {
            if value == "true" {
                expanded.push(format!("--{flag}"));
            }
        } else {
            expanded.push(format!("--{flag}"));
            expanded.push(value);
        }
    }
    Ok(expanded)
}

#[cfg(test)]
#[path = "config_file_tests.rs"]
mod tests;
