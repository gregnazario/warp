use std::collections::HashSet;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::multi_agent::{AppState, check_auth};

/// The client's `POST /ai/relevant_files` body: a natural-language query plus
/// the outline (path and symbol summary) of every file in the indexed repo.
#[derive(Debug, serde::Deserialize)]
pub struct GetRelevantFiles {
    pub query: String,
    #[serde(default)]
    pub files: Vec<FileContext>,
}

#[derive(Debug, serde::Deserialize)]
pub struct FileContext {
    pub path: String,
    #[serde(default)]
    pub symbols: String,
}

/// Ranks the client's file outline against the query and returns the most
/// relevant paths.
///
/// The outline only contains paths and symbol names, so this is a lexical
/// relevance heuristic: query tokens matched against the path (weighted
/// higher) and symbol summary. It needs no external service and gives the
/// agent's `search_codebase` tool useful signal for symbol-shaped queries.
pub async fn relevant_files(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Response {
    if !check_auth(&state, &headers).await {
        return (
            axum::http::StatusCode::UNAUTHORIZED,
            "invalid or missing API key",
        )
            .into_response();
    }
    state.metrics.relevant_file_request();
    let Ok(request) = serde_json::from_str::<GetRelevantFiles>(&body) else {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            "undecodable request body",
        )
            .into_response();
    };

    let query_tokens = tokenize(&request.query);
    let mut scored: Vec<(f64, &str)> = request
        .files
        .iter()
        .map(|file| (score_file(&query_tokens, file), file.path.as_str()))
        .collect();
    scored.retain(|(score, _)| *score > 0.0);
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let paths: Vec<String> = scored
        .into_iter()
        .take(MAX_RESULTS)
        .map(|(_, path)| path.to_owned())
        .collect();
    (axum::response::Json(json!({ "relevant_file_paths": paths }))).into_response()
}

/// Only paths that score above zero come back, so cap generously.
const MAX_RESULTS: usize = 10;

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| token.len() > 1)
        .flat_map(|token| {
            // Split camelCase and snake_case compounds so "parseConfigFile"
            // matches a query for "parse config file".
            split_compounds(token)
        })
        .map(|token| token.to_ascii_lowercase())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

fn split_compounds(token: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    for (index, c) in token.char_indices() {
        if index > start && c.is_uppercase() {
            parts.push(&token[start..index]);
            start = index;
        }
    }
    parts.push(&token[start..]);
    parts
}

fn score_file(query_tokens: &[String], file: &FileContext) -> f64 {
    let path_tokens = tokenize(&file.path.replace(['/', '_', '-', '.'], " "));
    let symbol_tokens = tokenize(&file.symbols.replace(['(', ')', ',', '<', '>'], " "));
    query_tokens
        .iter()
        .map(|token| {
            let mut score = 0.0;
            if path_tokens.iter().any(|candidate| candidate == token) {
                score += 3.0;
            }
            if symbol_tokens.iter().any(|candidate| candidate == token) {
                score += 1.0;
            }
            score
        })
        .sum()
}

#[cfg(test)]
#[path = "relevant_files_tests.rs"]
mod tests;
