use anyhow::{Context as _, Result};
use serde_json::Value;

use crate::multi_agent::llm::ToolDef;

/// The name of the server-executed web-search tool. It is offered to the LLM
/// only — the Warp client never sees it; the server runs the search itself and
/// feeds the results back into the conversation.
pub const WEB_SEARCH_TOOL_NAME: &str = "web_search";

/// Definition for the LLM-facing web-search tool.
pub fn tool_def() -> ToolDef {
    ToolDef {
        name: WEB_SEARCH_TOOL_NAME,
        description: "Search the web for current information. Returns titles, URLs, and short \
                      snippets for the top results.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        }),
    }
}

/// One search result, kept structured so it can feed both the LLM and the
/// client's web-search UI message.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Runs `query` against a SearXNG-compatible JSON search endpoint
/// (`<base>?q=<query>&format=json`).
pub async fn search(
    client: &reqwest::Client,
    base_url: &str,
    query: &str,
) -> Result<Vec<SearchResult>> {
    let response = client
        .get(base_url)
        .query(&[("q", query), ("format", "json")])
        .send()
        .await
        .context("search endpoint unreachable")?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("search endpoint returned {status}: {}", body.trim());
    }
    let body: Value = response.json().await.context("search reply was not JSON")?;

    let results = body["results"]
        .as_array()
        .map(|results| {
            results
                .iter()
                .filter_map(|result| {
                    Some(SearchResult {
                        title: result["title"].as_str().unwrap_or_default().to_owned(),
                        url: result["url"].as_str()?.to_owned(),
                        snippet: result["content"].as_str().unwrap_or_default().to_owned(),
                    })
                })
                .take(MAX_RESULTS)
                .collect()
        })
        .unwrap_or_default();
    Ok(results)
}

const MAX_RESULTS: usize = 5;

/// Renders results for the LLM.
pub fn render_for_llm(query: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        return format!("No web results found for '{query}'.");
    }
    let mut text = format!("Web results for '{query}':\n");
    for result in results {
        text.push_str(&format!(
            "\n- {} ({})\n{}",
            result.title, result.url, result.snippet
        ));
    }
    text
}

#[cfg(test)]
#[path = "websearch_tests.rs"]
mod tests;
