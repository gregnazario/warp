use anyhow::Result;
use serde_json::Value;

use crate::config::LlmBackend;

/// One candidate local backend to probe.
pub struct ProbeTarget {
    pub backend: LlmBackend,
    /// Base URL of the OpenAI-compatible API (without `/v1` for Ollama).
    pub base_url: &'static str,
}

/// The well-known local backends, in probe order.
pub fn default_targets() -> Vec<ProbeTarget> {
    vec![
        ProbeTarget {
            backend: LlmBackend::Ollama,
            base_url: "http://127.0.0.1:11434",
        },
        ProbeTarget {
            backend: LlmBackend::LmStudio,
            base_url: "http://127.0.0.1:1234",
        },
        ProbeTarget {
            backend: LlmBackend::Mlx,
            base_url: "http://127.0.0.1:8080",
        },
    ]
}

/// A successfully probed backend: where it lives and which models it serves.
#[derive(Debug, Clone)]
pub struct DetectedBackend {
    pub backend: LlmBackend,
    pub base_url: String,
    pub models: Vec<String>,
}

/// Probes candidates in order and returns the first backend that responds to
/// `GET {base}/v1/models` with a model list. Ollama's `/v1/models` only lists
/// models the OpenAI endpoint accepts, so its richer `/api/tags` is preferred
/// when present.
pub async fn detect(http: &reqwest::Client, targets: &[ProbeTarget]) -> Option<DetectedBackend> {
    for target in targets {
        match probe(http, target).await {
            Ok(detected) => return Some(detected),
            Err(error) => {
                tracing::debug!(
                    backend = ?target.backend,
                    %error,
                    "probe failed"
                );
            }
        }
    }
    None
}

async fn probe(http: &reqwest::Client, target: &ProbeTarget) -> Result<DetectedBackend> {
    let models_url = format!("{}/v1/models", target.base_url.trim_end_matches('/'));
    let response = http
        .get(&models_url)
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .context(format!(
            "{} at {models_url} is not responding",
            target.backend
        ))?;
    if !response.status().is_success() {
        anyhow::bail!(
            "{} at {models_url} answered {}",
            target.backend,
            response.status()
        );
    }
    let body: Value = response.json().await.context(format!(
        "{} returned non-JSON from {models_url}",
        target.backend
    ))?;

    let mut models = model_ids(&body);
    if target.backend == LlmBackend::Ollama {
        // Prefer /api/tags, which lists every pulled model.
        if let Ok(tags) = http
            .get(format!(
                "{}/api/tags",
                target.base_url.trim_end_matches('/')
            ))
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
            && tags.status().is_success()
            && let Ok(body) = tags.json::<Value>().await
        {
            let tagged = model_ids(&body);
            if !tagged.is_empty() {
                models = tagged;
            }
        }
    }

    Ok(DetectedBackend {
        backend: target.backend,
        base_url: target.base_url.to_owned(),
        models,
    })
}

/// Extracts model ids from either `{data: [{id: ...}]}` (OpenAI style) or
/// `{models: [{name: ...}]}` (Ollama /api/tags style).
fn model_ids(body: &Value) -> Vec<String> {
    body["data"]
        .as_array()
        .iter()
        .flat_map(|entries| entries.iter().filter_map(|e| e["id"].as_str()))
        .chain(
            body["models"]
                .as_array()
                .iter()
                .flat_map(|entries| entries.iter().filter_map(|e| e["name"].as_str())),
        )
        .map(ToOwned::to_owned)
        .collect()
}

use anyhow::Context as _;

#[cfg(test)]
#[path = "detect_tests.rs"]
mod tests;
