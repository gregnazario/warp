//! Live model catalogs for keyed providers, so their models appear in the
//! client's picker automatically (the way first-party catalogs do) instead of
//! via hand-maintained lists that go stale.

use serde_json::Value;

use crate::config::{LlmSchema, ProviderKind};

/// Where a provider's OpenAI-shaped `/models` list lives and which model ids
/// are agent-usable chat models (provider lists include embeddings, tts, and
/// friends that cannot drive an agent).
pub struct CatalogEndpoint {
    pub base_url: &'static str,
    pub schema: LlmSchema,
    /// Lowercase prefixes; an empty slice keeps everything (curated lists).
    pub chat_prefixes: &'static [&'static str],
}

pub fn endpoint_for(kind: &ProviderKind) -> Option<CatalogEndpoint> {
    match kind {
        ProviderKind::Openai => Some(CatalogEndpoint {
            base_url: "https://api.openai.com/v1",
            schema: LlmSchema::Openai,
            chat_prefixes: &["gpt", "o1", "o3", "o4", "chatgpt", "codex"],
        }),
        ProviderKind::Anthropic => Some(CatalogEndpoint {
            base_url: "https://api.anthropic.com/v1",
            schema: LlmSchema::Anthropic,
            chat_prefixes: &["claude"],
        }),
        ProviderKind::Google => Some(CatalogEndpoint {
            base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
            schema: LlmSchema::Openai,
            chat_prefixes: &["gemini"],
        }),
        ProviderKind::Xai => Some(CatalogEndpoint {
            base_url: "https://api.x.ai/v1",
            schema: LlmSchema::Openai,
            chat_prefixes: &["grok"],
        }),
        ProviderKind::Zai => Some(CatalogEndpoint {
            base_url: "https://api.z.ai/api/coding/paas/v4",
            schema: LlmSchema::Openai,
            chat_prefixes: &["glm"],
        }),
        // OpenCode's public list is already curated to agent models.
        ProviderKind::Opencode => Some(CatalogEndpoint {
            base_url: "https://opencode.ai/zen/v1",
            schema: LlmSchema::Openai,
            chat_prefixes: &[],
        }),
        ProviderKind::Meta => Some(CatalogEndpoint {
            base_url: "https://api.meta.ai/v1",
            schema: LlmSchema::Openai,
            chat_prefixes: &["muse"],
        }),
        // OpenRouter's catalog is hundreds of models; fetching it wholesale
        // would flood the picker. It stays route-only (any `vendor/model` id).
        ProviderKind::OpenRouter
        | ProviderKind::Chatgpt
        | ProviderKind::AzureFoundry
        | ProviderKind::Vertex => None,
    }
}

/// Fetches the provider's chat-model ids. Empty when unreachable or
/// unauthenticated — callers keep their static fallbacks.
pub async fn fetch_models(
    http: &reqwest::Client,
    endpoint: &CatalogEndpoint,
    api_key: &str,
) -> Vec<String> {
    let url = format!("{}/models", endpoint.base_url.trim_end_matches('/'));
    let mut request = http.get(&url).timeout(std::time::Duration::from_secs(4));
    // Anthropic authenticates with the x-api-key header pair, everyone else
    // is bearer-shaped.
    if endpoint.schema == LlmSchema::Anthropic {
        request = request
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01");
    } else {
        request = request.bearer_auth(api_key);
    }
    let Ok(response) = request.send().await else {
        return Vec::new();
    };
    if !response.status().is_success() {
        return Vec::new();
    }
    let Ok(body) = response.json::<Value>().await else {
        return Vec::new();
    };
    crate::detect::model_ids(&body)
        .into_iter()
        .filter(|model| {
            endpoint.chat_prefixes.is_empty()
                || endpoint
                    .chat_prefixes
                    .iter()
                    .any(|prefix| model.to_ascii_lowercase().starts_with(prefix))
        })
        .collect()
}

#[cfg(test)]
#[path = "provider_catalog_tests.rs"]
mod tests;
