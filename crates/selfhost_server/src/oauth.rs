use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context as _, Result};
use instant::Instant;

use crate::config::DynamicAuth;

/// Resolves a dynamic-auth credential to a bearer access token, caching by
/// provider until shortly before expiry.
pub async fn resolve_token(
    http: &reqwest::Client,
    auth: &DynamicAuth,
    cache: &TokenCache,
) -> Result<String> {
    let cache_key = match auth {
        DynamicAuth::AzureClientCredentials { .. } => "azure-foundry",
        DynamicAuth::VertexAdc { .. } => "vertex",
    };
    if let Ok(entries) = cache.lock()
        && let Some((token, expires_at)) = entries.get(cache_key)
        && Instant::now() < *expires_at
    {
        return Ok(token.clone());
    }

    let (token_url, form): (String, Vec<(&str, String)>) = match auth {
        DynamicAuth::AzureClientCredentials {
            token_url,
            client_id,
            client_secret,
            scope,
        } => (
            token_url.clone(),
            vec![
                ("grant_type", "client_credentials".to_owned()),
                ("client_id", client_id.clone()),
                ("client_secret", client_secret.clone()),
                ("scope", scope.clone()),
            ],
        ),
        DynamicAuth::VertexAdc {
            token_url,
            client_id,
            client_secret,
            refresh_token,
        } => (
            token_url.clone(),
            vec![
                ("grant_type", "refresh_token".to_owned()),
                ("refresh_token", refresh_token.clone()),
                ("client_id", client_id.clone()),
                ("client_secret", client_secret.clone()),
            ],
        ),
    };

    let response = http
        .post(&token_url)
        .form(&form)
        .send()
        .await
        .context("OAuth token endpoint unreachable")?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!("OAuth token endpoint returned {status}: {}", body.trim());
    }
    let payload: serde_json::Value = serde_json::from_str(&body).with_context(|| {
        format!(
            "OAuth token reply was not JSON: {}",
            &body[..body.len().min(200)]
        )
    })?;
    let token = payload["access_token"]
        .as_str()
        .context("OAuth token reply missing access_token")?
        .to_owned();
    let expires_in = payload["expires_in"].as_u64().unwrap_or(3600);

    if let Ok(mut entries) = cache.lock() {
        entries.insert(
            cache_key.to_owned(),
            (
                token.clone(),
                Instant::now() + Duration::from_secs(expires_in.saturating_sub(60)),
            ),
        );
    }
    Ok(token)
}

/// Shared token cache keyed by provider.
pub type TokenCache = std::sync::Mutex<HashMap<String, (String, Instant)>>;

#[cfg(test)]
#[path = "oauth_tests.rs"]
mod tests;
