//! Generic OAuth 2.0 device-authorization flow (RFC 8628), used by the
//! `--foundry-login` and `--vertex-login` commands so users can sign in with
//! their work accounts instead of provisioning service principals or running
//! vendor CLIs.

use std::time::Duration;

use anyhow::{Context as _, Result};
use instant::Instant;
use serde_json::Value;

/// One provider's device-flow endpoints and public client identity. The
/// clients are the vendors' own first-party public apps (the Azure CLI's and
/// the Google Cloud SDK's), whose ids are public by design for native tools.
pub struct DeviceFlow {
    pub device_url: String,
    pub token_url: String,
    pub client_id: &'static str,
    pub client_secret: Option<&'static str>,
    pub scope: &'static str,
}

/// The public client of the Azure CLI; Microsoft's documented first-party
/// device-flow client for native tools.
pub const AZURE_CLI_CLIENT_ID: &str = "04b07795-8ddb-461a-bbee-02f9e1bf7b46";

/// The Google Cloud SDK's public installed-app client (as embedded in gcloud).
pub const GCLOUD_CLIENT_ID: &str = "32555940559.apps.googleusercontent.com";
pub const GCLOUD_CLIENT_SECRET: &str = "ZmssLNjJy2998hD4CTg2ejr2";

pub fn azure_foundry_flow() -> DeviceFlow {
    DeviceFlow {
        device_url: "https://login.microsoftonline.com/organizations/oauth2/v2.0/devicecode"
            .to_owned(),
        token_url: "https://login.microsoftonline.com/organizations/oauth2/v2.0/token".to_owned(),
        client_id: AZURE_CLI_CLIENT_ID,
        client_secret: None,
        scope: "https://cognitiveservices.azure.com/.default",
    }
}

pub fn gcloud_flow() -> DeviceFlow {
    DeviceFlow {
        device_url: "https://oauth2.googleapis.com/device/code".to_owned(),
        token_url: "https://oauth2.googleapis.com/token".to_owned(),
        client_id: GCLOUD_CLIENT_ID,
        client_secret: Some(GCLOUD_CLIENT_SECRET),
        scope: "https://www.googleapis.com/auth/cloud-platform",
    }
}

#[derive(Debug)]
pub struct DeviceGrant {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_in: u64,
}

/// Runs the flow to completion: requests a device code, hands the
/// verification URL to `open` (which typically prints and browser-opens it),
/// and polls until the user approves.
pub async fn run(
    http: &reqwest::Client,
    flow: &DeviceFlow,
    mut open: impl FnMut(&str, &str),
) -> Result<DeviceGrant> {
    let mut form = vec![
        ("client_id", flow.client_id.to_owned()),
        ("scope", flow.scope.to_owned()),
    ];
    if let Some(secret) = flow.client_secret {
        form.push(("client_secret", secret.to_owned()));
    }
    let response = http
        .post(&flow.device_url)
        .form(&form)
        .send()
        .await
        .context("device-authorization endpoint unreachable")?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!("device-authorization failed ({status}): {}", body.trim());
    }
    let code: Value = serde_json::from_str(&body).with_context(|| {
        format!(
            "device-authorization reply was not JSON: {}",
            &body[..body.len().min(200)]
        )
    })?;
    let device_code = code["device_code"]
        .as_str()
        .context("device-authorization reply missing device_code")?
        .to_owned();
    let user_code = code["user_code"].as_str().unwrap_or_default().to_owned();
    let verification = code["verification_uri"]
        .as_str()
        .or_else(|| code["verification_url"].as_str())
        .unwrap_or("https://microsoft.com/devicelogin")
        .to_owned();
    let mut interval = code["interval"].as_u64().unwrap_or(5).max(1);

    open(&verification, &user_code);

    let deadline = Instant::now() + Duration::from_secs(code["expires_in"].as_u64().unwrap_or(900));
    loop {
        if Instant::now() > deadline {
            anyhow::bail!("device login expired before approval");
        }
        tokio::time::sleep(Duration::from_secs(interval)).await;

        let mut form = vec![
            (
                "grant_type",
                "urn:ietf:params:oauth:grant-type:device_code".to_owned(),
            ),
            ("client_id", flow.client_id.to_owned()),
            ("device_code", device_code.clone()),
        ];
        if let Some(secret) = flow.client_secret {
            form.push(("client_secret", secret.to_owned()));
        }
        let response = http
            .post(&flow.token_url)
            .form(&form)
            .send()
            .await
            .context("token endpoint unreachable while waiting for device-login approval")?;
        let body = response.text().await.unwrap_or_default();
        let payload: Value = match serde_json::from_str(&body) {
            Ok(payload) => payload,
            Err(_) => anyhow::bail!(
                "token endpoint returned non-JSON: {}",
                &body[..body.len().min(200)]
            ),
        };
        if let Some(token) = payload["access_token"].as_str() {
            return Ok(DeviceGrant {
                access_token: token.to_owned(),
                refresh_token: payload["refresh_token"].as_str().map(ToOwned::to_owned),
                expires_in: payload["expires_in"].as_u64().unwrap_or(3600),
            });
        }
        match payload["error"].as_str().unwrap_or_default() {
            "authorization_pending" => {}
            "slow_down" => interval += 5,
            error => anyhow::bail!("device login failed: {error}"),
        }
    }
}

#[cfg(test)]
#[path = "device_login_tests.rs"]
mod tests;
