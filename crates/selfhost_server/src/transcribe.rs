use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde_json::{Value, json};

use crate::multi_agent::{AppState, check_auth};

/// Handles the Warp client's `POST /ai/transcribe` request.
///
/// The client sends a JSON body with a base64-encoded 16 kHz mono WAV file
/// (`audio`), an optional ISO-639-1 `language`, and optional context under
/// `prompt` / `wispr_properties`. The audio is forwarded to the configured
/// OpenAI-compatible speech-to-text endpoint and the transcription is returned
/// as `{"text": ...}`.
pub async fn transcribe(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Response {
    if !check_auth(&state, &headers).await {
        return unauthorized();
    }

    let Some(base_url) = state
        .config
        .transcribe_base_url
        .as_ref()
        .filter(|url| !url.is_empty())
    else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "error": "Voice transcription is not configured on this server. \
                          Start it with --transcribe-base-url pointing at an OpenAI-compatible \
                          speech-to-text endpoint."
            })),
        )
            .into_response();
    };

    let request: Value = match serde_json::from_str(&body) {
        Ok(request) => request,
        Err(_) => return bad_request("undecodable request body"),
    };
    let Some(audio_base64) = request["audio"].as_str() else {
        return bad_request("request is missing 'audio'");
    };
    let audio = match BASE64_STANDARD.decode(audio_base64) {
        Ok(audio) => audio,
        Err(_) => return bad_request("'audio' is not valid base64"),
    };
    if audio.is_empty() {
        return bad_request("'audio' is empty");
    }

    let mut prompt = request["prompt"].as_str().unwrap_or_default().to_owned();
    // Wispr-style context can guide the transcription even without an explicit
    // prompt.
    let wispr = &request["wispr_properties"];
    for key in ["before_text", "selected_text", "after_text", "dictionary"] {
        if let Some(text) = wispr[key].as_str()
            && !text.is_empty()
        {
            if !prompt.is_empty() {
                prompt.push(' ');
            }
            prompt.push_str(text);
        }
    }

    let mut form = reqwest::multipart::Form::new()
        .part(
            "file",
            reqwest::multipart::Part::bytes(audio)
                .file_name("audio.wav")
                .mime_str("audio/wav")
                .expect("audio/wav is a valid mime type"),
        )
        .text("model", state.config.transcribe_model.clone())
        .text("response_format", "json");
    if let Some(language) = request["language"].as_str()
        && !language.is_empty()
    {
        form = form.text("language", language.to_owned());
    }
    if !prompt.is_empty() {
        form = form.text("prompt", prompt);
    }

    let mut request_builder = state
        .http
        .post(format!(
            "{}/audio/transcriptions",
            base_url.trim_end_matches('/')
        ))
        .multipart(form);
    if let Some(key) = &state.config.transcribe_api_key {
        request_builder = request_builder.bearer_auth(key);
    }

    let response = match request_builder.send().await {
        Ok(response) => response,
        Err(error) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": format!("speech-to-text endpoint unreachable: {error}")})),
            )
                .into_response();
        }
    };

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({
                "error": format!("speech-to-text endpoint returned {status}: {}", body.trim())
            })),
        )
            .into_response();
    }

    // OpenAI-compatible endpoints reply with `{"text": ...}` for JSON format;
    // accept a plain-text body too.
    let text = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|value| value["text"].as_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| body.trim().to_owned());
    Json(json!({ "text": text })).into_response()
}

fn unauthorized() -> Response {
    (StatusCode::UNAUTHORIZED, "invalid or missing API key").into_response()
}

fn bad_request(message: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error": message}))).into_response()
}

#[cfg(test)]
#[path = "transcribe_tests.rs"]
mod tests;
