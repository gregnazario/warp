use std::sync::{Arc, Mutex};

use super::*;
use crate::config::Config;

/// Captured request from the stub speech-to-text endpoint.
type CapturedRequest = (String, Vec<u8>);

#[derive(Clone)]
struct StubStt {
    requests: Arc<Mutex<Vec<CapturedRequest>>>,
    /// Response returned per request, popped front-first.
    replies: Arc<Mutex<Vec<(StatusCode, String)>>>,
}

async fn spawn_stub_stt(stub: StubStt) -> std::net::SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().route(
        "/v1/audio/transcriptions",
        axum::routing::post(move |headers: HeaderMap, body: axum::body::Bytes| {
            let stub = stub.clone();
            async move {
                let content_type = headers
                    .get("content-type")
                    .and_then(|value| value.to_str().ok())
                    .unwrap_or_default()
                    .to_owned();
                stub.requests
                    .lock()
                    .unwrap()
                    .push((content_type, body.to_vec()));
                let (status, text) = stub
                    .replies
                    .lock()
                    .unwrap()
                    .pop()
                    .unwrap_or((StatusCode::OK, "{\"text\": \"\"}".to_owned()));
                (status, text)
            }
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

fn config_with_stt(stt_addr: std::net::SocketAddr) -> Config {
    Config {
        transcribe_base_url: Some(format!("http://{stt_addr}/v1")),
        ..Config::test_default()
    }
}

fn config_without_stt() -> Config {
    Config {
        transcribe_base_url: None,
        transcribe_model: "whisper-1".to_owned(),
        ..config_with_stt("127.0.0.1:1".parse().unwrap())
    }
}

async fn spawn_server(config: Config) -> std::net::SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, crate::multi_agent::router(config))
            .await
            .unwrap()
    });
    addr
}

fn transcribe_body(audio: &[u8], language: Option<&str>) -> String {
    let mut body = json!({
        "provider": "wispr",
        "audio": BASE64_STANDARD.encode(audio),
    });
    if let Some(language) = language {
        body["language"] = json!(language);
    }
    body["wispr_properties"] = json!({"dictionary": "warp warpdb"});
    body.to_string()
}

#[tokio::test]
async fn transcribes_audio_via_stt_backend() {
    let wav: &[u8] = b"RIFFfake-wave-data";
    let stub = StubStt {
        requests: Arc::new(Mutex::new(Vec::new())),
        replies: Arc::new(Mutex::new(vec![(
            StatusCode::OK,
            "{\"text\": \"hello world\"}".to_owned(),
        )])),
    };
    let stt_addr = spawn_stub_stt(stub.clone()).await;
    let addr = spawn_server(config_with_stt(stt_addr)).await;

    let response = reqwest::Client::new()
        .post(format!("http://{addr}/ai/transcribe"))
        .json(&serde_json::from_str::<Value>(&transcribe_body(wav, Some("en"))).unwrap())
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let reply: Value = response.json().await.unwrap();
    assert_eq!(reply["text"], "hello world");

    let (content_type, body) = stub.requests.lock().unwrap().remove(0);
    assert!(
        content_type.starts_with("multipart/form-data"),
        "{content_type}"
    );
    let body = String::from_utf8_lossy(&body);
    assert!(body.contains("name=\"model\"\r\n\r\nwhisper-1"), "{body}");
    assert!(body.contains("name=\"language\"\r\n\r\nen"), "{body}");
    assert!(
        body.contains("name=\"prompt\"") && body.contains("warp warpdb"),
        "wispr dictionary should reach the prompt: {body}"
    );
    assert!(
        body.contains("name=\"response_format\"\r\n\r\njson"),
        "{body}"
    );
    // The audio part carries the decoded WAV bytes.
    assert!(body.contains("filename=\"audio.wav\""), "{body}");
    assert!(
        body.as_bytes().windows(wav.len()).any(|w| w == wav),
        "{body}"
    );
}

#[tokio::test]
async fn accepts_plain_text_stt_reply() {
    let stub = StubStt {
        requests: Arc::new(Mutex::new(Vec::new())),
        replies: Arc::new(Mutex::new(vec![(StatusCode::OK, "hello there".to_owned())])),
    };
    let stt_addr = spawn_stub_stt(stub).await;
    let addr = spawn_server(config_with_stt(stt_addr)).await;

    let response = reqwest::Client::new()
        .post(format!("http://{addr}/ai/transcribe"))
        .body(transcribe_body(b"riff", None))
        .header("content-type", "application/json")
        .send()
        .await
        .unwrap();
    let reply: Value = response.json().await.unwrap();
    assert_eq!(reply["text"], "hello there");
}

#[tokio::test]
async fn surfaces_stt_backend_errors() {
    let stub = StubStt {
        requests: Arc::new(Mutex::new(Vec::new())),
        replies: Arc::new(Mutex::new(vec![(
            StatusCode::INTERNAL_SERVER_ERROR,
            "gpu exploded".to_owned(),
        )])),
    };
    let stt_addr = spawn_stub_stt(stub).await;
    let addr = spawn_server(config_with_stt(stt_addr)).await;

    let response = reqwest::Client::new()
        .post(format!("http://{addr}/ai/transcribe"))
        .body(transcribe_body(b"riff", None))
        .header("content-type", "application/json")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let reply: Value = response.json().await.unwrap();
    assert!(
        reply["error"]
            .as_str()
            .is_some_and(|error| error.contains("gpu exploded"))
    );
}

#[tokio::test]
async fn transcription_disabled_returns_not_implemented() {
    let addr = spawn_server(config_without_stt()).await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/ai/transcribe"))
        .body(transcribe_body(b"riff", None))
        .header("content-type", "application/json")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    let reply: Value = response.json().await.unwrap();
    assert!(
        reply["error"]
            .as_str()
            .is_some_and(|error| error.contains("--transcribe-base-url"))
    );
}

#[tokio::test]
async fn rejects_bad_requests() {
    let stub = StubStt {
        requests: Arc::new(Mutex::new(Vec::new())),
        replies: Arc::new(Mutex::new(Vec::new())),
    };
    let stt_addr = spawn_stub_stt(stub).await;
    let addr = spawn_server(config_with_stt(stt_addr)).await;
    let client = reqwest::Client::new();

    let missing_audio = client
        .post(format!("http://{addr}/ai/transcribe"))
        .body(json!({"provider": "wispr"}).to_string())
        .header("content-type", "application/json")
        .send()
        .await
        .unwrap();
    assert_eq!(missing_audio.status(), StatusCode::BAD_REQUEST);

    let bad_base64 = client
        .post(format!("http://{addr}/ai/transcribe"))
        .body(json!({"audio": "!!!not base64!!!"}).to_string())
        .header("content-type", "application/json")
        .send()
        .await
        .unwrap();
    assert_eq!(bad_base64.status(), StatusCode::BAD_REQUEST);

    let undecodable = client
        .post(format!("http://{addr}/ai/transcribe"))
        .body("not json")
        .header("content-type", "application/json")
        .send()
        .await
        .unwrap();
    assert_eq!(undecodable.status(), StatusCode::BAD_REQUEST);
}
