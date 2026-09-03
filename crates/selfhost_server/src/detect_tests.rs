use std::sync::{Arc, Mutex};

use super::*;

/// Boots stub backends and asserts probe order, model extraction, and the
/// Ollama /api/tags preference.
#[tokio::test]
async fn probes_candidates_in_order_and_extracts_models() {
    let lmstudio = spawn_models_stub("/v1/models", &["lmstudio-model"]).await;
    let ollama = spawn_models_stub("/v1/models", &["ignored-openai-list"]).await;

    let targets = vec![
        ProbeTarget {
            backend: LlmBackend::Ollama,
            base_url: Box::leak(format!("http://{ollama}").into_boxed_str()),
        },
        ProbeTarget {
            backend: LlmBackend::LmStudio,
            base_url: Box::leak(format!("http://{lmstudio}").into_boxed_str()),
        },
    ];
    let detected = detect(&reqwest::Client::new(), &targets)
        .await
        .expect("probe succeeds");

    // Ollama is probed first and answers with its /api/tags model names.
    assert_eq!(detected.backend, LlmBackend::Ollama);
    assert_eq!(detected.models, vec!["ollama-tagged-model".to_owned()]);
}

#[tokio::test]
async fn falls_through_unresponsive_backends() {
    // Nothing listens on port 1.
    let lmstudio = spawn_models_stub("/v1/models", &["qwen3"]).await;
    let targets = vec![
        ProbeTarget {
            backend: LlmBackend::Ollama,
            base_url: "http://127.0.0.1:1",
        },
        ProbeTarget {
            backend: LlmBackend::LmStudio,
            base_url: Box::leak(format!("http://{lmstudio}").into_boxed_str()),
        },
    ];
    let detected = detect(&reqwest::Client::new(), &targets)
        .await
        .expect("probe succeeds");
    assert_eq!(detected.backend, LlmBackend::LmStudio);
    assert_eq!(detected.models, vec!["qwen3".to_owned()]);
}

#[tokio::test]
async fn no_backends_returns_none() {
    let targets = vec![ProbeTarget {
        backend: LlmBackend::Mlx,
        base_url: "http://127.0.0.1:1",
    }];
    assert!(detect(&reqwest::Client::new(), &targets).await.is_none());
}

/// Serves `{data: [{id: ...}]}` at `path` plus an Ollama-style
/// `/api/tags` with the fixed name "ollama-tagged-model".
async fn spawn_models_stub(
    path: &'static str,
    ids: &'static [&'static str],
) -> std::net::SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let data: Vec<serde_json::Value> = ids.iter().map(|id| serde_json::json!({"id": id})).collect();
    let app = axum::Router::new()
        .route(
            path,
            axum::routing::get(move || {
                let data = data.clone();
                async move { axum::Json(serde_json::json!({ "data": data })) }
            }),
        )
        .route(
            "/api/tags",
            axum::routing::get(|| async {
                axum::Json(serde_json::json!({
                    "models": [{"name": "ollama-tagged-model"}]
                }))
            }),
        );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

// Arc/Mutex keep the unused-import lint quiet when stub helpers change.
#[allow(unused)]
fn _typecheck() {
    let _: Option<Arc<Mutex<()>>> = None;
}
