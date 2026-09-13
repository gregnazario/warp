use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

/// Serves a device-authorization endpoint that answers `authorization_pending`
/// twice and then grants, exercising the polling loop end to end.
#[tokio::test]
async fn device_flow_polls_until_granted() {
    let polls = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let counter = polls.clone();
    let app = axum::Router::new()
        .route(
            "/device/code",
            axum::routing::post(|| async {
                axum::Json(serde_json::json!({
                    "device_code": "dc-1",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "http://127.0.0.1:1/verify",
                    "interval": 0,
                    "expires_in": 60,
                }))
            }),
        )
        .route(
            "/token",
            axum::routing::post(move || {
                let counter = counter.clone();
                async move {
                    if counter.fetch_add(1, Ordering::SeqCst) < 2 {
                        axum::Json(serde_json::json!({"error": "authorization_pending"}))
                    } else {
                        axum::Json(serde_json::json!({
                            "access_token": "at-1",
                            "refresh_token": "rt-1",
                            "expires_in": 3600,
                        }))
                    }
                }
            }),
        );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let flow = DeviceFlow {
        device_url: format!("http://{addr}/device/code"),
        token_url: format!("http://{addr}/token"),
        client_id: "test-client",
        client_secret: Some("test-secret"),
        scope: "scope",
    };
    let mut opened = None;
    let grant = run(&reqwest::Client::new(), &flow, |url, user_code| {
        opened = Some((url.to_owned(), user_code.to_owned()));
    })
    .await
    .unwrap();

    assert_eq!(grant.access_token, "at-1");
    assert_eq!(grant.refresh_token.as_deref(), Some("rt-1"));
    assert_eq!(polls.load(Ordering::SeqCst), 3);
    let (url, user_code) = opened.expect("verification URL handed to opener");
    assert_eq!(user_code, "ABCD-EFGH");
    assert!(url.ends_with("/verify"));
}

#[tokio::test]
async fn device_flow_surfaces_denial() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new()
        .route(
            "/device/code",
            axum::routing::post(|| async {
                axum::Json(serde_json::json!({
                    "device_code": "dc-1",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "http://127.0.0.1:1/verify",
                    "interval": 0,
                    "expires_in": 60,
                }))
            }),
        )
        .route(
            "/token",
            axum::routing::post(|| async {
                axum::Json(serde_json::json!({"error": "access_denied"}))
            }),
        );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let flow = DeviceFlow {
        device_url: format!("http://{addr}/device/code"),
        token_url: format!("http://{addr}/token"),
        client_id: "test-client",
        client_secret: None,
        scope: "scope",
    };
    let error = run(&reqwest::Client::new(), &flow, |_, _| {})
        .await
        .unwrap_err();
    assert!(format!("{error:#}").contains("access_denied"));
}
