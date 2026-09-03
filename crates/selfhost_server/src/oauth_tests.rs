use super::*;
use crate::config::DynamicAuth;

#[tokio::test]
async fn azure_client_credentials_exchanges_and_caches() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let requests_in_handler = requests.clone();
    let app = axum::Router::new().route(
        "/token",
        axum::routing::post(move |body: String| {
            let requests = requests_in_handler.clone();
            async move {
                requests.lock().unwrap().push(body);
                axum::Json(serde_json::json!({"access_token": "azure-token", "expires_in": 3600}))
            }
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let auth = DynamicAuth::AzureClientCredentials {
        token_url: format!("http://{addr}/token"),
        client_id: "client".to_owned(),
        client_secret: "secret".to_owned(),
        scope: "https://cognitiveservices.azure.com/.default".to_owned(),
    };
    let cache = std::sync::Mutex::new(HashMap::new());
    let http = reqwest::Client::new();

    let token = resolve_token(&http, &auth, &cache).await.unwrap();
    assert_eq!(token, "azure-token");
    // Second call is served from the cache without another HTTP request.
    let token = resolve_token(&http, &auth, &cache).await.unwrap();
    assert_eq!(token, "azure-token");
    assert_eq!(requests.lock().unwrap().len(), 1);

    let form = requests.lock().unwrap()[0].clone();
    assert!(form.contains("grant_type=client_credentials"));
    assert!(form.contains("client_id=client"));
    assert!(form.contains("scope=https%3A%2F%2Fcognitiveservices.azure.com%2F.default"));
}

#[tokio::test]
async fn vertex_adc_refreshes_and_surfaces_errors() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().route(
        "/token",
        axum::routing::post(|| async {
            axum::Json(serde_json::json!({"access_token": "vertex-token", "expires_in": 30}))
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let auth = DynamicAuth::VertexAdc {
        token_url: format!("http://{addr}/token"),
        client_id: "id".to_owned(),
        client_secret: "secret".to_owned(),
        refresh_token: "refresh".to_owned(),
    };
    let cache = std::sync::Mutex::new(HashMap::new());
    let token = resolve_token(&reqwest::Client::new(), &auth, &cache)
        .await
        .unwrap();
    assert_eq!(token, "vertex-token");

    // A rejected refresh surfaces the endpoint error.
    let dead = DynamicAuth::VertexAdc {
        token_url: "http://127.0.0.1:1/token".to_owned(),
        client_id: "id".to_owned(),
        client_secret: "secret".to_owned(),
        refresh_token: "refresh".to_owned(),
    };
    let error = resolve_token(&reqwest::Client::new(), &dead, &cache)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("unreachable"), "{error}");
}

#[tokio::test]
async fn expired_cache_entries_refetch() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let count = std::sync::Arc::new(std::sync::Mutex::new(0));
    let count_in_handler = count.clone();
    let app = axum::Router::new().route(
        "/token",
        axum::routing::post(move || {
            let count = count_in_handler.clone();
            async move {
                *count.lock().unwrap() += 1;
                axum::Json(serde_json::json!({"access_token": format!("t{}", count.lock().unwrap()), "expires_in": 0}))
            }
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let auth = DynamicAuth::AzureClientCredentials {
        token_url: format!("http://{addr}/token"),
        client_id: "client".to_owned(),
        client_secret: "secret".to_owned(),
        scope: "scope".to_owned(),
    };
    let cache = std::sync::Mutex::new(HashMap::new());
    let http = reqwest::Client::new();
    // expires_in = 0 means already expired: every call refetches.
    let first = resolve_token(&http, &auth, &cache).await.unwrap();
    let second = resolve_token(&http, &auth, &cache).await.unwrap();
    assert_eq!(first, "t1");
    assert_eq!(second, "t2");
}
