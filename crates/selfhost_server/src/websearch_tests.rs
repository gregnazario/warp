use std::sync::{Arc, Mutex};

use super::*;

#[tokio::test]
async fn parses_searxng_json_results() {
    let captured_query = Arc::new(Mutex::new(String::new()));
    let captured_in_handler = captured_query.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().route(
        "/search",
        axum::routing::get(move |raw_query: axum::extract::RawQuery| {
            let captured = captured_in_handler.clone();
            async move {
                let query = raw_query
                    .0
                    .as_deref()
                    .and_then(|raw| {
                        raw.split('&').find_map(|pair| {
                            let (key, value) = pair.split_once('=')?;
                            (key == "q").then(|| value.replace('+', " ").replace("%20", " "))
                        })
                    })
                    .unwrap_or_default();
                *captured.lock().unwrap() = query;
                axum::Json(serde_json::json!({
                    "results": [
                        {"title": "Rust book", "url": "https://doc.rust-lang.org", "content": "The Rust programming language."},
                        {"title": "No content", "url": "https://example.com"},
                        {"title": "Ignored 1", "url": "https://ignore1.com", "content": "x"},
                        {"title": "Ignored 2", "url": "https://ignore2.com", "content": "y"},
                        {"title": "Ignored 3", "url": "https://ignore3.com", "content": "z"},
                        {"title": "Ignored 4", "url": "https://ignore4.com", "content": "w"},
                    ]
                }))
            }
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let results = search(
        &reqwest::Client::new(),
        &format!("http://{addr}/search"),
        "what is rust",
    )
    .await
    .unwrap();

    // Capped at MAX_RESULTS, and entries without content still parse.
    assert_eq!(results.len(), MAX_RESULTS);
    assert_eq!(results[0].title, "Rust book");
    assert_eq!(results[0].url, "https://doc.rust-lang.org");
    assert_eq!(results[1].snippet, "");
    assert_eq!(captured_query.lock().unwrap().as_str(), "what is rust");
}

#[tokio::test]
async fn non_json_backend_reply_is_reported() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().route("/search", axum::routing::get(|| async { "boom" }));
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // The stub replies 200 with a non-JSON body, so parsing fails.
    let error = search(
        &reqwest::Client::new(),
        &format!("http://{addr}/search"),
        "q",
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("JSON"));
}

#[tokio::test]
async fn unreachable_backend_is_reported() {
    // Port 1 is never our stub, so the connection fails.
    let error = search(&reqwest::Client::new(), "http://127.0.0.1:1/search", "q")
        .await
        .unwrap_err();
    assert!(error.to_string().contains("unreachable"));
}

#[test]
fn renders_results_for_llm() {
    let results = vec![SearchResult {
        title: "Rust book".to_owned(),
        url: "https://doc.rust-lang.org".to_owned(),
        snippet: "The Rust programming language.".to_owned(),
    }];
    let text = render_for_llm("what is rust", &results);
    assert!(text.contains("what is rust"));
    assert!(text.contains("Rust book"));
    assert!(text.contains("https://doc.rust-lang.org"));

    let empty = render_for_llm("nothing", &[]);
    assert!(empty.contains("No web results"));
}
