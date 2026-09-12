use super::*;

#[test]
fn get_user_response_has_the_shape_the_client_expects() {
    let response = get_user_response(&["qwen3-coder".to_owned()]);
    let user = &response["data"]["user"];
    assert_eq!(user["__typename"], "UserOutput");
    assert_eq!(user["user"]["isOnboarded"], true);
    assert_eq!(user["user"]["profile"]["uid"], "selfhosted-user0000000");
    let agent_mode = &user["user"]["llms"]["agentMode"];
    assert_eq!(agent_mode["defaultId"], "qwen3-coder");
    assert_eq!(agent_mode["choices"][0]["id"], "qwen3-coder");
    assert_eq!(
        agent_mode["choices"][0]["contextWindow"]["isConfigurable"],
        false
    );
}

#[test]
fn unconfigured_model_gets_a_placeholder_name() {
    let response = get_user_response(&[]);
    assert_eq!(
        response["data"]["user"]["user"]["llms"]["agentMode"]["defaultId"],
        "selfhosted-model"
    );
}

#[tokio::test]
async fn catalog_models_reprobes_the_llm_endpoint_when_stale() {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::metrics::MetricsHandle;
    use crate::multi_agent::ModelCache;

    let served: Arc<std::sync::RwLock<Vec<String>>> =
        Arc::new(std::sync::RwLock::new(vec!["before".to_owned()]));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let endpoint_models = served.clone();
    tokio::spawn(async move {
        axum::serve(
            listener,
            axum::Router::new().route(
                "/v1/models",
                axum::routing::get(move || {
                    let endpoint_models = endpoint_models.clone();
                    async move {
                        let ids = endpoint_models.read().unwrap().clone();
                        axum::Json(serde_json::json!({"data": ids.iter()
                                .map(|id| serde_json::json!({"id": id}))
                                .collect::<Vec<_>>()}))
                    }
                }),
            ),
        )
        .await
        .unwrap();
    });

    let state = AppState {
        model_cache: Arc::new(std::sync::Mutex::new(ModelCache {
            models: vec!["startup-model".to_owned()],
            refreshed_at: None,
        })),
        config: Arc::new(crate::config::Config {
            llm_base_url: format!("http://{addr}/v1"),
            ..crate::config::Config::test_default()
        }),
        http: reqwest::Client::new(),
        token_cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
        metrics: MetricsHandle::new(),
    };

    // A stale cache is replaced by the endpoint's live list.
    assert_eq!(catalog_models(&state).await, vec!["before".to_owned()]);

    // The endpoint's list changes; after the TTL the probe picks it up and a
    // new model appears without a backend restart.
    *served.write().unwrap() = vec!["before".to_owned(), "added-later".to_owned()];
    *state.model_cache.lock().unwrap() = crate::multi_agent::ModelCache {
        models: vec!["before".to_owned()],
        refreshed_at: Some(instant::Instant::now() - std::time::Duration::from_secs(31)),
    };
    assert_eq!(
        catalog_models(&state).await,
        vec!["before".to_owned(), "added-later".to_owned()]
    );
}
