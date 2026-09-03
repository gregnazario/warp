//! Verifies that the self-hosted server's hand-written `GetUser` response
//! decodes through the exact cynic deserializer the client uses, so a schema
//! drift breaks here instead of at client login.

use cynic::GraphQlResponse;
use warp_graphql::queries::get_user::GetUser;

#[tokio::test]
async fn selfhost_get_user_response_decodes_with_the_client_schema() {
    let app = selfhost_server::router(test_config());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // The client posts the GetUser operation; the server answers it with its
    // static user record.
    let client = reqwest::Client::new();
    let response = client
        .post(format!("http://{addr}/graphql/v2"))
        .json(&serde_json::json!({
            "operationName": "GetUser",
            "query": "query GetUser { user { globalSkills isOnboarded profile { uid } llms { agentMode { defaultId } } } }",
        }))
        .send()
        .await
        .unwrap();
    let raw: serde_json::Value = response.json().await.unwrap();

    let decoded: GraphQlResponse<GetUser> = serde_json::from_value(raw)
        .expect("self-hosted GetUser response must decode with the client's schema");
    let data = decoded.data.expect("response must include data");
    let user = match data.user {
        warp_graphql::queries::get_user::UserResult::UserOutput(output) => output,
        warp_graphql::queries::get_user::UserResult::Unknown => {
            panic!("self-hosted user record decoded as Unknown")
        }
    };
    assert!(user.user.is_onboarded);
    assert_eq!(user.user.profile.uid, "selfhosted-user0000000");
    assert!(!user.user.llms.agent_mode.choices.is_empty());
}

fn test_config() -> selfhost_server::Config {
    selfhost_server::Config {
        llm_base_url: String::new(),
        llm_api_key: None,
        llm_schema: selfhost_server::config::LlmSchema::Openai,
        llm_model: Some("test-model".to_owned()),
        llm_models: Vec::new(),
        api_key: None,
        system_prompt: None,
        context_window_tokens: 0,
        web_search_url: None,
        byok_direct: false,
        auth_introspect_url: None,
        transcribe_base_url: None,
        transcribe_api_key: None,
        transcribe_model: "whisper-1".to_owned(),
    }
}

#[tokio::test]
async fn selfhost_feature_model_choices_response_decodes_with_the_client_schema() {
    let app = selfhost_server::router(test_config());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let client = reqwest::Client::new();
    let response = client
        .post(format!("http://{addr}/graphql/v2"))
        .json(&serde_json::json!({
            "operationName": "GetFeatureModelChoices",
            "query": "query GetFeatureModelChoices { user { workspaces { featureModelChoice { agentMode { defaultId choices { id } } } } } }",
        }))
        .send()
        .await
        .unwrap();
    let raw: serde_json::Value = response.json().await.unwrap();

    let decoded: GraphQlResponse<
        warp_graphql::queries::get_feature_model_choices::GetFeatureModelChoices,
    > = serde_json::from_value(raw)
        .expect("self-hosted model catalog must decode with the client's schema");
    let data = decoded.data.expect("response must include data");
    let user = match data.user {
        warp_graphql::queries::get_feature_model_choices::UserResult::UserOutput(output) => output,
        warp_graphql::queries::get_feature_model_choices::UserResult::Unknown => {
            panic!("model catalog decoded as Unknown")
        }
    };
    let workspace = user
        .user
        .workspaces
        .first()
        .expect("catalog must include a workspace");
    assert_eq!(
        workspace.feature_model_choice.agent_mode.default_id,
        "test-model"
    );
    // The provider enum must decode to a real variant, not cynic's fallback:
    // the client converts the fallback into a reported error.
    let choice = &workspace.feature_model_choice.agent_mode.choices[0];
    assert!(matches!(
        choice.provider,
        warp_graphql::queries::get_feature_model_choices::LlmProvider::Openai
    ));
}

/// Every operation the self-hosted server stubs must decode through the
/// client's cynic schema, or the client logs errors and retries forever.
#[tokio::test]
async fn selfhost_stub_responses_decode_with_the_client_schema() {
    let app = selfhost_server::router(test_config());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();

    let post = |operation: &str, query: &str| {
        let url = format!("http://{addr}/graphql/v2");
        let body = serde_json::json!({"operationName": operation, "query": query});
        let client = client.clone();
        async move { client.post(url).json(&body).send().await.unwrap() }
    };

    // GetWorkspacesMetadataForUser
    let raw: serde_json::Value = post(
        "GetWorkspacesMetadataForUser",
        "query GetWorkspacesMetadataForUser { user { profile { uid } workspaces { featureModelChoice { defaultId } } } pricingInfo { __typename } }",
    )
    .await
    .json()
    .await
    .unwrap();
    let decoded: GraphQlResponse<
        warp_graphql::queries::get_workspaces_metadata_for_user::GetWorkspacesMetadataForUser,
    > = serde_json::from_value(raw)
        .expect("workspaces metadata must decode with the client's schema");
    let data = decoded.data.expect("data present");
    let user = match data.user {
        warp_graphql::queries::get_workspaces_metadata_for_user::UserResult::UserOutput(output) => {
            output.user
        }
        _ => panic!("workspaces metadata decoded as Unknown"),
    };
    assert!(user.workspaces.is_empty());

    // GetUpdatedCloudObjects
    let raw: serde_json::Value = post(
        "GetUpdatedCloudObjects",
        "query GetUpdatedCloudObjects { updatedCloudObjects { deletedObjectUids { folderUids } responseContext { serverVersion } } }",
    )
    .await
    .json()
    .await
    .unwrap();
    let decoded: GraphQlResponse<
        warp_graphql::queries::get_updated_cloud_objects::GetUpdatedCloudObjects,
    > = serde_json::from_value(raw)
        .expect("updated cloud objects must decode with the client's schema");
    match decoded.data.expect("data present").updated_cloud_objects {
        warp_graphql::queries::get_updated_cloud_objects::UpdatedCloudObjectsResult::UpdatedCloudObjectsOutput(output) => {
            assert!(output.action_histories.is_none());
        }
        _ => panic!("updated cloud objects decoded as Unknown"),
    }

    // GetRequestLimitInfo
    let raw: serde_json::Value = post(
        "GetRequestLimitInfo",
        "query GetRequestLimitInfo { user { workspaces { uid } requestLimitInfo { isUnlimited } bonusGrants { reason } } }",
    )
    .await
    .json()
    .await
    .unwrap();
    let decoded: GraphQlResponse<
        warp_graphql::queries::get_request_limit_info::GetRequestLimitInfo,
    > = serde_json::from_value(raw)
        .expect("request limit info must decode with the client's schema");
    let user = match decoded.data.expect("data present").user {
        warp_graphql::queries::get_request_limit_info::UserResult::UserOutput(output) => {
            output.user
        }
        _ => panic!("request limit info decoded as Unknown"),
    };
    assert!(user.request_limit_info.is_unlimited);

    // GetAICreditAvailability
    let raw: serde_json::Value = post(
        "GetAICreditAvailability",
        "query GetAICreditAvailability { user { aiCreditAvailability { available denialReason } } }",
    )
    .await
    .json()
    .await
    .unwrap();
    let decoded: GraphQlResponse<
        warp_graphql::queries::get_ai_credit_availability::GetAICreditAvailability,
    > = serde_json::from_value(raw)
        .expect("credit availability must decode with the client's schema");
    let user = match decoded.data.expect("data present").user {
        warp_graphql::queries::get_ai_credit_availability::UserResult::UserOutput(output) => {
            output.user
        }
        _ => panic!("credit availability decoded as Unknown"),
    };
    assert!(user.ai_credit_availability.available);

    // ListAIConversationMetadata
    let raw: serde_json::Value = post(
        "ListAIConversationMetadata",
        "query ListAIConversationMetadata { listAIConversations { conversations { conversationId } responseContext { serverVersion } } }",
    )
    .await
    .json()
    .await
    .unwrap();
    let decoded: GraphQlResponse<
        warp_graphql::queries::list_ai_conversations::ListAIConversationMetadata,
    > = serde_json::from_value(raw)
        .expect("conversation metadata must decode with the client's schema");
    match decoded
        .data
        .expect("data present")
        .list_ai_conversations
    {
        warp_graphql::queries::list_ai_conversations::ListAIConversationMetadataResult::ListAIConversationsOutput(output) => {
            assert!(output.conversations.is_empty());
        }
        _ => panic!("conversation metadata decoded as Unknown"),
    }

    // FreeAvailableModels
    let raw: serde_json::Value = post(
        "FreeAvailableModels",
        "query FreeAvailableModels { freeAvailableModels { featureModelChoice { defaultId } responseContext { serverVersion } } }",
    )
    .await
    .json()
    .await
    .unwrap();
    let decoded: GraphQlResponse<
        warp_graphql::queries::free_available_models::FreeAvailableModels,
    > = serde_json::from_value(raw)
        .expect("free available models must decode with the client's schema");
    match decoded.data.expect("data present").free_available_models {
        warp_graphql::queries::free_available_models::FreeAvailableModelsResult::FreeAvailableModelsOutput(output) => {
            assert!(!output.feature_model_choice.agent_mode.choices.is_empty());
        }
        _ => panic!("free available models decoded as Unknown"),
    }
}

#[tokio::test]
async fn selfhost_straggler_stub_responses_decode_with_the_client_schema() {
    let app = selfhost_server::router(test_config());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();

    let post = |operation: &'static str, query: &'static str| {
        let url = format!("http://{addr}/graphql/v2");
        let client = client.clone();
        async move {
            client
                .post(url)
                .json(&serde_json::json!({"operationName": operation, "query": query}))
                .send()
                .await
                .unwrap()
                .json::<serde_json::Value>()
                .await
                .unwrap()
        }
    };

    // BulkCreateObjects
    let raw = post(
        "BulkCreateObjects",
        "mutation BulkCreateObjects { bulkCreateObjects { genericStringObjects { objects { uid } } responseContext { serverVersion } } }",
    )
    .await;
    let decoded: GraphQlResponse<warp_graphql::mutations::bulk_create_objects::BulkCreateObjects> =
        serde_json::from_value(raw)
            .expect("bulk create objects must decode with the client's schema");
    match decoded
        .data
        .expect("data present")
        .bulk_create_objects
    {
        warp_graphql::mutations::bulk_create_objects::BulkCreateObjectsResult::BulkCreateObjectsOutput(output) => {
            // No input objects were sent, so none come back.
            assert!(
                output
                    .generic_string_objects
                    .expect("output present")
                    .objects
                    .is_empty()
            );
        }
        _ => panic!("bulk create objects decoded as Unknown"),
    }

    // GetAvailableHarnesses
    let raw = post(
        "GetAvailableHarnesses",
        "query GetAvailableHarnesses { user { availableHarnesses { harnesses { harness displayName } } } }",
    )
    .await;
    let decoded: GraphQlResponse<
        warp_graphql::queries::get_available_harnesses::GetAvailableHarnesses,
    > = serde_json::from_value(raw)
        .expect("available harnesses must decode with the client's schema");
    let user = match decoded.data.expect("data present").user {
        warp_graphql::queries::get_available_harnesses::UserResult::UserOutput(output) => {
            output.user
        }
        _ => panic!("available harnesses decoded as Unknown"),
    };
    assert!(user.available_harnesses.harnesses.is_empty());

    // GetUserSettings
    let raw = post(
        "GetUserSettings",
        "query GetUserSettings { user { settings { isCloudConversationStorageEnabled } } }",
    )
    .await;
    let decoded: GraphQlResponse<warp_graphql::queries::get_user_settings::GetUserSettings> =
        serde_json::from_value(raw).expect("user settings must decode with the client's schema");
    let user = match decoded.data.expect("data present").user {
        warp_graphql::queries::get_user_settings::UserResult::UserOutput(output) => output.user,
        _ => panic!("user settings decoded as Unknown"),
    };
    assert!(
        !user
            .settings
            .expect("settings present")
            .is_cloud_conversation_storage_enabled
    );

    // GetCloudEnvironmentsQuery
    let raw = post(
        "GetCloudEnvironmentsQuery",
        "query GetCloudEnvironmentsQuery { getCloudEnvironments { cloudEnvironments { uid } responseContext { serverVersion } } }",
    )
    .await;
    let decoded: GraphQlResponse<
        warp_graphql::queries::get_cloud_environments::GetCloudEnvironmentsQuery,
    > = serde_json::from_value(raw)
        .expect("cloud environments must decode with the client's schema");
    match decoded.data.expect("data present").get_cloud_environments {
        warp_graphql::queries::get_cloud_environments::GetCloudEnvironmentsResult::GetCloudEnvironmentsOutput(output) => {
            assert!(output.cloud_environments.is_empty());
        }
        _ => panic!("cloud environments decoded as Unknown"),
    }
}
