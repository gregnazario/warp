use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use serde_json::{Value, json};

use crate::config::Config;
use crate::multi_agent::{AppState, check_auth};

pub async fn graphql(State(state): State<AppState>, headers: HeaderMap, body: String) -> Response {
    if !check_auth(&state, &headers).await {
        return (StatusCode::UNAUTHORIZED, "invalid or missing API key").into_response();
    }

    // The client labels every request with an operation name; serve the
    // operations the self-hosted backend implements and answer everything
    // else with a well-formed GraphQL error so the client degrades locally
    // instead of reaching Warp-operated servers.
    let payload = serde_json::from_str::<Value>(&body).unwrap_or(Value::Null);
    let operation = payload["operationName"]
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_default();
    let variables = &payload["variables"];
    tracing::info!(operation = %operation, "GraphQL request");
    let response = match operation.as_str() {
        "GetUser" => Json(get_user_response(&state.config)).into_response(),
        "GetFeatureModelChoices" => {
            Json(feature_model_choices_response(&state.config)).into_response()
        }
        "GetWorkspacesMetadataForUser" => Json(workspaces_metadata_response()).into_response(),
        "GetUpdatedCloudObjects" => Json(updated_cloud_objects_response()).into_response(),
        "GetRequestLimitInfo" => Json(request_limit_info_response()).into_response(),
        "GetAICreditAvailability" => Json(credit_availability_response()).into_response(),
        "ListAIConversationMetadata" => Json(conversation_metadata_response()).into_response(),
        "FreeAvailableModels" => {
            Json(free_available_models_response(&state.config)).into_response()
        }
        "BulkCreateObjects" => Json(bulk_create_objects_response(variables)).into_response(),
        "GetAvailableHarnesses" => Json(available_harnesses_response()).into_response(),
        "GetUserSettings" => Json(user_settings_response()).into_response(),
        "GetCloudEnvironmentsQuery" => Json(cloud_environments_response()).into_response(),
        "UpdateUserSettings" => Json(update_user_settings_response()).into_response(),
        "CreateGenericStringObject" => {
            Json(create_generic_string_object_response(variables)).into_response()
        }
        _ => {
            // Fall back to the query-text marker for clients (and tests) that
            // post the GetUser query without an operation name.
            if body.contains(GET_USER_MARKER) {
                Json(get_user_response(&state.config)).into_response()
            } else {
                unsupported()
            }
        }
    };
    tracing::info!(operation = %operation, "GraphQL response sent");
    response
}

fn unsupported() -> Response {
    (
        StatusCode::OK,
        Json(json!({
            "data": Value::Null,
            "errors": [{"message":
                "This operation is not supported by the self-hosted agent server."}],
        })),
    )
        .into_response()
}

/// A marker unique to the client's `GetUser` GraphQL query.
const GET_USER_MARKER: &str = "globalSkills";

pub(crate) fn get_user_response(config: &Config) -> Value {
    json!({
        "data": {
            "user": {
                "__typename": "UserOutput",
                "apiKeyOwnerType": Value::Null,
                "principalType": Value::Null,
                "user": {
                    "anonymousUserInfo": Value::Null,
                    "experiments": Value::Null,
                    "globalSkills": [],
                    "isOnboarded": true,
                    "isOnWorkDomain": false,
                    "profile": user_profile(),
                    "llms": catalog(config),
                },
            }
        }
    })
}

/// The `GetFeatureModelChoices` response, which populates the client's model
/// picker from this server's configured models.
pub(crate) fn feature_model_choices_response(config: &Config) -> Value {
    json!({
        "data": {
            "user": {
                "__typename": "UserOutput",
                "user": {
                    "workspaces": [
                        {"featureModelChoice": catalog(config)}
                    ]
                },
            }
        }
    })
}

/// The model catalog served for every feature surface.
fn catalog(config: &Config) -> Value {
    // The catalog lists every configured model in preference order; with no
    // explicit configuration, a single placeholder is served and the client's
    // selection is passed through to the LLM endpoint unchanged.
    let models: Vec<String> = if !config.llm_models.is_empty() {
        config.llm_models.clone()
    } else if let Some(single) = &config.llm_model {
        vec![single.clone()]
    } else {
        vec!["selfhosted-model".to_owned()]
    };
    let available_llms = || {
        json!({
            "defaultId": models[0],
            "choices": models.iter().map(|model| llm_info(model)).collect::<Vec<_>>(),
            "preferredCodexModelId": Value::Null,
        })
    };
    json!({
        "agentMode": available_llms(),
        "planning": available_llms(),
        "coding": available_llms(),
        "cliAgent": available_llms(),
        "computerUseAgent": available_llms(),
    })
}

fn user_profile() -> Value {
    json!({
        "displayName": "Self-Hosted Agent",
        "email": "agent@selfhosted.invalid",
        "needsSsoLink": false,
        "photoUrl": Value::Null,
        "uid": "selfhosted-user0000000",
    })
}

fn llm_info(model: &str) -> Value {
    json!({
        "displayName": model,
        "baseModelName": model,
        "id": model,
        "reasoningLevel": Value::Null,
        "usageMetadata": {"creditMultiplier": Value::Null, "requestMultiplier": 1},
        "description": "Model served by your self-hosted agent server",
        "disableReason": Value::Null,
        "visionSupported": true,
        "spec": Value::Null,
        "provider": "OPENAI",
        "hostConfigs": [],
        "pricing": {"discountPercentage": Value::Null},
        "contextWindow": {
            "isConfigurable": false,
            "min": 1024,
            "max": 1_000_000,
            "default": 128_000,
        },
    })
}

#[cfg(test)]
#[path = "graphql_tests.rs"]
mod tests;
/// The `GetWorkspacesMetadataForUser` response: a single personal workspace
/// with no team, billing, or credit restrictions.
pub(crate) fn workspaces_metadata_response() -> Value {
    json!({
        "data": {
            "user": {
                "__typename": "UserOutput",
                "user": {
                    "profile": {"uid": "selfhosted-user"},
                    "aiCreditAvailability": credit_availability(),
                    "billingMetadata": Value::Null,
                    "workspaces": [],
                    "experiments": Value::Null,
                    "discoverableTeams": [],
                }
            },
            "pricingInfo": {
                "__typename": "PricingInfoOutput",
                "pricingInfo": pricing_info(),
            },
        }
    })
}

/// The `GetUpdatedCloudObjects` (Warp Drive sync) response: nothing changed.
pub(crate) fn updated_cloud_objects_response() -> Value {
    json!({
        "data": {
            "updatedCloudObjects": {
                "__typename": "UpdatedCloudObjectsOutput",
                "actionHistories": Value::Null,
                "deletedObjectUids": {
                    "folderUids": Value::Null,
                    "genericStringObjectUids": Value::Null,
                    "notebookUids": Value::Null,
                    "workflowUids": Value::Null,
                },
                "folders": Value::Null,
                "genericStringObjects": Value::Null,
                "mcpGallery": Value::Null,
                "notebooks": Value::Null,
                "responseContext": response_context(),
                "userProfiles": Value::Null,
                "workflows": Value::Null,
            }
        }
    })
}

/// The `GetRequestLimitInfo` response: unlimited local requests.
pub(crate) fn request_limit_info_response() -> Value {
    json!({
        "data": {
            "user": {
                "__typename": "UserOutput",
                "user": {
                    "workspaces": [],
                    "requestLimitInfo": {
                        "isUnlimited": true,
                        "nextRefreshTime": "2036-01-01T00:00:00Z",
                        "requestLimit": 0,
                        "requestsUsedSinceLastRefresh": 0,
                        "requestLimitRefreshDuration": "MONTHLY",
                        "isUnlimitedVoice": true,
                        "voiceRequestLimit": 0,
                        "voiceRequestsUsedSinceLastRefresh": 0,
                        "isUnlimitedCodebaseIndices": true,
                        "maxCodebaseIndices": 0,
                        "maxFilesPerRepo": 0,
                        "embeddingGenerationBatchSize": 0,
                    },
                    "bonusGrants": [],
                }
            }
        }
    })
}

/// The `GetAICreditAvailability` response: always available, no credits meter.
pub(crate) fn credit_availability_response() -> Value {
    json!({
        "data": {
            "user": {
                "__typename": "UserOutput",
                "user": {"aiCreditAvailability": credit_availability()},
            }
        }
    })
}

/// The `ListAIConversationMetadata` response: no cloud-stored conversations.
pub(crate) fn conversation_metadata_response() -> Value {
    json!({
        "data": {
            "listAIConversations": {
                "__typename": "ListAIConversationsOutput",
                "conversations": [],
                "responseContext": response_context(),
            }
        }
    })
}

/// The `FreeAvailableModels` response: the served catalog, free of charge.
pub(crate) fn free_available_models_response(config: &Config) -> Value {
    json!({
        "data": {
            "freeAvailableModels": {
                "__typename": "FreeAvailableModelsOutput",
                "featureModelChoice": catalog(config),
                "responseContext": response_context(),
            }
        }
    })
}

/// Public pricing with no purchasable plans — everything is local and free.
fn pricing_info() -> Value {
    json!({
        "plans": [{
            "plan": "PRO",
            "monthlyPlanPricePerMonthUsdCents": 0,
            "yearlyPlanPricePerMonthUsdCents": 0,
            "requestLimit": Value::Null,
            "codebaseLimit": 0,
            "codebaseContextFileLimit": 0,
            "maxTeamSize": Value::Null,
        }],
        "overages": {"pricePerRequestUsdCents": 0},
        "addonCreditsOptions": [],
        "promotionMessage": Value::Null,
    })
}

fn credit_availability() -> Value {
    json!({
        "available": true,
        "denialReason": "NONE",
        "creditSource": Value::Null,
    })
}

/// A fixed far-future timestamp for stubbed metadata fields.
const STUB_TIME: &str = "2036-01-01T00:00:00Z";

/// Warp object uids are exactly 22 characters (`ServerId`); fabricated uids
/// must match or the client's sync queue rejects them.
pub(crate) fn stub_uid() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..22].to_owned()
}

fn response_context() -> Value {
    json!({"serverVersion": "selfhosted-server"})
}

/// The `BulkCreateObjects` mutation: acknowledges cloud object uploads without
/// persisting anything (self-hosted state stays local). Each input object gets
/// an output echoing its client id so the client marks it as synced.
pub(crate) fn bulk_create_objects_response(variables: &Value) -> Value {
    let inputs = variables["input"]["genericStringObjects"]["objects"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    json!({
        "data": {
            "bulkCreateObjects": {
                "__typename": "BulkCreateObjectsOutput",
                "genericStringObjects": {
                    "objects": inputs
                        .iter()
                        .map(generic_string_object_output)
                        .collect::<Vec<_>>(),
                },
                "responseContext": response_context(),
            }
        }
    })
}

/// The `CreateGenericStringObject` mutation: acknowledges a cloud object
/// upload without persisting anything.
pub(crate) fn create_generic_string_object_response(variables: &Value) -> Value {
    let input = &variables["input"]["genericStringObject"];
    json!({
        "data": {
            "createGenericStringObject": {
                "__typename": "CreateGenericStringObjectOutput",
                "clientId": input["clientId"],
                "genericStringObject": served_generic_string_object(input),
                "responseContext": response_context(),
                "revisionTs": STUB_TIME,
            }
        }
    })
}

/// Builds a `CreateGenericStringObjectOutput` echoing the input's client id,
/// format, and serialized model.
fn generic_string_object_output(input: &Value) -> Value {
    json!({
        "clientId": input["clientId"],
        "genericStringObject": served_generic_string_object(input),
        "responseContext": response_context(),
        "revisionTs": STUB_TIME,
    })
}

/// A decode-valid `GenericStringObject` mirroring the uploaded one.
fn served_generic_string_object(input: &Value) -> Value {
    json!({
        "format": input["format"],
        "metadata": {
            "creatorUid": Value::Null,
            "currentEditorUid": Value::Null,
            "isWelcomeObject": false,
            "lastEditorUid": Value::Null,
            "metadataLastUpdatedTs": STUB_TIME,
            "parent": {
                "__typename": "Space",
                "uid": "selfhosted-space000000",
                "type": "User",
            },
            "revisionTs": STUB_TIME,
            "trashedTs": Value::Null,
            "uid": stub_uid(),
        },
        "permissions": {
            "guests": [],
            "lastUpdatedTs": STUB_TIME,
            "anyoneLinkSharing": Value::Null,
            "space": {"uid": "selfhosted-space000000", "type": "User"},
        },
        "serializedModel": input["serializedModel"],
    })
}

/// The `GetAvailableHarnesses` response: no cloud harnesses.
pub(crate) fn available_harnesses_response() -> Value {
    json!({
        "data": {
            "user": {
                "__typename": "UserOutput",
                "user": {"availableHarnesses": {"harnesses": []}},
            }
        }
    })
}

/// The `GetUserSettings` response: every cloud-backed feature is off.
pub(crate) fn user_settings_response() -> Value {
    json!({
        "data": {
            "user": {
                "__typename": "UserOutput",
                "user": {
                    "settings": {
                        "isCloudConversationStorageEnabled": false,
                        "isCrashReportingEnabled": false,
                        "isTelemetryEnabled": false,
                    }
                },
            }
        }
    })
}

/// The `GetCloudEnvironmentsQuery` response: no cloud environments.
pub(crate) fn cloud_environments_response() -> Value {
    json!({
        "data": {
            "getCloudEnvironments": {
                "__typename": "GetCloudEnvironmentsOutput",
                "cloudEnvironments": [],
                "responseContext": response_context(),
            }
        }
    })
}

/// The `UpdateUserSettings` mutation: acknowledges settings writes (the served
/// settings themselves always report cloud features as disabled).
pub(crate) fn update_user_settings_response() -> Value {
    json!({
        "data": {
            "updateUserSettings": {
                "__typename": "UpdateUserSettingsOutput",
                "responseContext": response_context(),
            }
        }
    })
}
