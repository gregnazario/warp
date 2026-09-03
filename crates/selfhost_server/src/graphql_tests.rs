use super::*;

#[test]
fn get_user_response_has_the_shape_the_client_expects() {
    let config = crate::config::Config {
        llm_model: Some("qwen3-coder".to_owned()),
        ..crate::config::Config::test_default()
    };
    let response = get_user_response(&config);
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
    let config = crate::config::Config::test_default();
    let response = get_user_response(&config);
    assert_eq!(
        response["data"]["user"]["user"]["llms"]["agentMode"]["defaultId"],
        "selfhosted-model"
    );
}
