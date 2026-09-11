use serde_json::Value;
use warp_multi_agent_api::request::settings::custom_model_providers::CustomEndpointSchema;

/// A local LLM backend the server can autodetect and talk to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmBackend {
    Auto,
    Ollama,
    LmStudio,
    Mlx,
}

impl std::str::FromStr for LlmBackend {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "ollama" => Ok(Self::Ollama),
            "lmstudio" | "lm-studio" => Ok(Self::LmStudio),
            "mlx" => Ok(Self::Mlx),
            other => anyhow::bail!(
                "unknown LLM backend '{other}' (expected auto, ollama, lmstudio, or mlx)"
            ),
        }
    }
}

impl std::fmt::Display for LlmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Ollama => f.write_str("Ollama"),
            Self::LmStudio => f.write_str("LM Studio"),
            Self::Mlx => f.write_str("MLX"),
        }
    }
}

/// Runtime configuration for the self-hosted agent server.
#[derive(Debug, Clone)]
pub struct Config {
    /// Base URL of the LLM endpoint. `/chat/completions` (OpenAI) or
    /// `/messages` (Anthropic) is appended depending on the schema.
    pub llm_base_url: String,
    pub llm_api_key: Option<String>,
    pub llm_schema: LlmSchema,
    /// Model name sent to the LLM endpoint. When `None`, the model selected in
    /// the Warp client is passed through.
    pub llm_model: Option<String>,
    /// When set, Warp clients must present this bearer token.
    pub api_key: Option<String>,
    /// Replacement for the default agent system prompt.
    pub system_prompt: Option<String>,
    /// Base URL of an OpenAI-compatible speech-to-text endpoint;
    /// `/audio/transcriptions` is appended. Voice transcription is disabled
    /// when this is `None`.
    pub transcribe_base_url: Option<String>,
    pub transcribe_api_key: Option<String>,
    /// Model name sent to the speech-to-text endpoint.
    pub transcribe_model: String,
    /// Catalog of model names exposed to the Warp client's model picker, in
    /// preference order (the first is the default). The selected ID is passed
    /// through to the LLM endpoint. `llm_model` overrides this.
    pub llm_models: Vec<String>,
    /// Context-window budget (in estimated tokens) at which the server
    /// compacts older conversation history into a summary. `0` means auto:
    /// honor the client's per-model limit when set, else 96k.
    pub context_window_tokens: u64,
    /// SearXNG-compatible search endpoint (`?q=<query>&format=json` is
    /// appended). When set, the server offers a `web_search` tool to the LLM
    /// and executes searches itself; results never leave this server except
    /// to the search endpoint.
    pub web_search_url: Option<String>,
    /// Route requests to the provider named by the client's BYOK API keys
    /// (settings.api_keys) based on the requested model's name, instead of to
    /// the server's own LLM endpoint.
    pub byok_direct: bool,
    /// Force all non-custom-endpoint traffic to this provider.
    pub provider: Option<ProviderKind>,
    /// API keys for explicit provider routing.
    pub openai_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub google_api_key: Option<String>,
    pub xai_api_key: Option<String>,
    /// Z.ai coding-plan API key (https://api.z.ai/api/coding/paas/v4).
    pub zai_api_key: Option<String>,
    /// OpenCode Zen API key (https://opencode.ai/zen/v1).
    pub opencode_api_key: Option<String>,
    /// Azure AI Foundry (Entra ID client-credentials OAuth).
    pub azure_tenant: Option<String>,
    pub azure_client_id: Option<String>,
    pub azure_client_secret: Option<String>,
    /// Foundry endpoint prefix, e.g.
    /// `https://<resource>.services.ai.azure.com/models`.
    pub azure_foundry_url: Option<String>,
    pub azure_api_version: String,
    /// Google Vertex AI via gcloud application-default credentials.
    pub vertex_enabled: bool,
    /// ChatGPT OAuth token cache for `--provider chatgpt`.
    pub codex_token_file: String,
    pub vertex_base_url: String,
    pub vertex_adc_path: Option<String>,
    /// When set, every client request must present a bearer token that this
    /// URL accepts (POSTed the token as JSON `{"token": ...}`, expects a 2xx
    /// response). This lets an external auth service (SSO, SSO gateways,
    /// Foundry-style auth brokers) own authentication instead of static keys.
    pub auth_introspect_url: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmSchema {
    Openai,
    Anthropic,
    /// ChatGPT-backend Codex endpoint, speaking the OpenAI Responses API.
    Chatgpt,
}

impl std::str::FromStr for LlmSchema {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "chatgpt" | "codex" | "openai-codex" => Ok(Self::Chatgpt),
            "openai" => Ok(Self::Openai),
            "anthropic" => Ok(Self::Anthropic),
            other => {
                anyhow::bail!("unknown LLM schema '{other}' (expected 'openai' or 'anthropic')")
            }
        }
    }
}

/// A remote LLM provider the server can route traffic to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Chatgpt,
    Openai,
    Anthropic,
    Google,
    OpenRouter,
    Xai,
    Zai,
    Opencode,
    AzureFoundry,
    Vertex,
}

impl std::str::FromStr for ProviderKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "chatgpt" | "codex" | "openai-codex" => Ok(Self::Chatgpt),
            "openai" => Ok(Self::Openai),
            "anthropic" => Ok(Self::Anthropic),
            "google" => Ok(Self::Google),
            "openrouter" => Ok(Self::OpenRouter),
            "xai" | "grok" => Ok(Self::Xai),
            "zai" | "z.ai" => Ok(Self::Zai),
            "opencode" => Ok(Self::Opencode),
            "azure-foundry" | "foundry" | "azure" => Ok(Self::AzureFoundry),
            "vertex" => Ok(Self::Vertex),
            other => anyhow::bail!("unknown provider '{other}'"),
        }
    }
}

/// Configuration for one LLM request: which endpoint to hit, which model to
/// ask for, and which key to authenticate with.
///
/// A request whose model matches one of the client's configured custom
/// endpoints uses that endpoint directly; everything else uses the server's
/// own LLM configuration.
#[derive(Debug, Clone)]
pub struct ResolvedLlm {
    pub base_url: String,
    pub api_key: Option<String>,
    pub schema: LlmSchema,
    pub model: String,
    /// Posts to this exact URL instead of `{base_url}/chat/completions` or
    /// `{base_url}/messages` (used by Azure Foundry's api-version query).
    pub endpoint: Option<String>,
    /// OAuth credentials resolved to a bearer token per request.
    pub dynamic_auth: Option<DynamicAuth>,
    /// ChatGPT account id, sent as a header to the Codex backend.
    pub account_id: Option<String>,
    /// Extra request headers required by specific providers (e.g. OpenCode's
    /// referer/title headers).
    pub headers: Vec<(String, String)>,
}

/// OAuth credential flows whose access tokens are fetched (and cached) per
/// request.
#[derive(Debug, Clone)]
pub enum DynamicAuth {
    /// Azure Entra ID client-credentials flow.
    AzureClientCredentials {
        token_url: String,
        client_id: String,
        client_secret: String,
        scope: String,
    },
    /// Google Vertex AI via gcloud application-default credentials.
    VertexAdc {
        token_url: String,
        client_id: String,
        client_secret: String,
        refresh_token: String,
    },
}

impl Config {
    /// Resolves the LLM target for a client request selecting `requested_model`.
    ///
    /// Priority: the client's configured custom endpoint for that model, then
    /// the server's explicit `--provider`, then (with `byok_direct`) the
    /// provider matching the client's BYOK API key, then the local backend.
    pub fn resolve_llm(
        &self,
        requested_model: &str,
        custom_providers: Option<&warp_multi_agent_api::request::settings::CustomModelProviders>,
        api_keys: Option<&warp_multi_agent_api::request::settings::ApiKeys>,
    ) -> ResolvedLlm {
        // The client's "auto" selection means "server, pick the default" —
        // resolve it to the default served model.
        let requested_model = if requested_model.eq_ignore_ascii_case("auto") {
            self.llm_models
                .first()
                .cloned()
                .or_else(|| self.llm_model.clone())
                .unwrap_or_else(|| requested_model.to_owned())
        } else {
            requested_model.to_owned()
        };

        if let Some(provider) = custom_providers.and_then(|providers| {
            providers.providers.iter().find(|provider| {
                provider
                    .models
                    .iter()
                    .any(|m| m.config_key == requested_model)
            })
        }) {
            let slug = provider
                .models
                .iter()
                .find(|m| m.config_key == requested_model)
                .map(|m| m.slug.clone())
                .unwrap_or_else(|| requested_model.to_owned());
            return ResolvedLlm {
                base_url: provider.base_url.clone(),
                api_key: non_empty(&provider.api_key),
                schema: CustomEndpointSchema::try_from(provider.schema)
                    .map(LlmSchema::from)
                    .unwrap_or(LlmSchema::Openai),
                model: slug,
                endpoint: None,
                dynamic_auth: None,
                account_id: None,
                headers: Vec::new(),
            };
        }

        if self.provider.is_some()
            && let Some(resolved) = self.explicit_provider_route(&requested_model)
        {
            return resolved;
        }

        if let Some(resolved) = self.native_key_route(&requested_model) {
            return resolved;
        }

        if self.byok_direct
            && let Some(resolved) = byok_provider_route(&requested_model, api_keys)
        {
            return resolved;
        }

        ResolvedLlm {
            base_url: self.llm_base_url.clone(),
            api_key: self.llm_api_key.clone(),
            schema: self.llm_schema,
            model: self
                .llm_model
                .clone()
                .unwrap_or_else(|| requested_model.to_owned()),
            endpoint: None,
            dynamic_auth: None,
            account_id: None,
            headers: Vec::new(),
        }
    }

    /// Routes to a cloud provider whose server-side key is configured and
    /// whose model names match the request — so `--zai-api-key`/`--opencode-api-key`
    /// (and the others) work natively, like BYOK routing does for client keys.
    ///
    /// Checked in declaration order; OpenCode's names come before the xAI
    /// prefix rule because `grok-code` is OpenCode's model.
    fn native_key_route(&self, requested_model: &str) -> Option<ResolvedLlm> {
        let model = requested_model.to_ascii_lowercase();
        let mk = |base_url: &str, key: &str, schema: LlmSchema, headers: Vec<(String, String)>| {
            Some(ResolvedLlm {
                base_url: base_url.to_owned(),
                api_key: non_empty(key.trim()),
                schema,
                model: requested_model.to_owned(),
                endpoint: None,
                dynamic_auth: None,
                account_id: None,
                headers,
            })
            .filter(|resolved| resolved.api_key.is_some())
        };

        if self
            .zai_api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty())
            && model.starts_with("glm")
        {
            return mk(
                "https://api.z.ai/api/coding/paas/v4",
                self.zai_api_key.as_deref().unwrap_or_default(),
                LlmSchema::Openai,
                Vec::new(),
            );
        }

        if self
            .opencode_api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty())
            && (model.starts_with("grok-code")
                || model.starts_with("code-supernova")
                || model.starts_with("kimi-k2")
                || model.starts_with("qwen3-coder"))
        {
            return mk(
                "https://opencode.ai/zen/v1",
                self.opencode_api_key.as_deref().unwrap_or_default(),
                LlmSchema::Openai,
                vec![
                    (
                        "https-referer".to_owned(),
                        "https://opencode.ai/".to_owned(),
                    ),
                    ("x-title".to_owned(), "opencode".to_owned()),
                ],
            );
        }

        if self
            .xai_api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty())
            && model.starts_with("grok")
        {
            return mk(
                "https://api.x.ai/v1",
                self.xai_api_key.as_deref().unwrap_or_default(),
                LlmSchema::Openai,
                Vec::new(),
            );
        }

        if self
            .anthropic_api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty())
            && model.starts_with("claude")
        {
            return mk(
                "https://api.anthropic.com/v1",
                self.anthropic_api_key.as_deref().unwrap_or_default(),
                LlmSchema::Anthropic,
                Vec::new(),
            );
        }

        if self
            .openai_api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty())
            && ["gpt", "o1", "o3", "o4", "chatgpt", "codex"]
                .iter()
                .any(|prefix| model.starts_with(prefix))
        {
            return mk(
                "https://api.openai.com/v1",
                self.openai_api_key.as_deref().unwrap_or_default(),
                LlmSchema::Openai,
                Vec::new(),
            );
        }

        if self
            .google_api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty())
            && model.starts_with("gemini")
        {
            return mk(
                "https://generativelanguage.googleapis.com/v1beta/openai",
                self.google_api_key.as_deref().unwrap_or_default(),
                LlmSchema::Openai,
                Vec::new(),
            );
        }

        None
    }

    /// Routes to the explicitly configured `--provider`, when its credentials
    /// are present.
    fn explicit_provider_route(&self, requested_model: &str) -> Option<ResolvedLlm> {
        let model = || requested_model.to_owned();
        // Static-key providers fall through to the next route when their key
        // is missing, so a half-configured provider never sends unauthenticated
        // requests.
        let static_provider = |base_url: &str, key: &Option<String>, schema: LlmSchema| {
            Some(ResolvedLlm {
                base_url: base_url.to_owned(),
                api_key: key.clone(),
                schema,
                model: requested_model.to_owned(),
                endpoint: None,
                dynamic_auth: None,
                account_id: None,
                headers: Vec::new(),
            })
            .filter(|_| key.as_ref().is_some_and(|key| !key.is_empty()))
        };
        match self.provider? {
            ProviderKind::Chatgpt => Some(ResolvedLlm {
                base_url: "https://chatgpt.com/backend-api/codex".to_owned(),
                api_key: None,
                schema: LlmSchema::Chatgpt,
                model: model(),
                endpoint: None,
                dynamic_auth: None,
                account_id: None,
                headers: Vec::new(),
            }),
            ProviderKind::Openai => static_provider(
                "https://api.openai.com/v1",
                &self.openai_api_key,
                LlmSchema::Openai,
            ),
            ProviderKind::Anthropic => static_provider(
                "https://api.anthropic.com/v1",
                &self.anthropic_api_key,
                LlmSchema::Anthropic,
            ),
            ProviderKind::Google => static_provider(
                "https://generativelanguage.googleapis.com/v1beta/openai",
                &self.google_api_key,
                LlmSchema::Openai,
            ),
            ProviderKind::OpenRouter => {
                static_provider("https://openrouter.ai/api/v1", &None, LlmSchema::Openai)
            }
            ProviderKind::Xai => {
                static_provider("https://api.x.ai/v1", &self.xai_api_key, LlmSchema::Openai)
            }
            ProviderKind::Zai => static_provider(
                "https://api.z.ai/api/coding/paas/v4",
                &self.zai_api_key,
                LlmSchema::Openai,
            ),
            ProviderKind::Opencode => static_provider(
                "https://opencode.ai/zen/v1",
                &self.opencode_api_key,
                LlmSchema::Openai,
            ),
            ProviderKind::AzureFoundry => {
                let foundry_url = self
                    .azure_foundry_url
                    .as_ref()
                    .filter(|url| !url.is_empty())?;
                let tenant = self.azure_tenant.as_ref().filter(|t| !t.is_empty())?;
                let client_id = self.azure_client_id.as_ref().filter(|id| !id.is_empty())?;
                let client_secret = self
                    .azure_client_secret
                    .as_ref()
                    .filter(|secret| !secret.is_empty())?;
                Some(ResolvedLlm {
                    base_url: format!(
                        "{}/chat/completions?api-version={}",
                        foundry_url.trim_end_matches('/'),
                        self.azure_api_version
                    ),
                    api_key: None,
                    schema: LlmSchema::Openai,
                    model: model(),
                    endpoint: None,
                    account_id: None,
                    dynamic_auth: Some(DynamicAuth::AzureClientCredentials {
                        token_url: format!(
                            "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
                        ),
                        client_id: client_id.clone(),
                        client_secret: client_secret.clone(),
                        scope: "https://cognitiveservices.azure.com/.default".to_owned(),
                    }),
                    headers: Vec::new(),
                })
            }
            ProviderKind::Vertex => {
                if !self.vertex_enabled {
                    return None;
                }
                let adc = self
                    .vertex_adc_path
                    .clone()
                    .unwrap_or_else(default_adc_path);
                let credentials = std::fs::read_to_string(&adc).ok()?;
                let credentials: Value = serde_json::from_str(&credentials).ok()?;
                Some(ResolvedLlm {
                    base_url: self.vertex_base_url.clone(),
                    api_key: None,
                    schema: LlmSchema::Openai,
                    model: model(),
                    endpoint: None,
                    account_id: None,
                    dynamic_auth: Some(DynamicAuth::VertexAdc {
                        token_url: "https://oauth2.googleapis.com/token".to_owned(),
                        client_id: credentials["client_id"].as_str()?.to_owned(),
                        client_secret: credentials["client_secret"].as_str()?.to_owned(),
                        refresh_token: credentials["refresh_token"].as_str()?.to_owned(),
                    }),
                    headers: Vec::new(),
                })
            }
        }
    }
}

/// Default location of gcloud's application-default credentials.
pub fn default_adc_path() -> String {
    if let Ok(appdata) = std::env::var("APPDATA") {
        return format!("{appdata}\\gcloud\\application_default_credentials.json");
    }
    format!(
        "{}/.config/gcloud/application_default_credentials.json",
        std::env::var("HOME").unwrap_or_default()
    )
}

/// Routes a request directly to the provider the client supplied a BYOK key
/// for, matched by model-name prefix. Returns `None` when no key or provider
/// matches.
fn byok_provider_route(
    requested_model: &str,
    api_keys: Option<&warp_multi_agent_api::request::settings::ApiKeys>,
) -> Option<ResolvedLlm> {
    let keys = api_keys?;
    let model = requested_model.to_ascii_lowercase();
    let route = |base_url: &str, key: &str, schema: LlmSchema| -> Option<ResolvedLlm> {
        let key = non_empty(key.trim())?;
        Some(ResolvedLlm {
            base_url: base_url.to_owned(),
            api_key: Some(key),
            schema,
            model: requested_model.to_owned(),
            endpoint: None,
            dynamic_auth: None,
            account_id: None,
            headers: Vec::new(),
        })
    };

    // The client exchanges a Grok subscription OAuth token itself and sends
    // it alongside BYOK keys; route it to xAI's API.
    if let Some(token) = non_empty(&keys.grok_oauth_access_token)
        && model.starts_with("grok")
    {
        return Some(ResolvedLlm {
            base_url: "https://api.x.ai/v1".to_owned(),
            api_key: Some(token),
            schema: LlmSchema::Openai,
            model: requested_model.to_owned(),
            endpoint: None,
            dynamic_auth: None,
            account_id: None,
            headers: Vec::new(),
        });
    }

    if model.starts_with("claude") {
        route(
            "https://api.anthropic.com/v1",
            &keys.anthropic,
            LlmSchema::Anthropic,
        )
    } else if model.starts_with("gemini") {
        route(
            "https://generativelanguage.googleapis.com/v1beta/openai",
            &keys.google,
            LlmSchema::Openai,
        )
    } else if model.contains('/') && non_empty(&keys.open_router).is_some() {
        route(
            "https://openrouter.ai/api/v1",
            &keys.open_router,
            LlmSchema::Openai,
        )
    } else if ["gpt", "o1", "o3", "o4", "chatgpt", "codex"]
        .iter()
        .any(|prefix| model.starts_with(prefix))
    {
        route("https://api.openai.com/v1", &keys.openai, LlmSchema::Openai)
    } else {
        None
    }
}

fn non_empty(s: &str) -> Option<String> {
    (!s.is_empty()).then(|| s.to_owned())
}

impl From<warp_multi_agent_api::request::settings::custom_model_providers::CustomEndpointSchema>
    for LlmSchema
{
    fn from(
        schema: warp_multi_agent_api::request::settings::custom_model_providers::CustomEndpointSchema,
    ) -> Self {
        match schema {
            CustomEndpointSchema::AnthropicMessages => Self::Anthropic,
            CustomEndpointSchema::OpenaiChatCompletions | CustomEndpointSchema::OpenaiResponses => {
                Self::Openai
            }
        }
    }
}

#[cfg(test)]
impl Config {
    /// A minimal disabled-config base for tests; tests override the fields
    /// they exercise.
    pub(crate) fn test_default() -> Self {
        Self {
            llm_base_url: String::new(),
            llm_api_key: None,
            llm_schema: LlmSchema::Openai,
            llm_model: None,
            api_key: None,
            system_prompt: None,
            transcribe_base_url: None,
            transcribe_api_key: None,
            transcribe_model: "whisper-1".to_owned(),
            llm_models: Vec::new(),
            context_window_tokens: 0,
            web_search_url: None,
            byok_direct: false,
            auth_introspect_url: None,
            provider: None,
            openai_api_key: None,
            anthropic_api_key: None,
            google_api_key: None,
            xai_api_key: None,
            zai_api_key: None,
            opencode_api_key: None,
            azure_tenant: None,
            azure_client_id: None,
            azure_client_secret: None,
            azure_foundry_url: None,
            azure_api_version: "2024-05-01-preview".to_owned(),
            vertex_enabled: false,
            vertex_base_url: String::new(),
            vertex_adc_path: None,
            codex_token_file: String::new(),
        }
    }
}

#[cfg(test)]
impl ResolvedLlm {
    pub(crate) fn test_default(base_url: String) -> Self {
        Self {
            base_url,
            api_key: None,
            schema: LlmSchema::Openai,
            model: "test-model".to_owned(),
            endpoint: None,
            dynamic_auth: None,
            account_id: None,
            headers: Vec::new(),
        }
    }
}
