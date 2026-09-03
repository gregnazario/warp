use std::net::SocketAddr;

use anyhow::{Context as _, Result};
use clap::Parser;
use selfhost_server::config::{LlmBackend, ProviderKind};
use selfhost_server::{Config, detect};

/// Self-hosted Warp agent backend.
///
/// Serves the subset of Warp's server API that Agent Mode needs, translating
/// between Warp's multi-agent wire protocol and an OpenAI- or
/// Anthropic-compatible LLM endpoint. Point a Warp client at it with
/// `WARP_SELF_HOSTED_SERVER_URL` and agent traffic never reaches
/// Warp-operated servers.
#[derive(Debug, Parser)]
#[command(name = "selfhost-server")]
struct Args {
    /// Address to bind the server to.
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,

    /// Base URL of the OpenAI-compatible LLM endpoint. The server appends
    /// `/chat/completions` (OpenAI schema) or `/messages` (Anthropic schema).
    /// When unset, the backend is autodetected (Ollama, LM Studio, MLX-LM).
    #[arg(long)]
    llm_base_url: Option<String>,

    /// Which local backend to autodetect: `auto`, `ollama`, `lmstudio`, or
    /// `mlx`. Only used when `--llm-base-url` is unset.
    #[arg(long, default_value = "auto")]
    llm_backend: String,

    /// API key sent to the LLM endpoint, if it requires one.
    #[arg(long, env = "SELFHOST_LLM_API_KEY")]
    llm_api_key: Option<String>,

    /// LLM schema spoken by the endpoint: `openai` or `anthropic`.
    #[arg(long, default_value = "openai")]
    llm_schema: String,

    /// Model name to request from the LLM endpoint. When unset, the model ID
    /// selected in the Warp client is passed through unchanged.
    #[arg(long)]
    llm_model: Option<String>,

    /// If set, Warp clients must present this value as their bearer token
    /// (`WARP_API_KEY`).
    #[arg(long, env = "SELFHOST_API_KEY")]
    api_key: Option<String>,

    /// Path to a file whose contents replace the default agent system prompt.
    #[arg(long)]
    system_prompt_file: Option<String>,

    /// Base URL of an OpenAI-compatible speech-to-text endpoint;
    /// `/audio/transcriptions` is appended. Voice transcription is disabled
    /// when unset.
    #[arg(long, env = "SELFHOST_TRANSCRIBE_BASE_URL")]
    transcribe_base_url: Option<String>,

    /// API key sent to the speech-to-text endpoint, if it requires one.
    #[arg(long, env = "SELFHOST_TRANSCRIBE_API_KEY")]
    transcribe_api_key: Option<String>,

    /// Speech-to-text model to request (e.g. `whisper-1`, `base.en`).
    #[arg(long, default_value = "whisper-1")]
    transcribe_model: String,

    /// Comma-separated model catalog exposed to the Warp client's model picker
    /// (e.g. `--llm-models qwen3-coder:30b,qwen3:8b`). Selected IDs are passed
    /// through to the LLM endpoint.
    #[arg(long)]
    llm_models: Option<String>,

    /// Context-window budget in estimated tokens at which older history is
    /// compacted into a summary. 0 (the default) auto-detects: the client's
    /// per-model limit when set, else 96k.
    #[arg(long, default_value_t = 0)]
    context_window_tokens: u64,

    /// SearXNG-compatible search endpoint enabling the server-side web-search
    /// tool (e.g. http://127.0.0.1:8888/search).
    #[arg(long, env = "SELFHOST_WEB_SEARCH_URL")]
    web_search_url: Option<String>,

    /// Run the ChatGPT (Codex) OAuth login flow now, save the tokens, and
    /// exit. Required once before `--provider chatgpt`.
    #[arg(long, default_value_t = false)]
    codex_login: bool,
    /// Where Codex OAuth tokens are cached (default
    /// ~/.cache/selfhost-server/codex-tokens.json).
    #[arg(long)]
    codex_token_file: Option<String>,

    /// Route requests carrying the client's BYOK API keys directly to the
    /// provider named by the key (Anthropic/OpenAI/Google/OpenRouter) based on
    /// the requested model's name.
    #[arg(long, default_value_t = false)]
    byok_direct: bool,

    /// Validate every client bearer token by POSTing it as JSON
    /// `{"token": ...}` to this URL and requiring a 2xx response. Lets an
    /// external auth service (SSO, an auth gateway, Foundry-style brokers)
    /// own authentication instead of `--api-key`.
    #[arg(long, env = "SELFHOST_AUTH_INTROSPECT_URL")]
    auth_introspect_url: Option<String>,

    /// Force all non-custom-endpoint traffic to this provider: openai,
    /// anthropic, google, openrouter, xai, zai, opencode, azure-foundry, or
    /// vertex.
    #[arg(long)]
    provider: Option<String>,

    /// API key for OpenAI routing (also `--provider openai`).
    #[arg(long, env = "SELFHOST_OPENAI_API_KEY")]
    openai_api_key: Option<String>,
    #[arg(long, env = "SELFHOST_ANTHROPIC_API_KEY")]
    anthropic_api_key: Option<String>,
    #[arg(long, env = "SELFHOST_GOOGLE_API_KEY")]
    google_api_key: Option<String>,
    /// API key for xAI routing (`--provider xai`, or grok* models).
    #[arg(long, env = "SELFHOST_XAI_API_KEY")]
    xai_api_key: Option<String>,
    /// Z.ai coding-plan API key (https://api.z.ai/api/coding/paas/v4).
    #[arg(long, env = "SELFHOST_ZAI_API_KEY")]
    zai_api_key: Option<String>,
    /// OpenCode Zen API key (https://opencode.ai/zen/v1).
    #[arg(long, env = "SELFHOST_OPENCODE_API_KEY")]
    opencode_api_key: Option<String>,

    /// Azure tenant for Azure AI Foundry Entra-ID authentication.
    #[arg(long)]
    azure_tenant: Option<String>,
    #[arg(long)]
    azure_client_id: Option<String>,
    #[arg(long)]
    azure_client_secret: Option<String>,
    /// Foundry endpoint prefix, e.g.
    /// `https://<resource>.services.ai.azure.com/models`.
    #[arg(long)]
    azure_foundry_url: Option<String>,
    #[arg(long, default_value = "2024-05-01-preview")]
    azure_api_version: String,
    /// Enable Google Vertex AI routing via gcloud application-default
    /// credentials (~/.config/gcloud/application_default_credentials.json).
    #[arg(long, default_value_t = false)]
    vertex: bool,
    #[arg(
        long,
        default_value = "https://aiplatform.googleapis.com/v1beta1/publishers/google/models"
    )]
    vertex_base_url: String,
    #[arg(long)]
    vertex_adc_path: Option<String>,
}

impl Args {
    async fn into_config(mut self) -> Result<Config> {
        let system_prompt = match &self.system_prompt_file {
            Some(path) => Some(
                std::fs::read_to_string(path)
                    .with_context(|| format!("failed to read system prompt file {path}"))?,
            ),
            None => None,
        };

        // ChatGPT provider traffic never touches a local backend.
        let chatgpt_provider = self
            .provider
            .as_deref()
            .is_some_and(|name| matches!(name.parse::<ProviderKind>(), Ok(ProviderKind::Chatgpt)));

        // Local backend: an explicit base URL wins; otherwise autodetect
        // among Ollama / LM Studio / MLX-LM (or probe only the requested one).
        let mut detected_models = Vec::new();
        if chatgpt_provider && self.llm_base_url.is_none() {
            self.llm_base_url = Some("https://chatgpt.com/backend-api/codex".to_owned());
        }
        if self.llm_base_url.is_none() {
            let backend: LlmBackend = self.llm_backend.parse()?;
            let http = reqwest::Client::new();
            let targets = match backend {
                LlmBackend::Auto => detect::default_targets(),
                LlmBackend::Ollama => detect::default_targets()
                    .into_iter()
                    .filter(|target| target.backend == LlmBackend::Ollama)
                    .collect(),
                LlmBackend::LmStudio => detect::default_targets()
                    .into_iter()
                    .filter(|target| target.backend == LlmBackend::LmStudio)
                    .collect(),
                LlmBackend::Mlx => detect::default_targets()
                    .into_iter()
                    .filter(|target| target.backend == LlmBackend::Mlx)
                    .collect(),
            };
            match detect::detect(&http, &targets).await {
                Some(detected) => {
                    tracing::info!(
                        "autodetected {} at {} with {} model(s)",
                        detected.backend,
                        detected.base_url,
                        detected.models.len()
                    );
                    self.llm_base_url =
                        Some(format!("{}/v1", detected.base_url.trim_end_matches('/')));
                    detected_models = detected.models;
                }
                None => {
                    tracing::warn!(
                        "no local LLM backend detected (tried {}); falling back to the Ollama default",
                        targets
                            .iter()
                            .map(|target| target.base_url.to_owned())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                    self.llm_base_url = Some("http://127.0.0.1:11434/v1".to_owned());
                }
            }
        }

        let provider = match &self.provider {
            Some(name) => Some(name.parse::<ProviderKind>()?),
            None => None,
        };

        let codex_token_file = self
            .codex_token_file
            .unwrap_or_else(selfhost_server::codex::default_token_file);
        if self.codex_login {
            let http = reqwest::Client::new();
            let tokens = selfhost_server::codex::login(&http, &codex_token_file, |url| {
                println!("Opening browser (if it does not open, visit the URL)...");
                let opener = if cfg!(target_os = "macos") {
                    "open"
                } else if cfg!(target_os = "windows") {
                    "cmd"
                } else {
                    "xdg-open"
                };
                let mut command = command::blocking::Command::new(opener);
                if cfg!(target_os = "windows") {
                    command.arg("/c").arg("start");
                }
                command.arg(url).spawn().ok();
            })
            .await?;
            println!(
                "Codex tokens saved to {codex_token_file} (account: {}).",
                tokens.account_id.as_deref().unwrap_or("unknown")
            );
            std::process::exit(0);
        }
        let mut llm_models = match self.llm_models {
            Some(models) => models
                .split(',')
                .map(str::trim)
                .filter(|model| !model.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
            None => std::mem::take(&mut detected_models),
        };
        if provider == Some(ProviderKind::Chatgpt) && llm_models.is_empty() {
            llm_models = vec!["gpt-5-codex".to_owned(), "gpt-5".to_owned()];
        }
        // Native key routing: when a provider key is present, its models
        // belong in the picker so they can be selected.
        if self.zai_api_key.is_some() {
            for model in ["glm-4.7", "glm-4.6", "glm-4.5-air"] {
                if !llm_models.iter().any(|existing| existing == model) {
                    llm_models.push(model.to_owned());
                }
            }
        }
        if self.opencode_api_key.is_some() {
            for model in ["grok-code", "code-supernova", "kimi-k2", "qwen3-coder"] {
                if !llm_models.iter().any(|existing| existing == model) {
                    llm_models.push(model.to_owned());
                }
            }
        }

        Ok(Config {
            llm_base_url: self.llm_base_url.expect("set above"),
            llm_api_key: self.llm_api_key,
            llm_schema: self.llm_schema.parse()?,
            llm_models,
            llm_model: self.llm_model,
            api_key: self.api_key,
            system_prompt,
            transcribe_base_url: self.transcribe_base_url.filter(|url| !url.is_empty()),
            transcribe_api_key: self.transcribe_api_key,
            transcribe_model: self.transcribe_model,

            context_window_tokens: self.context_window_tokens,
            web_search_url: self.web_search_url.filter(|url| !url.is_empty()),
            provider,
            openai_api_key: self.openai_api_key.filter(|key| !key.is_empty()),
            anthropic_api_key: self.anthropic_api_key.filter(|key| !key.is_empty()),
            google_api_key: self.google_api_key.filter(|key| !key.is_empty()),
            xai_api_key: self.xai_api_key.filter(|key| !key.is_empty()),
            zai_api_key: self.zai_api_key.filter(|key| !key.is_empty()),
            opencode_api_key: self.opencode_api_key.filter(|key| !key.is_empty()),
            azure_tenant: self.azure_tenant,
            azure_client_id: self.azure_client_id,
            azure_client_secret: self.azure_client_secret,
            azure_foundry_url: self.azure_foundry_url.filter(|url| !url.is_empty()),
            azure_api_version: self.azure_api_version,
            vertex_enabled: self.vertex,
            vertex_base_url: self.vertex_base_url,
            vertex_adc_path: self.vertex_adc_path,
            codex_token_file,
            byok_direct: self.byok_direct,
            auth_introspect_url: self.auth_introspect_url.filter(|url| !url.is_empty()),
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "selfhost_server=info".into()),
        )
        .init();

    let args = Args::parse();
    let bind_addr: SocketAddr = args
        .bind
        .parse()
        .context("failed to parse --bind address")?;
    let config = args.into_config().await?;
    let app = selfhost_server::router(config);

    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .with_context(|| format!("failed to bind {bind_addr}"))?;
    tracing::info!("self-hosted agent server listening on http://{bind_addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
