//! Prometheus-format counters for the self-hosted server.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Counters shared through [`crate::multi_agent::AppState`].
#[derive(Debug, Default)]
pub struct Metrics {
    pub graphql_requests: AtomicU64,
    pub agent_requests: AtomicU64,
    pub agent_llm_errors: AtomicU64,
    pub agent_tool_calls: AtomicU64,
    pub transcribe_requests: AtomicU64,
    pub relevant_file_requests: AtomicU64,
    pub mcp_requests: AtomicU64,
    pub llm_input_tokens: AtomicU64,
    pub llm_output_tokens: AtomicU64,
}

impl Metrics {
    fn render(&self) -> String {
        let mut out = String::new();
        let mut counter = |name: &str, help: &str, value: u64| {
            out.push_str(&format!("# HELP {name} {help}\n"));
            out.push_str(&format!("# TYPE {name} counter\n{name} {value}\n"));
        };
        counter(
            "selfhost_graphql_requests_total",
            "GraphQL requests received.",
            self.graphql_requests.load(Ordering::Relaxed),
        );
        counter(
            "selfhost_agent_requests_total",
            "Multi-agent requests received.",
            self.agent_requests.load(Ordering::Relaxed),
        );
        counter(
            "selfhost_agent_llm_errors_total",
            "LLM completions that ended in an error.",
            self.agent_llm_errors.load(Ordering::Relaxed),
        );
        counter(
            "selfhost_agent_tool_calls_total",
            "Tool calls emitted to clients.",
            self.agent_tool_calls.load(Ordering::Relaxed),
        );
        counter(
            "selfhost_transcribe_requests_total",
            "Voice transcription requests.",
            self.transcribe_requests.load(Ordering::Relaxed),
        );
        counter(
            "selfhost_relevant_file_requests_total",
            "Codebase relevance requests.",
            self.relevant_file_requests.load(Ordering::Relaxed),
        );
        counter(
            "selfhost_mcp_requests_total",
            "Factory MCP JSON-RPC requests.",
            self.mcp_requests.load(Ordering::Relaxed),
        );
        counter(
            "selfhost_llm_input_tokens_total",
            "Input tokens reported by LLM backends.",
            self.llm_input_tokens.load(Ordering::Relaxed),
        );
        counter(
            "selfhost_llm_output_tokens_total",
            "Output tokens reported by LLM backends.",
            self.llm_output_tokens.load(Ordering::Relaxed),
        );
        out
    }
}

/// Handle for incrementing metrics; cheap to clone.
#[derive(Clone)]
pub struct MetricsHandle {
    inner: Arc<Metrics>,
}

impl MetricsHandle {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Metrics::default()),
        }
    }

    pub fn shared(&self) -> Arc<Metrics> {
        self.inner.clone()
    }

    pub fn agent_request(&self) {
        self.inner.agent_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn agent_llm_error(&self) {
        self.inner.agent_llm_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn agent_tool_calls(&self, count: usize) {
        self.inner
            .agent_tool_calls
            .fetch_add(count as u64, Ordering::Relaxed);
    }

    pub fn usage(&self, input: u64, output: u64) {
        self.inner
            .llm_input_tokens
            .fetch_add(input, Ordering::Relaxed);
        self.inner
            .llm_output_tokens
            .fetch_add(output, Ordering::Relaxed);
    }

    pub fn relevant_file_request(&self) {
        self.inner
            .relevant_file_requests
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn mcp_request(&self) {
        self.inner.mcp_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn transcribe_request(&self) {
        self.inner
            .transcribe_requests
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn render(&self) -> String {
        self.inner.render()
    }
}

impl Default for MetricsHandle {
    fn default() -> Self {
        Self::new()
    }
}
