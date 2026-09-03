//! Self-hosted Warp agent backend.
//!
//! Serves the subset of Warp's server API that Agent Mode needs
//! (`/ai/multi-agent`, `GetUser` GraphQL, voice transcription, relevant-file
//! search) and translates between Warp's wire protocols and OpenAI- or
//! Anthropic-compatible LLM endpoints. The `selfhost-server` binary is a thin
//! wrapper over this library.
pub mod codex;
pub mod config;
pub mod detect;
pub mod graphql;
pub mod metrics;
pub mod multi_agent;
pub mod oauth;
pub mod relevant_files;
pub mod transcribe;
pub mod websearch;

pub use config::Config;
pub use multi_agent::router;
