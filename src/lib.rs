//! # ollama-classifier-rs
//!
//! A Rust port of the Python [`ollama-classifier`](https://github.com/paluigi/ollama-classifier)
//! library — a backend-agnostic text classifier that delegates inference to any
//! LLM server and produces calibrated confidence scores.
//!
//! ## Supported Backends
//!
//! - **Ollama** — native Ollama API (`/api/chat`), forced constrained generation
//! - **vLLM** — OpenAI-compatible, `structured_outputs.choice` constraints
//! - **SGLang** — OpenAI-compatible, `regex` constraints
//! - **llama.cpp** — OpenAI-compatible, GBNF `grammar` constraints
//!
//! ## Classification Methods
//!
//! - **[`classify`][crate::LLMClassifier::classify]** — multi-call completion
//!   scoring. Makes one backend call per label and normalizes the
//!   geometric-mean logprobs with a stable softmax. Exact, N calls for N labels.
//! - **[`generate`][crate::LLMClassifier::generate]** — hierarchical
//!   constrained-generation scoring. The first call constrains the model to all
//!   labels and produces an internally consistent probability distribution.
//!   Supplementary calls (when `max_calls > 1`) resolve label clusters by
//!   reproportioning probability mass *within* a cluster — never changing
//!   between-group totals, so accuracy never degrades as the call budget grows.
//!   `max_calls` controls the budget (`1` = single call, `None` = resolve all
//!   clusters).
//!
//! Sync, async (`a*`), and batch (`batch_*` / `abatch_*`) variants are provided.

pub mod backends;
pub mod classifier;
pub mod prompts;
pub mod scoring;
pub mod types;

pub use classifier::LLMClassifier;
pub use types::{Choices, ChoicesType, ClassificationResult};

// Re-export anyhow::Result as the crate-level error type for convenience.
pub use anyhow::Result;
