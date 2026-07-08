//! # ollama-classifier-rs
//!
//! A Rust port of the Python [`ollama-classifier`](https://github.com/paluigi/ollama-classifier)
//! library — a backend-agnostic text classifier that delegates inference to any
//! LLM server and produces calibrated confidence scores.
//!
//! ## Supported Backends
//!
//! - **Ollama** — native Ollama API (`/api/chat`, `/api/generate`, `/api/tokenize`)
//! - **vLLM** — OpenAI-compatible, `guided_choice` constraints
//! - **SGLang** — OpenAI-compatible, `regex` constraints
//! - **llama.cpp** — OpenAI-compatible, GBNF `grammar` constraints
//!
//! ## Classification Methods
//!
//! - **[`classify`][crate::LLMClassifier::classify]** — multi-call completion
//!   scoring. Makes one backend call per label and normalizes the
//!   geometric-mean logprobs with a stable softmax. Exact, N calls for N labels.
//! - **[`generate`][crate::LLMClassifier::generate]** — adaptive
//!   constrained-generation scoring. Tokenizes labels, builds a trie, and
//!   resolves ambiguity with a bounded number of constrained calls controlled
//!   by `max_calls` (`1` = single fast approximate call, `None` = fully exact).
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
