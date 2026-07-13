//! Inference engine backends for ollama-classifier-rs.
//!
//! Each backend implements the [`LLMBackend`] trait. The OpenAI-compatible
//! backends (vLLM, SGLang, llama.cpp) share a common HTTP core; Ollama speaks
//! its own native protocol.
//!
//! All backends use empirical forced constrained generation for tokenization
//! and echo/prefill or forced generation for completion scoring.
//!
//! # Available Backends
//!
//! - [`OllamaBackend`] — native Ollama API
//! - [`VLLMBackend`] — vLLM (OpenAI-compatible, `structured_outputs.choice`)
//! - [`SGLangBackend`] — SGLang (OpenAI-compatible, `regex`)
//! - [`LlamaCppBackend`] — llama.cpp / `llama-server` (OpenAI-compatible, GBNF `grammar`)

pub mod base;
pub mod llamacpp;
pub mod ollama;
pub mod openai_compat;
pub mod sglang;
pub mod vllm;

pub use base::{ChatMessage, ChatResponse, LLMBackend, ScoringResponse, Token, TokenLogprob};
pub use llamacpp::LlamaCppBackend;
pub use ollama::OllamaBackend;
pub use sglang::SGLangBackend;
pub use vllm::VLLMBackend;
