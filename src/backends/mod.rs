//! Inference engine backends for ollama-classifier-rs.
//!
//! Each backend implements the [`LLMBackend`] trait and communicates with
//! its respective inference engine via HTTP (OpenAI-compatible API).
//!
//! # Available Backends
//!
//! - [`VLLMBackend`] — vLLM inference server
//! - [`SGLangBackend`] — SGLang inference server
//! - [`LlamaCppBackend`] — llama.cpp server (llama-server)

pub mod base;
pub mod llamacpp;
pub mod sglang;
pub mod vllm;

pub use base::{ChatMessage, ChatResponse, LLMBackend};
pub use llamacpp::LlamaCppBackend;
pub use sglang::SGLangBackend;
pub use vllm::VLLMBackend;
