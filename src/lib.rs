//! # ollama-classifier-rs
//!
//! A Rust port of the Python `ollama-classifier` library — a backend-agnostic
//! text classifier that delegates inference to any LLM server via the
//! OpenAI-compatible chat completions API.
//!
//! ## Supported Backends
//!
//! - **vLLM** — high-throughput serving engine
//! - **SGLang** — fast structured generation serving
//! - **llama.cpp** — lightweight local inference via `llama-server`
//!
//! ## Quick Start
//!
//! ```no_run
//! use ollama_classifier_rs::backends::VLLMBackend;
//! use ollama_classifier_rs::LLMClassifier;
//!
//! let backend = VLLMBackend::new("meta-llama/Llama-3.2-3B-Instruct");
//! let classifier = LLMClassifier::new(backend);
//!
//! let result = classifier.classify(
//!     "I love this product!",
//!     vec!["positive", "negative", "neutral"],
//!     None,
//! ).unwrap();
//!
//! println!("Prediction: {}", result.prediction);
//! println!("Confidence: {:.2}%", result.confidence * 100.0);
//! ```
//!
//! ## Classification Methods
//!
//! - **`generate`** — Fast single-call classification using JSON schema constraints.
//!   Returns only the predicted label, no confidence scores.
//! - **`classify`** — Multi-call evaluation with softmax-calibrated
//!   probabilities. Makes N API calls for N choices and provides confidence scores.
//! - **`batch_*`** — Process multiple texts sequentially.
//! - **`a*`** — Async versions of all methods (requires `tokio` runtime).

pub mod backends;
pub mod classifier;
pub mod prompts;
pub mod types;

pub use classifier::LLMClassifier;
pub use types::{Choices, ClassificationResult};

// Re-export anyhow::Result as the crate-level error type for convenience.
pub use anyhow::Result;
