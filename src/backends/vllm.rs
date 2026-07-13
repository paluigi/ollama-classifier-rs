//! vLLM inference backend.
//!
//! vLLM exposes an OpenAI-compatible API and supports
//! `structured_outputs.choice` (vLLM v0.12.0+) for constraining output to a
//! label set, generating bare label text with no JSON wrapper.
//!
//! `score()` uses echo/prefill (`/v1/completions` with `echo=true`) to recover
//! genuine per-label logprobs. `tokenize()` uses forced constrained generation
//! so token boundaries match the actual constrained-generation output.
//!
//! # Example
//!
//! ```no_run
//! use ollama_classifier_rs::backends::VLLMBackend;
//! use ollama_classifier_rs::LLMClassifier;
//!
//! let backend = VLLMBackend::new("meta-llama/Llama-3.2-3B-Instruct");
//! let classifier = LLMClassifier::new(backend);
//! ```

use std::collections::HashMap;
use std::time::Duration;

use serde_json::Value;

use super::openai_compat::{BoundaryStrategy, Constraint, OpenAICompatCoreBuilder};

/// Default vLLM base URL.
pub const DEFAULT_BASE_URL: &str = "http://localhost:8000/v1";

/// Backend for a vLLM inference server.
pub struct VLLMBackend {
    pub(crate) core: super::openai_compat::OpenAICompatCore,
}

crate::impl_openai_compat_backend!(VLLMBackend);

impl VLLMBackend {
    /// Create a new vLLM backend pointed at the default URL.
    pub fn new(model: impl Into<String>) -> Self {
        Self::with_config(model, DEFAULT_BASE_URL)
    }

    /// Create a vLLM backend with a custom base URL.
    pub fn with_config(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::builder(model, base_url).build()
    }

    /// Create a builder for fine-grained configuration.
    pub fn builder(model: impl Into<String>, base_url: impl Into<String>) -> VLLMBackendBuilder {
        VLLMBackendBuilder {
            inner: OpenAICompatCoreBuilder {
                model: model.into(),
                base_url: base_url.into(),
                api_key: "not-needed".into(),
                timeout: Duration::from_secs(120),
                max_tokens: 256,
                extra_body: HashMap::new(),
                constraint: Constraint::StructuredOutputsChoice,
                boundary: BoundaryStrategy::Ids,
            },
        }
    }
}

/// Builder for [`VLLMBackend`].
pub struct VLLMBackendBuilder {
    inner: OpenAICompatCoreBuilder,
}

impl VLLMBackendBuilder {
    /// Set the API key. Defaults to `"not-needed"`.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.inner.api_key = key.into();
        self
    }

    /// Set the request timeout. Defaults to 120s.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.inner.timeout = timeout;
        self
    }

    /// Set the maximum tokens to generate. Defaults to 256.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.inner.max_tokens = max_tokens;
        self
    }

    /// Add an extra parameter merged into every request body.
    pub fn extra_body(mut self, key: impl Into<String>, value: Value) -> Self {
        self.inner.extra_body.insert(key.into(), value);
        self
    }

    /// Build the [`VLLMBackend`].
    pub fn build(self) -> VLLMBackend {
        VLLMBackend {
            core: self.inner.build(),
        }
    }
}
