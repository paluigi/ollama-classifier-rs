//! llama.cpp inference backend.
//!
//! Talks to a `llama-server` instance via its OpenAI-compatible API. llama.cpp
//! supports GBNF `grammar` constraints for restricting output to a label set.
//! The server's JSON-schema `response_format` is deliberately avoided (it is
//! unreliable on current versions), so the bare-label grammar is used instead.
//!
//! # Example
//!
//! ```no_run
//! use ollama_classifier_rs::backends::LlamaCppBackend;
//! use ollama_classifier_rs::LLMClassifier;
//!
//! let backend = LlamaCppBackend::new("model.gguf");
//! let classifier = LLMClassifier::new(backend);
//! ```

use std::collections::HashMap;
use std::time::Duration;

use serde_json::Value;

use super::openai_compat::{BoundaryStrategy, Constraint, OpenAICompatCoreBuilder};

/// Default llama-server base URL.
pub const DEFAULT_BASE_URL: &str = "http://localhost:8080/v1";

/// Backend for a `llama-server` instance.
pub struct LlamaCppBackend {
    pub(crate) core: super::openai_compat::OpenAICompatCore,
}

crate::impl_openai_compat_backend!(LlamaCppBackend);

impl LlamaCppBackend {
    /// Create a new llama.cpp backend pointed at the default URL.
    pub fn new(model: impl Into<String>) -> Self {
        Self::with_config(model, DEFAULT_BASE_URL)
    }

    /// Create a llama.cpp backend with a custom base URL.
    pub fn with_config(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::builder(model, base_url).build()
    }

    /// Create a builder for fine-grained configuration.
    pub fn builder(
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> LlamaCppBackendBuilder {
        LlamaCppBackendBuilder {
            inner: OpenAICompatCoreBuilder {
                model: model.into(),
                base_url: base_url.into(),
                api_key: "not-needed".into(),
                timeout: Duration::from_secs(120),
                max_tokens: 256,
                extra_body: HashMap::new(),
                constraint: Constraint::Grammar,
                boundary: BoundaryStrategy::FillMiddle,
            },
        }
    }
}

/// Builder for [`LlamaCppBackend`].
pub struct LlamaCppBackendBuilder {
    inner: OpenAICompatCoreBuilder,
}

impl LlamaCppBackendBuilder {
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

    /// Build the [`LlamaCppBackend`].
    pub fn build(self) -> LlamaCppBackend {
        LlamaCppBackend {
            core: self.inner.build(),
        }
    }
}
