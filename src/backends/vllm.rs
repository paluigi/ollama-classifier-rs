//! vLLM inference backend.
//!
//! Supports both local and remote vLLM servers via the OpenAI-compatible API.
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

use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;

use super::base::{
    build_body, build_headers, parse_response, ChatMessage, ChatResponse, LLMBackend,
};

/// Backend for vLLM inference server.
///
/// vLLM provides a high-throughput serving engine with an OpenAI-compatible
/// API endpoint. It supports guided decoding (JSON schema constraints) and
/// logprob return out of the box.
pub struct VLLMBackend {
    model: String,
    base_url: String,
    api_key: String,
    max_tokens: u32,
    extra_body: HashMap<String, Value>,
    client: reqwest::blocking::Client,
    async_client: reqwest::Client,
}

impl VLLMBackend {
    /// Create a new vLLM backend.
    ///
    /// # Arguments
    /// * `model` — Model identifier (must match the model loaded on the server).
    pub fn new(model: impl Into<String>) -> Self {
        Self::with_config(model, "http://localhost:8000/v1")
    }

    /// Create a vLLM backend with custom base URL.
    pub fn with_config(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::builder(model, base_url).build()
    }

    /// Create a builder for fine-grained configuration.
    pub fn builder(model: impl Into<String>, base_url: impl Into<String>) -> VLLMBackendBuilder {
        VLLMBackendBuilder {
            model: model.into(),
            base_url: base_url.into(),
            api_key: "not-needed".into(),
            max_tokens: 256,
            extra_body: HashMap::new(),
        }
    }

    fn post(&self, body: Value) -> Result<ChatResponse> {
        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .headers(build_headers(&self.api_key))
            .json(&body)
            .send()?;
        response.error_for_status_ref()?;
        let data: Value = response.json()?;
        Ok(parse_response(&data))
    }

    async fn apost(&self, body: Value) -> Result<ChatResponse> {
        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .async_client
            .post(&url)
            .headers(build_headers(&self.api_key))
            .json(&body)
            .send()
            .await?;
        response
            .error_for_status_ref()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let data: Value = response.json().await?;
        Ok(parse_response(&data))
    }

    fn make_body(
        &self,
        messages: &[ChatMessage],
        temperature: f64,
        guided_json: Option<Value>,
        logprobs: bool,
        top_logprobs: u32,
    ) -> Value {
        build_body(
            &self.model,
            messages,
            temperature,
            guided_json,
            logprobs,
            top_logprobs,
            self.max_tokens,
            &self.extra_body,
        )
    }
}

#[async_trait::async_trait]
impl LLMBackend for VLLMBackend {
    fn model(&self) -> &str {
        &self.model
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn chat(
        &self,
        messages: &[ChatMessage],
        temperature: f64,
        guided_json: Option<Value>,
        logprobs: bool,
        top_logprobs: u32,
    ) -> Result<ChatResponse> {
        let body = self.make_body(messages, temperature, guided_json, logprobs, top_logprobs);
        self.post(body)
    }

    async fn achat(
        &self,
        messages: &[ChatMessage],
        temperature: f64,
        guided_json: Option<Value>,
        logprobs: bool,
        top_logprobs: u32,
    ) -> Result<ChatResponse> {
        let body = self.make_body(messages, temperature, guided_json, logprobs, top_logprobs);
        self.apost(body).await
    }
}

/// Builder for [`VLLMBackend`].
pub struct VLLMBackendBuilder {
    model: String,
    base_url: String,
    api_key: String,
    max_tokens: u32,
    extra_body: HashMap<String, Value>,
}

impl VLLMBackendBuilder {
    /// Set the API key. Defaults to "not-needed".
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = key.into();
        self
    }

    /// Set the maximum tokens to generate. Defaults to 256.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Add extra parameters merged into every request body.
    pub fn extra_body(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra_body.insert(key.into(), value);
        self
    }

    /// Build the [`VLLMBackend`].
    pub fn build(self) -> VLLMBackend {
        let client = reqwest::blocking::Client::builder()
            .build()
            .expect("failed to build sync client");
        let async_client = reqwest::Client::builder()
            .build()
            .expect("failed to build async client");
        VLLMBackend {
            model: self.model,
            base_url: self.base_url,
            api_key: self.api_key,
            max_tokens: self.max_tokens,
            extra_body: self.extra_body,
            client,
            async_client,
        }
    }
}
