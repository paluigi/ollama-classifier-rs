//! Ollama inference backend.
//!
//! Talks to [Ollama](https://ollama.com) via its native API
//! (`/api/chat`, `/api/generate`, `/api/tokenize`) over HTTP. Logprobs support
//! requires Ollama ≥ 0.12. Unlike the OpenAI-compatible backends, Ollama
//! wraps the constrained label in a JSON object `{"label": "..."}`, so it
//! reports [`supports_bare_label_constraint`](super::LLMBackend::supports_bare_label_constraint)
//! as `false`.
//!
//! # Example
//!
//! ```no_run
//! use ollama_classifier_rs::backends::OllamaBackend;
//! use ollama_classifier_rs::LLMClassifier;
//!
//! let backend = OllamaBackend::new("llama3.2");
//! let classifier = LLMClassifier::new(backend);
//! ```

use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::{
    build_headers, normalize_base_url, ChatMessage, ChatResponse, LLMBackend, ScoringResponse,
    Token, TokenLogprob,
};

/// Default Ollama host URL.
pub const DEFAULT_HOST: &str = "http://localhost:11434";

/// JSON prefix that Ollama emits before the label text when using the
/// JSON-schema enum constraint. Used as the tokenization context for labels so
/// that token boundaries match what the server actually generates.
pub const JSON_LABEL_CONTEXT: &str = "{\"label\": \"";

/// Backend for a native Ollama server.
pub struct OllamaBackend {
    model: String,
    host: String,
    /// Configured request timeout (applied to the reqwest clients at build
    /// time; retained for introspection).
    #[allow(dead_code)]
    timeout: Duration,
    max_tokens: u32,
    extra_body: HashMap<String, Value>,
    client: reqwest::blocking::Client,
    async_client: reqwest::Client,
}

impl OllamaBackend {
    /// Create a new Ollama backend pointed at the default host.
    pub fn new(model: impl Into<String>) -> Self {
        Self::with_config(model, DEFAULT_HOST)
    }

    /// Create an Ollama backend with a custom host URL.
    pub fn with_config(model: impl Into<String>, host: impl Into<String>) -> Self {
        Self::builder(model, host).build()
    }

    /// Create a builder for fine-grained configuration.
    pub fn builder(model: impl Into<String>, host: impl Into<String>) -> OllamaBackendBuilder {
        OllamaBackendBuilder {
            model: model.into(),
            host: host.into(),
            timeout: Duration::from_secs(120),
            max_tokens: 256,
            extra_body: HashMap::new(),
        }
    }
}

#[async_trait]
impl LLMBackend for OllamaBackend {
    fn model(&self) -> &str {
        &self.model
    }

    fn base_url(&self) -> &str {
        &self.host
    }

    fn supports_bare_label_constraint(&self) -> bool {
        false
    }

    fn chat(
        &self,
        messages: &[ChatMessage],
        temperature: f64,
        constrain_labels: Option<&[String]>,
        logprobs: bool,
        top_logprobs: u32,
    ) -> Result<ChatResponse> {
        let mut body = json!({
            "model": self.model,
            "messages": messages.iter().map(|m| {
                json!({"role": m.role, "content": m.content})
            }).collect::<Vec<_>>(),
            "stream": false,
            "options": { "temperature": temperature, "num_predict": self.max_tokens },
        });
        if let Some(labels) = constrain_labels {
            body["format"] = json_schema_enum(labels);
        }
        if logprobs {
            body["logprobs"] = json!(top_logprobs);
        }
        merge_extra(&mut body, &self.extra_body);

        let data = self.post("/api/chat", &body)?;
        let content = data["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let label = extract_label(&content);
        let lps = data
            .get("logprobs")
            .and_then(|v| v.as_array())
            .map(|a| parse_ollama_logprobs(a));
        Ok(ChatResponse {
            content,
            label,
            logprobs: lps,
            raw: data,
        })
    }

    fn score(&self, messages: &[ChatMessage], completion: &str) -> Result<ScoringResponse> {
        let prompt = render_prompt(messages);
        let mut body = json!({
            "model": self.model,
            "prompt": prompt,
            "suffix": completion,
            "stream": false,
            "logprobs": 1,
            "options": { "num_predict": 0, "temperature": 0.0 },
        });
        merge_extra(&mut body, &self.extra_body);

        let data = self.post("/api/generate", &body)?;
        let lps = data
            .get("logprobs")
            .and_then(|v| v.as_array())
            .map(|a| parse_ollama_logprobs(a))
            .unwrap_or_default();
        Ok(ScoringResponse {
            completion: completion.to_string(),
            logprobs: lps,
            raw: data,
        })
    }

    fn tokenize(&self, text: &str, context: Option<&str>) -> Result<Vec<Token>> {
        tokenize_native(&self.client, &self.host, &self.model, text, context)
    }

    async fn achat(
        &self,
        messages: &[ChatMessage],
        temperature: f64,
        constrain_labels: Option<&[String]>,
        logprobs: bool,
        top_logprobs: u32,
    ) -> Result<ChatResponse> {
        let mut body = json!({
            "model": self.model,
            "messages": messages.iter().map(|m| {
                json!({"role": m.role, "content": m.content})
            }).collect::<Vec<_>>(),
            "stream": false,
            "options": { "temperature": temperature, "num_predict": self.max_tokens },
        });
        if let Some(labels) = constrain_labels {
            body["format"] = json_schema_enum(labels);
        }
        if logprobs {
            body["logprobs"] = json!(top_logprobs);
        }
        merge_extra(&mut body, &self.extra_body);

        let data = self.apost("/api/chat", &body).await?;
        let content = data["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let label = extract_label(&content);
        let lps = data
            .get("logprobs")
            .and_then(|v| v.as_array())
            .map(|a| parse_ollama_logprobs(a));
        Ok(ChatResponse {
            content,
            label,
            logprobs: lps,
            raw: data,
        })
    }

    async fn ascore(&self, messages: &[ChatMessage], completion: &str) -> Result<ScoringResponse> {
        let prompt = render_prompt(messages);
        let mut body = json!({
            "model": self.model,
            "prompt": prompt,
            "suffix": completion,
            "stream": false,
            "logprobs": 1,
            "options": { "num_predict": 0, "temperature": 0.0 },
        });
        merge_extra(&mut body, &self.extra_body);

        let data = self.apost("/api/generate", &body).await?;
        let lps = data
            .get("logprobs")
            .and_then(|v| v.as_array())
            .map(|a| parse_ollama_logprobs(a))
            .unwrap_or_default();
        Ok(ScoringResponse {
            completion: completion.to_string(),
            logprobs: lps,
            raw: data,
        })
    }

    async fn atokenize(&self, text: &str, context: Option<&str>) -> Result<Vec<Token>> {
        tokenize_native_async(&self.async_client, &self.host, &self.model, text, context).await
    }
}

impl OllamaBackend {
    fn post(&self, path: &str, body: &Value) -> Result<Value> {
        let url = format!("{}{}", self.host, path);
        let response = self
            .client
            .post(&url)
            .headers(build_headers("not-needed"))
            .json(body)
            .send()?;
        response.error_for_status_ref()?;
        Ok(response.json()?)
    }

    async fn apost(&self, path: &str, body: &Value) -> Result<Value> {
        let url = format!("{}{}", self.host, path);
        let response = self
            .async_client
            .post(&url)
            .headers(build_headers("not-needed"))
            .json(body)
            .send()
            .await?;
        response
            .error_for_status_ref()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(response.json().await?)
    }
}

/// Builder for [`OllamaBackend`].
pub struct OllamaBackendBuilder {
    model: String,
    host: String,
    timeout: Duration,
    max_tokens: u32,
    extra_body: HashMap<String, Value>,
}

impl OllamaBackendBuilder {
    /// Set the request timeout. Defaults to 120s.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the maximum tokens to generate. Defaults to 256.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Add an extra parameter merged into every request body.
    pub fn extra_body(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra_body.insert(key.into(), value);
        self
    }

    /// Build the [`OllamaBackend`].
    pub fn build(self) -> OllamaBackend {
        let client = reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .build()
            .expect("failed to build sync client");
        let async_client = reqwest::Client::builder()
            .timeout(self.timeout)
            .build()
            .expect("failed to build async client");
        OllamaBackend {
            model: self.model,
            host: normalize_base_url(self.host),
            timeout: self.timeout,
            max_tokens: self.max_tokens,
            extra_body: self.extra_body,
            client,
            async_client,
        }
    }
}

// =========================================================================
// Helpers
// =========================================================================

/// Build the JSON schema Ollama uses as `format` to constrain output to one of
/// the given labels: `{"type":"object","properties":{"label":{"type":"string","enum":[...]}},"required":["label"]}`.
fn json_schema_enum(labels: &[String]) -> Value {
    json!({
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": labels,
            }
        },
        "required": ["label"],
    })
}

/// Extract the label from Ollama's JSON-wrapped content (`{"label": "..."}`),
/// falling back to the trimmed raw content.
fn extract_label(content: &str) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(content) {
        if let Some(s) = v.get("label").and_then(|l| l.as_str()) {
            return s.to_string();
        }
    }
    content.trim().to_string()
}

/// Merge `extra_body` keys into a request body.
fn merge_extra(body: &mut Value, extra_body: &HashMap<String, Value>) {
    for (k, v) in extra_body {
        body[k] = v.clone();
    }
}

/// Parse Ollama's logprobs array shape (`[{token, logprob, top_logprobs: [...]}]`).
fn parse_ollama_logprobs(arr: &[Value]) -> Vec<TokenLogprob> {
    arr.iter()
        .map(|entry| {
            let token = entry["token"].as_str().unwrap_or("").to_string();
            let logprob = entry["logprob"].as_f64().unwrap_or(0.0);
            let mut top = HashMap::new();
            if let Some(tlps) = entry.get("top_logprobs").and_then(|t| t.as_array()) {
                for t in tlps {
                    if let (Some(tok), Some(lp)) = (t["token"].as_str(), t["logprob"].as_f64()) {
                        top.insert(tok.to_string(), lp);
                    }
                }
            }
            TokenLogprob {
                token,
                token_id: -1,
                logprob,
                top_logprobs: top,
            }
        })
        .collect()
}

/// Render chat messages into the prompt used for Ollama `/api/generate` scoring.
///
/// Matches the OpenAI-compatible `<|system|> / <|user|> / <|assistant|>` framing
/// so that scoring is consistent across backends.
fn render_prompt(messages: &[ChatMessage]) -> String {
    super::base::render_prompt(messages)
}

/// Tokenize via `/api/tokenize`, optionally stripping a context-prefix token count.
fn tokenize_native(
    client: &reqwest::blocking::Client,
    host: &str,
    model: &str,
    text: &str,
    context: Option<&str>,
) -> Result<Vec<Token>> {
    match context {
        None => {
            let body = json!({ "model": model, "text": text });
            let data = post_native_sync(client, host, "/api/tokenize", &body)?;
            Ok(parse_ollama_tokens(&data, text))
        }
        Some(ctx) => {
            let combined = format!("{ctx}{text}");
            let body = json!({ "model": model, "text": combined });
            let data = post_native_sync(client, host, "/api/tokenize", &body)?;
            let ctx_body = json!({ "model": model, "text": ctx });
            let ctx_data = post_native_sync(client, host, "/api/tokenize", &ctx_body)?;
            let ctx_count = count_ollama_tokens(&ctx_data);
            let mut all = parse_ollama_tokens(&data, &combined);
            if ctx_count <= all.len() {
                all.drain(..ctx_count);
            }
            Ok(all)
        }
    }
}

async fn tokenize_native_async(
    client: &reqwest::Client,
    host: &str,
    model: &str,
    text: &str,
    context: Option<&str>,
) -> Result<Vec<Token>> {
    match context {
        None => {
            let body = json!({ "model": model, "text": text });
            let data = post_native_async(client, host, "/api/tokenize", &body).await?;
            Ok(parse_ollama_tokens(&data, text))
        }
        Some(ctx) => {
            let combined = format!("{ctx}{text}");
            let body = json!({ "model": model, "text": combined });
            let data = post_native_async(client, host, "/api/tokenize", &body).await?;
            let ctx_body = json!({ "model": model, "text": ctx });
            let ctx_data = post_native_async(client, host, "/api/tokenize", &ctx_body).await?;
            let ctx_count = count_ollama_tokens(&ctx_data);
            let mut all = parse_ollama_tokens(&data, &combined);
            if ctx_count <= all.len() {
                all.drain(..ctx_count);
            }
            Ok(all)
        }
    }
}

fn post_native_sync(
    client: &reqwest::blocking::Client,
    host: &str,
    path: &str,
    body: &Value,
) -> Result<Value> {
    let url = format!("{host}{path}");
    let response = client.post(&url).json(body).send()?;
    response.error_for_status_ref()?;
    Ok(response.json()?)
}

async fn post_native_async(
    client: &reqwest::Client,
    host: &str,
    path: &str,
    body: &Value,
) -> Result<Value> {
    let url = format!("{host}{path}");
    let response = client.post(&url).json(body).send().await?;
    response
        .error_for_status_ref()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(response.json().await?)
}

/// Parse Ollama's `/api/tokenize` response.
///
/// Returns `{ tokens: [str...] }`; token ids are reported by some servers in a
/// parallel `token_ids` array and are paired by index when present.
fn parse_ollama_tokens(data: &Value, _fallback_text: &str) -> Vec<Token> {
    let tokens = match data.get("tokens").and_then(|t| t.as_array()) {
        Some(a) => a,
        None => return Vec::new(),
    };
    let ids = data.get("token_ids").and_then(|v| v.as_array());
    tokens
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let text = t.as_str().unwrap_or("").to_string();
            let id = ids
                .and_then(|a| a.get(i).and_then(|v| v.as_i64()))
                .unwrap_or(-1);
            Token {
                text: if text.is_empty() {
                    format!("token_{id}")
                } else {
                    text
                },
                id,
            }
        })
        .collect()
}

fn count_ollama_tokens(data: &Value) -> usize {
    data.get("tokens")
        .and_then(|t| t.as_array())
        .map(|a| a.len())
        .unwrap_or(0)
}
