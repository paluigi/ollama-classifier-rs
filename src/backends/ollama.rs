//! Ollama inference backend.
//!
//! Talks to [Ollama](https://ollama.com) via its native API (`/api/chat`) over
//! HTTP. Logprobs support requires Ollama ≥ 0.12. Unlike the OpenAI-compatible
//! backends, Ollama wraps the constrained label in a JSON object
//! `{"label": "..."}`, so it reports
//! [`supports_bare_label_constraint`](super::LLMBackend::supports_bare_label_constraint)
//! as `false`.
//!
//! Modern Ollama removed the `/api/tokenize` endpoint and does not support
//! fill-in-the-middle ("insert") on instruct models. This backend therefore
//! obtains both label tokenization and completion scores through empirical
//! *forced constrained generation*: it forces a label as the only valid choice
//! in a `chat()` call and reads back the model's genuine per-token logprobs.
//! No `/api/tokenize` or `suffix`/insert calls are used. Tokenization results
//! are memoized per label.
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
use std::sync::Mutex;
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
    /// Per-label tokenization memoization cache.
    token_cache: Mutex<HashMap<String, Vec<Token>>>,
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
        // Force the completion as the only valid label via a JSON-enum
        // constrained chat() call and read back the model's genuine per-token
        // logprobs (teacher forcing).
        let labels = vec![completion.to_string()];
        let response = self.chat(messages, 0.0, Some(&labels), true, 1)?;
        let lps =
            label_token_logprobs(response.logprobs.as_deref().unwrap_or_default(), completion);
        if lps.is_empty() {
            anyhow::bail!("score({completion:?}): forced generation returned no value tokens");
        }
        Ok(ScoringResponse {
            completion: completion.to_string(),
            logprobs: lps,
            raw: response.raw,
        })
    }

    fn tokenize(&self, text: &str, _context: Option<&str>) -> Result<Vec<Token>> {
        // Check cache
        if let Some(cached) = self.token_cache.lock().unwrap().get(text) {
            return Ok(cached.clone());
        }

        // Force the text as the only valid label in a constrained chat() call.
        let messages = vec![ChatMessage::new("user", text)];
        let labels = vec![text.to_string()];
        let response = self.chat(&messages, 0.0, Some(&labels), true, 1)?;
        let lps = label_token_logprobs(response.logprobs.as_deref().unwrap_or_default(), text);
        let tokens: Vec<Token> = if lps.is_empty() {
            vec![Token {
                text: text.to_string(),
                id: -1,
            }]
        } else {
            lps.iter()
                .map(|lp| Token {
                    text: lp.token.clone(),
                    id: -1,
                })
                .collect()
        };

        // Memoize
        self.token_cache
            .lock()
            .unwrap()
            .insert(text.to_string(), tokens.clone());

        Ok(tokens)
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
        let labels = vec![completion.to_string()];
        let response = self.achat(messages, 0.0, Some(&labels), true, 1).await?;
        let lps =
            label_token_logprobs(response.logprobs.as_deref().unwrap_or_default(), completion);
        if lps.is_empty() {
            anyhow::bail!("ascore({completion:?}): forced generation returned no value tokens");
        }
        Ok(ScoringResponse {
            completion: completion.to_string(),
            logprobs: lps,
            raw: response.raw,
        })
    }

    async fn atokenize(&self, text: &str, _context: Option<&str>) -> Result<Vec<Token>> {
        // Check cache (shared between sync and async)
        if let Some(cached) = self.token_cache.lock().unwrap().get(text) {
            return Ok(cached.clone());
        }

        let messages = vec![ChatMessage::new("user", text)];
        let labels = vec![text.to_string()];
        let response = self.achat(&messages, 0.0, Some(&labels), true, 1).await?;
        let lps = label_token_logprobs(response.logprobs.as_deref().unwrap_or_default(), text);
        let tokens: Vec<Token> = if lps.is_empty() {
            vec![Token {
                text: text.to_string(),
                id: -1,
            }]
        } else {
            lps.iter()
                .map(|lp| Token {
                    text: lp.token.clone(),
                    id: -1,
                })
                .collect()
        };

        self.token_cache
            .lock()
            .unwrap()
            .insert(text.to_string(), tokens.clone());

        Ok(tokens)
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
            token_cache: Mutex::new(HashMap::new()),
        }
    }
}

// =========================================================================
// Helpers
// =========================================================================

/// Build the JSON schema Ollama uses as `format` to constrain output to one of
/// the given labels.
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

/// Extract the label-value tokens (with their logprobs) from a
/// `{"label": "<label>"}` constrained response.
///
/// Robust to model-specific whitespace in the emitted JSON. The returned
/// tokens keep their *exact* emitted strings so they match the tokens the
/// model produces during multi-label constrained generation in `generate()`.
///
/// Primary strategy: reconstruct the full emitted string, locate the value
/// span after the JSON `:` separator, and map that character span back to
/// token indices. Falls back to JSON-skeleton filtering if the span mapping
/// yields nothing.
fn label_token_logprobs(logprobs: &[TokenLogprob], label: &str) -> Vec<TokenLogprob> {
    if logprobs.is_empty() {
        return Vec::new();
    }

    // Reconstruct the full emitted string
    let full: String = logprobs.iter().map(|lp| lp.token.as_str()).collect();

    // ---- Primary: character-offset span mapping ----
    if let Some(result) = span_map_logprobs(&full, logprobs, label) {
        if !result.is_empty() {
            return result;
        }
    }

    // ---- Fallback: drop pure JSON-structure tokens / the "label" key ----
    logprobs
        .iter()
        .filter(|lp| {
            let stripped = lp.token.trim();
            let cleaned: String = stripped
                .chars()
                .filter(|c| !matches!(c, '"' | '{' | '}' | ':' | ' ' | '\t' | '\n'))
                .collect();
            !cleaned.is_empty() && stripped != "label"
        })
        .cloned()
        .collect()
}

/// Map the character span of `label` within the full emitted string back to
/// token indices.
fn span_map_logprobs(
    full: &str,
    logprobs: &[TokenLogprob],
    label: &str,
) -> Option<Vec<TokenLogprob>> {
    // Find the colon separator
    let colon = full.find(':')?;
    // Search for the label after the colon
    let after_colon = &full[colon + 1..];
    let label_rel = after_colon.find(label)?;
    let vstart = colon + 1 + label_rel;
    let vend = vstart + label.len();

    // Map char span to token indices
    let mut out = Vec::new();
    let mut pos = 0; // byte offset
    for lp in logprobs {
        let tok_len = lp.token.len();
        let tok_end = pos + tok_len;
        if tok_end > vstart && pos < vend {
            out.push(lp.clone());
        }
        pos = tok_end;
    }

    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_token_logprobs_single_token() {
        let logprobs = vec![
            TokenLogprob::new("{", -17.726),
            TokenLogprob::new(" \"", -13.196),
            TokenLogprob::new("label", 0.0),
            TokenLogprob::new("\":", 0.0),
            TokenLogprob::new(" \"", 0.0),
            TokenLogprob::new("sports", -1.288),
            TokenLogprob::new("\"", -0.001),
            TokenLogprob::new(" }", 0.0),
        ];
        let out = label_token_logprobs(&logprobs, "sports");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token, "sports");
        assert!((out[0].logprob - (-1.288)).abs() < 1e-9);
    }

    #[test]
    fn test_label_token_logprobs_multi_token() {
        let logprobs = vec![
            TokenLogprob::new("{\"label\": \"", -10.0),
            TokenLogprob::new("tech", -0.5),
            TokenLogprob::new(" support", -0.7),
            TokenLogprob::new("\" }", 0.0),
        ];
        let out = label_token_logprobs(&logprobs, "tech support");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].token, "tech");
        assert_eq!(out[1].token, " support");
    }

    #[test]
    fn test_label_token_logprobs_compact_json() {
        let logprobs = vec![
            TokenLogprob::new("{\"label\":\"", -10.0),
            TokenLogprob::new("sports", -1.288),
            TokenLogprob::new("\"}", 0.0),
        ];
        let out = label_token_logprobs(&logprobs, "sports");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token, "sports");
    }

    #[test]
    fn test_label_token_logprobs_fallback() {
        // Label text never appears → primary mapping fails, fallback keeps
        // non-structure tokens.
        let logprobs = vec![
            TokenLogprob::new("{", -10.0),
            TokenLogprob::new(" \"", -10.0),
            TokenLogprob::new("label", 0.0),
            TokenLogprob::new("\":", 0.0),
            TokenLogprob::new(" \"", 0.0),
            TokenLogprob::new("sports", -1.288),
            TokenLogprob::new("\"", 0.0),
            TokenLogprob::new("}", 0.0),
        ];
        let out = label_token_logprobs(&logprobs, "missing");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token, "sports");
    }

    #[test]
    fn test_label_token_logprobs_empty() {
        let out = label_token_logprobs(&[], "sports");
        assert!(out.is_empty());
    }
}
