//! Base backend trait and shared types for inference engines.
//!
//! All OpenAI-compatible backends (vLLM, SGLang, llama.cpp) share the helpers
//! in this module. The Ollama backend speaks its own native protocol and only
//! implements the [`LLMBackend`] trait.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// A single chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role (e.g. "system", "user", "assistant").
    pub role: String,
    /// Content of the message.
    pub content: String,
}

impl ChatMessage {
    /// Create a new chat message.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// A tokenizer token (text + id).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    /// Token text.
    pub text: String,
    /// Token id.
    pub id: i64,
}

/// Logprob info for a single generated/scored token position.
///
/// `top_logprobs` is flattened into a `{token: logprob}` map, matching the
/// shape the scoring functions consume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogprob {
    /// The token string.
    pub token: String,
    /// Token id (when reported by the server, otherwise `-1`).
    pub token_id: i64,
    /// Log probability of this token.
    pub logprob: f64,
    /// Candidate tokens and their logprobs at this position.
    pub top_logprobs: HashMap<String, f64>,
}

impl TokenLogprob {
    /// Create a new token-logprob entry with empty top-logprobs.
    pub fn new(token: impl Into<String>, logprob: f64) -> Self {
        Self {
            token: token.into(),
            token_id: -1,
            logprob,
            top_logprobs: HashMap::new(),
        }
    }
}

/// Response from a chat completion call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The generated content.
    pub content: String,
    /// The label extracted from the (possibly JSON-wrapped) content.
    pub label: String,
    /// Log probabilities per token, if requested.
    pub logprobs: Option<Vec<TokenLogprob>>,
    /// Raw JSON response from the API.
    pub raw: Value,
}

/// Response from a completion-scoring call (`backend.score`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringResponse {
    /// The completion text that was scored.
    pub completion: String,
    /// Per-token logprob info along the completion.
    pub logprobs: Vec<TokenLogprob>,
    /// Raw JSON response from the API.
    pub raw: Value,
}

/// Trait for LLM inference backends.
///
/// Each backend must provide three operations, each in synchronous and
/// asynchronous forms: [`chat`](LLMBackend::chat) (constrained generation),
/// [`score`](LLMBackend::score) (force-score a completion), and
/// [`tokenize`](LLMBackend::tokenize).
#[async_trait]
pub trait LLMBackend: Send + Sync {
    /// The model identifier.
    fn model(&self) -> &str;

    /// The base URL of the inference server.
    fn base_url(&self) -> &str;

    /// `true` if [`chat`](LLMBackend::chat) emits the bare label text as its
    /// content (vLLM/SGLang/llama.cpp); `false` when the label is JSON-wrapped
    /// (Ollama, which returns `{"label": "..."}`).
    fn supports_bare_label_constraint(&self) -> bool;

    /// Perform a synchronous constrained chat completion.
    ///
    /// When `constrain_labels` is `Some`, the backend must restrict output to
    /// exactly one of those labels using its native constraint mechanism.
    fn chat(
        &self,
        messages: &[ChatMessage],
        temperature: f64,
        constrain_labels: Option<&[String]>,
        logprobs: bool,
        top_logprobs: u32,
    ) -> Result<ChatResponse>;

    /// Score a forced completion synchronously.
    ///
    /// Appends `completion` to the rendered prompt and returns per-token
    /// logprobs along the completion tokens (no generation).
    fn score(&self, messages: &[ChatMessage], completion: &str) -> Result<ScoringResponse>;

    /// Tokenize text synchronously.
    ///
    /// When `context` is provided, tokenization is performed on `context +
    /// text` and the leading `context` token count is stripped, so the returned
    /// tokens describe `text` *in that context*.
    fn tokenize(&self, text: &str, context: Option<&str>) -> Result<Vec<Token>>;

    /// Asynchronous [`chat`](LLMBackend::chat).
    async fn achat(
        &self,
        messages: &[ChatMessage],
        temperature: f64,
        constrain_labels: Option<&[String]>,
        logprobs: bool,
        top_logprobs: u32,
    ) -> Result<ChatResponse>;

    /// Asynchronous [`score`](LLMBackend::score).
    async fn ascore(&self, messages: &[ChatMessage], completion: &str) -> Result<ScoringResponse>;

    /// Asynchronous [`tokenize`](LLMBackend::tokenize).
    async fn atokenize(&self, text: &str, context: Option<&str>) -> Result<Vec<Token>>;
}

// =========================================================================
// Shared HTTP helpers (OpenAI-compatible servers)
// =========================================================================

/// Strip a trailing `/` from a base URL (mirrors the Python base constructor).
pub fn normalize_base_url(url: impl Into<String>) -> String {
    let mut s = url.into();
    while s.ends_with('/') {
        s.pop();
    }
    s
}

/// Build HTTP headers for the OpenAI-compatible API.
pub fn build_headers(api_key: &str) -> reqwest::header::HeaderMap {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("application/json"),
    );
    headers.insert(
        reqwest::header::AUTHORIZATION,
        reqwest::header::HeaderValue::from_str(&format!("Bearer {api_key}"))
            .expect("api_key should be valid header value"),
    );
    headers
}

/// Build the base JSON body for an OpenAI-compatible `/chat/completions` call.
///
/// Constraint application is backend-specific and applied by each backend
/// after this shared body is constructed.
#[allow(clippy::too_many_arguments)]
pub fn build_chat_body(
    model: &str,
    messages: &[ChatMessage],
    temperature: f64,
    logprobs: bool,
    top_logprobs: u32,
    max_tokens: u32,
    extra_body: &HashMap<String, Value>,
) -> Value {
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages.iter().map(|m| {
            serde_json::json!({"role": m.role, "content": m.content})
        }).collect::<Vec<_>>(),
        "temperature": temperature,
        "max_tokens": max_tokens,
    });

    if logprobs {
        body["logprobs"] = serde_json::json!(true);
        body["top_logprobs"] = serde_json::json!(top_logprobs);
    } else {
        // Explicitly opt out so servers that default to returning logprobs
        // behave consistently.
        body["logprobs"] = serde_json::json!(false);
    }

    for (key, val) in extra_body {
        body[key] = val.clone();
    }

    body
}

/// Parse an OpenAI-compatible `/chat/completions` response.
///
/// Flattens each token position's `top_logprobs` array into a
/// `{token: logprob}` map.
pub fn parse_chat_response(data: &Value) -> ChatResponse {
    let choice = &data["choices"][0];
    let content = choice["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let logprobs_list = choice
        .get("logprobs")
        .and_then(|lp| lp.get("content"))
        .map(parse_token_logprob_array);

    let label = content.trim().to_string();

    ChatResponse {
        content,
        label,
        logprobs: logprobs_list,
        raw: data.clone(),
    }
}

/// Parse an array of OpenAI-style token-logprob objects into [`TokenLogprob`]s.
pub fn parse_token_logprob_array(arr: &Value) -> Vec<TokenLogprob> {
    let mut result = Vec::new();
    if let Some(arr) = arr.as_array() {
        for token_info in arr {
            let token = token_info["token"].as_str().unwrap_or("").to_string();
            let logprob = token_info["logprob"].as_f64().unwrap_or(0.0);
            let token_id = token_info["bytes"]
                .as_array()
                .map(|a| a.len() as i64)
                .unwrap_or(-1);
            let mut top = HashMap::new();
            if let Some(tlp_arr) = token_info.get("top_logprobs").and_then(|t| t.as_array()) {
                for entry in tlp_arr {
                    if let (Some(t), Some(lp)) =
                        (entry["token"].as_str(), entry["logprob"].as_f64())
                    {
                        top.insert(t.to_string(), lp);
                    }
                }
            }
            result.push(TokenLogprob {
                token,
                token_id,
                logprob,
                top_logprobs: top,
            });
        }
    }
    result
}

/// Render chat messages into a flat prompt string for the `/completions`
/// scoring path, matching the shape used by the OpenAI-compatible backends:
///
/// ```text
/// <|system|>
/// {system content}
///
/// <|user|>
/// {user content}
///
/// <|assistant|>
/// ```
pub fn render_prompt(messages: &[ChatMessage]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for m in messages {
        parts.push(format!(
            "<|{role}|>\n{content}",
            role = m.role,
            content = m.content
        ));
    }
    parts.push("<|assistant|>\n".to_string());
    parts.join("\n\n")
}
