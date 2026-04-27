//! Base backend trait and shared types for inference engines.

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

/// A single top-logprob entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopLogProb {
    /// Token string.
    pub token: String,
    /// Log probability.
    pub logprob: f64,
}

/// Logprob info for a single token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogProb {
    /// The token.
    pub token: String,
    /// Log probability of this token.
    pub logprob: f64,
    /// Top log probabilities for this token position.
    pub top_logprobs: Vec<TopLogProb>,
}

/// Response from a chat completion call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The generated content.
    pub content: String,
    /// Log probabilities per token, if requested.
    pub logprobs: Option<Vec<TokenLogProb>>,
    /// Raw JSON response from the API.
    pub raw: Value,
}

/// Trait for LLM inference backends.
///
/// All backends communicate via HTTP using the OpenAI-compatible chat
/// completions API (which vLLM, SGLang, and llama.cpp server all support).
#[async_trait]
pub trait LLMBackend: Send + Sync {
    /// The model identifier.
    fn model(&self) -> &str;

    /// The base URL of the inference server.
    fn base_url(&self) -> &str;

    /// Perform a synchronous chat completion.
    fn chat(
        &self,
        messages: &[ChatMessage],
        temperature: f64,
        guided_json: Option<Value>,
        logprobs: bool,
        top_logprobs: u32,
    ) -> Result<ChatResponse>;

    /// Perform an asynchronous chat completion.
    async fn achat(
        &self,
        messages: &[ChatMessage],
        temperature: f64,
        guided_json: Option<Value>,
        logprobs: bool,
        top_logprobs: u32,
    ) -> Result<ChatResponse>;
}

/// Shared helper: build HTTP headers for the OpenAI-compatible API.
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

#[allow(clippy::too_many_arguments)]
/// Shared helper: build the JSON request body for the OpenAI-compatible API.
pub fn build_body(
    model: &str,
    messages: &[ChatMessage],
    temperature: f64,
    guided_json: Option<Value>,
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

    if let Some(schema) = guided_json {
        body["guided_json"] = schema.clone();
        body["response_format"] = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "classification",
                "schema": schema,
                "strict": true,
            }
        });
    }

    if logprobs {
        body["logprobs"] = serde_json::json!(true);
        body["top_logprobs"] = serde_json::json!(top_logprobs);
    }

    // Merge extra_body overrides
    for (key, val) in extra_body {
        body[key] = val.clone();
    }

    body
}

/// Shared helper: parse the JSON response from the OpenAI-compatible API.
pub fn parse_response(data: &Value) -> ChatResponse {
    let choice = &data["choices"][0];
    let content = choice["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let logprobs_list = choice
        .get("logprobs")
        .and_then(|lp| lp.get("content"))
        .map(|lp_arr| {
            let mut result = Vec::new();
            if let Some(arr) = lp_arr.as_array() {
                for token_info in arr {
                    let token = token_info["token"].as_str().unwrap_or("").to_string();
                    let logprob = token_info["logprob"].as_f64().unwrap_or(0.0);
                    let top_logprobs = token_info
                        .get("top_logprobs")
                        .and_then(|tlp| tlp.as_array())
                        .map(|tlp_arr| {
                            tlp_arr
                                .iter()
                                .filter_map(|entry| {
                                    Some(TopLogProb {
                                        token: entry["token"].as_str()?.to_string(),
                                        logprob: entry["logprob"].as_f64()?,
                                    })
                                })
                                .collect()
                        })
                        .unwrap_or_default();

                    result.push(TokenLogProb {
                        token,
                        logprob,
                        top_logprobs,
                    });
                }
            }
            result
        });

    ChatResponse {
        content,
        logprobs: logprobs_list,
        raw: data.clone(),
    }
}
