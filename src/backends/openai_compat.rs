//! Shared core for OpenAI-compatible inference backends.
//!
//! vLLM, SGLang, and llama.cpp all expose the OpenAI-compatible
//! `/chat/completions`, `/completions`, and `/tokenize` endpoints. They differ
//! only in (a) the field used to constrain output to a label set and (b) how
//! the completion/prompt token boundary is found when scoring. Those
//! differences are captured by [`Constraint`] and [`BoundaryStrategy`]; the
//! common HTTP plumbing lives here.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;
use serde_json::Value;

use super::base::{
    build_chat_body, build_headers, normalize_base_url, parse_chat_response,
    parse_token_logprob_array, render_prompt, ChatMessage, ChatResponse, ScoringResponse, Token,
    TokenLogprob,
};

/// How an OpenAI-compatible backend constrains output to a label set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Constraint {
    /// vLLM: `guided_choice` = labels.
    GuidedChoice,
    /// SGLang: `regex` = `(label1|label2|...)`.
    Regex,
    /// llama.cpp: `grammar` = GBNF rule `root ::= "a" | "b" | ...`.
    Grammar,
}

impl Constraint {
    /// Apply this constraint to an OpenAI-compatible request body.
    pub fn apply(&self, body: &mut Value, labels: &[String]) {
        match self {
            Constraint::GuidedChoice => {
                body["guided_choice"] =
                    Value::Array(labels.iter().map(|l| Value::String(l.clone())).collect());
            }
            Constraint::Regex => {
                let alts: Vec<String> = labels.iter().map(|l| regex_escape(l)).collect();
                body["regex"] = Value::String(format!("({})", alts.join("|")));
            }
            Constraint::Grammar => {
                let alts: Vec<String> = labels
                    .iter()
                    .map(|l| format!("\"{}\"", grammar_escape(l)))
                    .collect();
                body["grammar"] = Value::String(format!("root ::= {}", alts.join(" | ")));
            }
        }
    }
}

/// Escape special characters for a Python `re`-style regex alternation.
fn regex_escape(s: &str) -> String {
    // vLLM/SGLang use Python-flavored regex; escape the same metacharacters.
    const SPECIAL: &[char] = &[
        '.', '^', '$', '*', '+', '?', '(', ')', '[', ']', '{', '}', '\\', '|', '/',
    ];
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if SPECIAL.contains(&c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Escape a label for inclusion in a GBNF string literal.
fn grammar_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// How to find the prompt/completion token boundary when scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryStrategy {
    /// vLLM: tokenize the full prompt by id and use its token count.
    Ids,
    /// SGLang: tokenize the full prompt by count and use its token count.
    Count,
    /// llama.cpp: the server fills the middle via `suffix`; locate completion
    /// tokens with a heuristic over the returned token array.
    FillMiddle,
}

/// Shared, reusable core for an OpenAI-compatible backend.
///
/// Constructed via [`OpenAICompatCore::builder`] and held by value inside each
/// public backend type.
pub struct OpenAICompatCore {
    pub(crate) model: String,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    /// Configured request timeout (applied to the reqwest clients at build
    /// time; retained for introspection).
    #[allow(dead_code)]
    pub(crate) timeout: Duration,
    pub(crate) max_tokens: u32,
    pub(crate) extra_body: HashMap<String, Value>,
    pub(crate) constraint: Constraint,
    pub(crate) boundary: BoundaryStrategy,
    pub(crate) client: reqwest::blocking::Client,
    pub(crate) async_client: reqwest::Client,
}

impl OpenAICompatCore {
    /// Build a chat request body, applying this backend's constraint when labels are given.
    pub(crate) fn chat_body(
        &self,
        messages: &[ChatMessage],
        temperature: f64,
        constrain_labels: Option<&[String]>,
        logprobs: bool,
        top_logprobs: u32,
    ) -> Value {
        let mut body = build_chat_body(
            &self.model,
            messages,
            temperature,
            logprobs,
            top_logprobs,
            self.max_tokens,
            &self.extra_body,
        );
        if let Some(labels) = constrain_labels {
            self.constraint.apply(&mut body, labels);
        }
        body
    }

    /// Synchronous `/chat/completions`.
    pub(crate) fn post_chat(&self, body: Value) -> Result<ChatResponse> {
        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .headers(build_headers(&self.api_key))
            .json(&body)
            .send()?;
        response.error_for_status_ref()?;
        let data: Value = response.json()?;
        Ok(parse_chat_response(&data))
    }

    /// Asynchronous `/chat/completions`.
    pub(crate) async fn apost_chat(&self, body: Value) -> Result<ChatResponse> {
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
        Ok(parse_chat_response(&data))
    }

    // ----- scoring (completions endpoint) --------------------------------

    /// Build the `/completions` scoring request body.
    fn scoring_body(&self, prompt: &str, suffix: Option<&str>) -> Value {
        let echo = matches!(
            self.boundary,
            BoundaryStrategy::Ids | BoundaryStrategy::Count
        );
        let max_tokens = match self.boundary {
            BoundaryStrategy::Ids => 1,
            BoundaryStrategy::Count => 1,
            BoundaryStrategy::FillMiddle => 0,
        };
        let logprobs_topn: i64 = match self.boundary {
            BoundaryStrategy::Ids => 0,
            BoundaryStrategy::Count => 1,
            BoundaryStrategy::FillMiddle => 1,
        };
        let mut body = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "logprobs": logprobs_topn,
            "echo": echo,
        });
        if let Some(sfx) = suffix {
            body["suffix"] = Value::String(sfx.to_string());
        }
        for (k, v) in &self.extra_body {
            body[k] = v.clone();
        }
        body
    }

    /// Synchronous scoring via `/completions`.
    pub(crate) fn post_score(
        &self,
        messages: &[ChatMessage],
        completion: &str,
    ) -> Result<ScoringResponse> {
        match self.boundary {
            BoundaryStrategy::Ids => self.post_score_ids(messages, completion),
            BoundaryStrategy::Count => self.post_score_count(messages, completion),
            BoundaryStrategy::FillMiddle => self.post_score_fill_middle(messages, completion),
        }
    }

    /// Asynchronous scoring via `/completions`.
    pub(crate) async fn apost_score(
        &self,
        messages: &[ChatMessage],
        completion: &str,
    ) -> Result<ScoringResponse> {
        match self.boundary {
            BoundaryStrategy::Ids => self.apost_score_ids(messages, completion).await,
            BoundaryStrategy::Count => self.apost_score_count(messages, completion).await,
            BoundaryStrategy::FillMiddle => {
                self.apost_score_fill_middle(messages, completion).await
            }
        }
    }

    fn post_score_ids(
        &self,
        messages: &[ChatMessage],
        completion: &str,
    ) -> Result<ScoringResponse> {
        let prompt = format!("{}{}", render_prompt(messages), completion);
        let prompt_token_count = self.tokenize_count_ids(&prompt)?;
        let body = self.scoring_body(&prompt, None);
        let data = self.post_completions_raw(&body)?;
        let all = extract_completions_logprobs(&data);
        let completion_logprobs = if prompt_token_count <= all.len() {
            all[prompt_token_count..].to_vec()
        } else {
            all
        };
        Ok(ScoringResponse {
            completion: completion.to_string(),
            logprobs: completion_logprobs,
            raw: data,
        })
    }

    fn post_score_count(
        &self,
        messages: &[ChatMessage],
        completion: &str,
    ) -> Result<ScoringResponse> {
        let prompt = format!("{}{}", render_prompt(messages), completion);
        let prompt_token_count = self.tokenize_count_text(&prompt)?;
        let body = self.scoring_body(&prompt, None);
        let data = self.post_completions_raw(&body)?;
        let all = extract_completions_logprobs(&data);
        let completion_logprobs = if prompt_token_count <= all.len() {
            all[prompt_token_count..].to_vec()
        } else {
            all
        };
        Ok(ScoringResponse {
            completion: completion.to_string(),
            logprobs: completion_logprobs,
            raw: data,
        })
    }

    fn post_score_fill_middle(
        &self,
        messages: &[ChatMessage],
        completion: &str,
    ) -> Result<ScoringResponse> {
        let prompt = render_prompt(messages);
        let body = self.scoring_body(&prompt, Some(completion));
        let data = self.post_completions_raw(&body)?;
        let all = extract_completions_logprobs(&data);
        let completion_logprobs = find_completion_tokens(&all, completion);
        Ok(ScoringResponse {
            completion: completion.to_string(),
            logprobs: completion_logprobs,
            raw: data,
        })
    }

    async fn apost_score_ids(
        &self,
        messages: &[ChatMessage],
        completion: &str,
    ) -> Result<ScoringResponse> {
        let prompt = format!("{}{}", render_prompt(messages), completion);
        let prompt_token_count = self.atokenize_count_ids(&prompt).await?;
        let body = self.scoring_body(&prompt, None);
        let data = self.apost_completions_raw(&body).await?;
        let all = extract_completions_logprobs(&data);
        let completion_logprobs = if prompt_token_count <= all.len() {
            all[prompt_token_count..].to_vec()
        } else {
            all
        };
        Ok(ScoringResponse {
            completion: completion.to_string(),
            logprobs: completion_logprobs,
            raw: data,
        })
    }

    async fn apost_score_count(
        &self,
        messages: &[ChatMessage],
        completion: &str,
    ) -> Result<ScoringResponse> {
        let prompt = format!("{}{}", render_prompt(messages), completion);
        let prompt_token_count = self.atokenize_count_text(&prompt).await?;
        let body = self.scoring_body(&prompt, None);
        let data = self.apost_completions_raw(&body).await?;
        let all = extract_completions_logprobs(&data);
        let completion_logprobs = if prompt_token_count <= all.len() {
            all[prompt_token_count..].to_vec()
        } else {
            all
        };
        Ok(ScoringResponse {
            completion: completion.to_string(),
            logprobs: completion_logprobs,
            raw: data,
        })
    }

    async fn apost_score_fill_middle(
        &self,
        messages: &[ChatMessage],
        completion: &str,
    ) -> Result<ScoringResponse> {
        let prompt = render_prompt(messages);
        let body = self.scoring_body(&prompt, Some(completion));
        let data = self.apost_completions_raw(&body).await?;
        let all = extract_completions_logprobs(&data);
        let completion_logprobs = find_completion_tokens(&all, completion);
        Ok(ScoringResponse {
            completion: completion.to_string(),
            logprobs: completion_logprobs,
            raw: data,
        })
    }

    // ----- tokenize ------------------------------------------------------

    pub(crate) fn post_tokenize(&self, text: &str) -> Result<Vec<Token>> {
        let url = format!("{}/tokenize", self.base_url);
        let body = serde_json::json!({ "model": self.model, "prompt": text });
        let response = self
            .client
            .post(&url)
            .headers(build_headers(&self.api_key))
            .json(&body)
            .send()?;
        response.error_for_status_ref()?;
        let data: Value = response.json()?;
        Ok(parse_tokenize_response(&data))
    }

    pub(crate) async fn apost_tokenize(&self, text: &str) -> Result<Vec<Token>> {
        let url = format!("{}/tokenize", self.base_url);
        let body = serde_json::json!({ "model": self.model, "prompt": text });
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
        Ok(parse_tokenize_response(&data))
    }

    fn tokenize_count_ids(&self, text: &str) -> Result<usize> {
        // vLLM `/tokenize` returns token ids.
        Ok(self.post_tokenize(text).map(|t| t.len()).unwrap_or(0))
    }

    fn tokenize_count_text(&self, text: &str) -> Result<usize> {
        // SGLang `/tokenize` returns token ids (same shape).
        Ok(self.post_tokenize(text).map(|t| t.len()).unwrap_or(0))
    }

    async fn atokenize_count_ids(&self, text: &str) -> Result<usize> {
        Ok(self
            .apost_tokenize(text)
            .await
            .map(|t| t.len())
            .unwrap_or(0))
    }

    async fn atokenize_count_text(&self, text: &str) -> Result<usize> {
        Ok(self
            .apost_tokenize(text)
            .await
            .map(|t| t.len())
            .unwrap_or(0))
    }

    // ----- low-level completions post -----------------------------------

    fn post_completions_raw(&self, body: &Value) -> Result<Value> {
        let url = format!("{}/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .headers(build_headers(&self.api_key))
            .json(body)
            .send()?;
        response.error_for_status_ref()?;
        Ok(response.json()?)
    }

    async fn apost_completions_raw(&self, body: &Value) -> Result<Value> {
        let url = format!("{}/completions", self.base_url);
        let response = self
            .async_client
            .post(&url)
            .headers(build_headers(&self.api_key))
            .json(body)
            .send()
            .await?;
        response
            .error_for_status_ref()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(response.json().await?)
    }
}

/// Extract the per-token logprob list from a `/completions` response.
///
/// Handles both the OpenAI shape (`choices[0].logprobs.content[]`) and the
/// flat vLLM/SGLang/llama.cpp completions shape
/// (`choices[0].logprobs.{tokens,token_logprobs,top_logprobs}`).
pub(crate) fn extract_completions_logprobs(data: &Value) -> Vec<TokenLogprob> {
    let choice = &data["choices"][0];
    let lp = match choice.get("logprobs") {
        Some(lp) => lp,
        None => return Vec::new(),
    };

    // Flat completions shape: { tokens, token_logprobs, top_logprobs }.
    if lp.get("tokens").and_then(|t| t.as_array()).is_some() {
        return parse_flat_completions_logprobs(lp);
    }
    // OpenAI chat-like nested shape: { content: [ ... ] }.
    if let Some(content) = lp.get("content") {
        return parse_token_logprob_array(content);
    }
    // Single-object fallback (some servers return logprobs as the array directly).
    if let Some(arr) = lp.as_array() {
        return parse_token_logprob_array(&Value::Array(arr.clone()));
    }
    Vec::new()
}

fn parse_flat_completions_logprobs(lp: &Value) -> Vec<TokenLogprob> {
    let tokens = lp["tokens"].as_array();
    let token_logprobs = lp["token_logprobs"].as_array();
    let top_logprobs = lp["top_logprobs"].as_array();
    let token_ids = lp["token_ids"].as_array();

    let n = tokens.map(|a| a.len()).unwrap_or(0);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let token = tokens
            .and_then(|a| a.get(i).and_then(|v| v.as_str()))
            .unwrap_or("")
            .to_string();
        let logprob = token_logprobs
            .and_then(|a| a.get(i).and_then(|v| v.as_f64()))
            .unwrap_or(0.0);
        let id = token_ids
            .and_then(|a| a.get(i).and_then(|v| v.as_i64()))
            .unwrap_or(-1);
        let mut top = HashMap::new();
        if let Some(arr) = top_logprobs.and_then(|a| a.get(i).and_then(|v| v.as_array())) {
            for entry in arr {
                if let (Some(t), Some(p)) = (entry["token"].as_str(), entry["logprob"].as_f64()) {
                    top.insert(t.to_string(), p);
                }
            }
        }
        out.push(TokenLogprob {
            token,
            token_id: id,
            logprob,
            top_logprobs: top,
        });
    }
    out
}

/// Heuristically locate the completion tokens within a returned token list.
///
/// Mirrors llama.cpp server behavior: with `suffix` + `echo`, the server
/// returns tokens for prompt + completion; we find the index where the
/// completion tokens begin by matching the first token whose text starts the
/// completion (after trimming).
fn find_completion_tokens(all: &[TokenLogprob], completion: &str) -> Vec<TokenLogprob> {
    let target = completion.trim();
    let start = all
        .iter()
        .position(|tlp| {
            let t = tlp.token.trim();
            !t.is_empty() && target.starts_with(t)
        })
        .unwrap_or(0);
    if start <= all.len() {
        all[start..].to_vec()
    } else {
        Vec::new()
    }
}

/// Parse a `/tokenize` response into [`Token`]s.
///
/// Accepts `{tokens: [id...]}`, `{tokens: [str...]}`, and `{tokens: [{id,...}]}`.
pub(crate) fn parse_tokenize_response(data: &Value) -> Vec<Token> {
    let arr = match data.get("tokens").and_then(|t| t.as_array()) {
        Some(a) => a,
        None => return Vec::new(),
    };
    let mut out = Vec::with_capacity(arr.len());
    for t in arr {
        if let Some(s) = t.as_str() {
            out.push(Token {
                text: s.to_string(),
                id: -1,
            });
        } else if let Some(i) = t.as_i64() {
            out.push(Token {
                text: format!("token_{i}"),
                id: i,
            });
        } else if let Some(obj) = t.as_object() {
            let id = obj.get("id").and_then(|v| v.as_i64()).unwrap_or(-1);
            let text = obj
                .get("token")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| format!("token_{id}"));
            out.push(Token { text, id });
        }
    }
    out
}

/// Builder for [`OpenAICompatCore`].
pub struct OpenAICompatCoreBuilder {
    pub(crate) model: String,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    pub(crate) timeout: Duration,
    pub(crate) max_tokens: u32,
    pub(crate) extra_body: HashMap<String, Value>,
    pub(crate) constraint: Constraint,
    pub(crate) boundary: BoundaryStrategy,
}

impl OpenAICompatCoreBuilder {
    /// Set the API key. Defaults to `"not-needed"`.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = key.into();
        self
    }

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

    /// Build the core.
    pub fn build(self) -> OpenAICompatCore {
        let client = reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .build()
            .expect("failed to build sync client");
        let async_client = reqwest::Client::builder()
            .timeout(self.timeout)
            .build()
            .expect("failed to build async client");
        OpenAICompatCore {
            model: self.model,
            base_url: normalize_base_url(self.base_url),
            api_key: self.api_key,
            timeout: self.timeout,
            max_tokens: self.max_tokens,
            extra_body: self.extra_body,
            constraint: self.constraint,
            boundary: self.boundary,
            client,
            async_client,
        }
    }
}

/// Implement [`LLMBackend`] for a backend type that owns an
/// [`OpenAICompatCore`], delegating every method to the shared core.
///
/// Implemented as a macro (rather than a blanket `impl<T>` over a helper trait)
/// because `#[async_trait]` desugars lifetimes in a way that is incompatible
/// with blanket trait implementations.
#[macro_export]
macro_rules! impl_openai_compat_backend {
    ($backend:ty) => {
        #[async_trait::async_trait]
        impl $crate::backends::base::LLMBackend for $backend {
            fn model(&self) -> &str {
                &self.core.model
            }

            fn base_url(&self) -> &str {
                &self.core.base_url
            }

            fn supports_bare_label_constraint(&self) -> bool {
                true
            }

            fn chat(
                &self,
                messages: &[$crate::backends::base::ChatMessage],
                temperature: f64,
                constrain_labels: Option<&[String]>,
                logprobs: bool,
                top_logprobs: u32,
            ) -> anyhow::Result<$crate::backends::base::ChatResponse> {
                let body = self.core.chat_body(
                    messages,
                    temperature,
                    constrain_labels,
                    logprobs,
                    top_logprobs,
                );
                self.core.post_chat(body)
            }

            fn score(
                &self,
                messages: &[$crate::backends::base::ChatMessage],
                completion: &str,
            ) -> anyhow::Result<$crate::backends::base::ScoringResponse> {
                self.core.post_score(messages, completion)
            }

            fn tokenize(
                &self,
                text: &str,
                context: Option<&str>,
            ) -> anyhow::Result<Vec<$crate::backends::base::Token>> {
                $crate::backends::openai_compat::tokenize_with_context(&self.core, text, context)
            }

            async fn achat(
                &self,
                messages: &[$crate::backends::base::ChatMessage],
                temperature: f64,
                constrain_labels: Option<&[String]>,
                logprobs: bool,
                top_logprobs: u32,
            ) -> anyhow::Result<$crate::backends::base::ChatResponse> {
                let body = self.core.chat_body(
                    messages,
                    temperature,
                    constrain_labels,
                    logprobs,
                    top_logprobs,
                );
                self.core.apost_chat(body).await
            }

            async fn ascore(
                &self,
                messages: &[$crate::backends::base::ChatMessage],
                completion: &str,
            ) -> anyhow::Result<$crate::backends::base::ScoringResponse> {
                self.core.apost_score(messages, completion).await
            }

            async fn atokenize(
                &self,
                text: &str,
                context: Option<&str>,
            ) -> anyhow::Result<Vec<$crate::backends::base::Token>> {
                $crate::backends::openai_compat::tokenize_with_context_async(
                    &self.core, text, context,
                )
                .await
            }
        }
    };
}

/// Shared logic for `tokenize(text, context)`: tokenize `context + text`,
/// then strip the leading context token count.
///
/// For the OpenAI-compatible backends the context token prefix is always
/// prepended to the text before tokenization (no JSON wrapping).
pub(crate) fn tokenize_with_context(
    core: &OpenAICompatCore,
    text: &str,
    context: Option<&str>,
) -> Result<Vec<Token>> {
    match context {
        None => core.post_tokenize(text),
        Some(ctx) => {
            let combined = format!("{ctx}{text}");
            let combined_tokens = core.post_tokenize(&combined)?;
            let ctx_tokens = core.post_tokenize(ctx).map(|t| t.len()).unwrap_or(0);
            Ok(combined_tokens.into_iter().skip(ctx_tokens).collect())
        }
    }
}

pub(crate) async fn tokenize_with_context_async(
    core: &OpenAICompatCore,
    text: &str,
    context: Option<&str>,
) -> Result<Vec<Token>> {
    match context {
        None => core.apost_tokenize(text).await,
        Some(ctx) => {
            let combined = format!("{ctx}{text}");
            let combined_tokens = core.apost_tokenize(&combined).await?;
            let ctx_tokens = core.apost_tokenize(ctx).await.map(|t| t.len()).unwrap_or(0);
            Ok(combined_tokens.into_iter().skip(ctx_tokens).collect())
        }
    }
}
