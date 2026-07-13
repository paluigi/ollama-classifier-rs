//! Shared core for OpenAI-compatible inference backends.
//!
//! vLLM, SGLang, and llama.cpp all expose the OpenAI-compatible
//! `/chat/completions`, `/completions`, and `/tokenize` endpoints. They differ
//! only in (a) the field used to constrain output to a label set and (b) how
//! completion scoring works. Those differences are captured by [`Constraint`]
//! and [`BoundaryStrategy`]; the common HTTP plumbing lives here.
//!
//! ## Scoring approaches
//!
//! - **Echo/prefill** (vLLM, SGLang): `/v1/completions` with `echo=true` to
//!   recover the model's genuine per-token logprobs. The `/tokenize` endpoint
//!   pinpoints the prompt/completion boundary.
//! - **Forced constrained generation** (llama.cpp): forces the completion as
//!   the only valid label via grammar constraint and reads back the model's
//!   genuine per-token logprobs. llama.cpp does not support `echo=true`.
//!
//! ## Tokenization approach
//!
//! All three backends use empirical **forced constrained generation** —
//! forcing the label as the only valid choice in a `chat()` call and reading
//! back the emitted value tokens. This is necessary because standalone BPE
//! tokenization produces different token boundaries than the model emits under
//! constraint guidance. Results are memoized per label.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;
use serde_json::Value;

use super::base::{
    build_chat_body, build_headers, normalize_base_url, parse_chat_response, render_prompt,
    ChatMessage, ChatResponse, ScoringResponse, Token, TokenLogprob,
};

/// End-of-sequence / special tokens to filter from constrained responses.
///
/// Covers Llama-3, Phi, and Qwen EOS markers.
pub const SPECIAL_TOKENS: &[&str] = &[
    "<|im_end|>",
    "<|endoftext|>",
    "</s>",
    "<|end_of_turn|>",
    "<|eot_id|>",
    "<|end|>",
    "<|eom_id|>",
];

/// Filter out special / end-of-sequence tokens from a logprobs list.
///
/// For bare-label backends, the constraint guarantees only label text is
/// generated, so we just need to remove special/EOS tokens and empty strings.
pub fn filter_special_tokens(logprobs: &[TokenLogprob]) -> Vec<TokenLogprob> {
    logprobs
        .iter()
        .filter(|lp| {
            let tok = lp.token.trim();
            !tok.is_empty() && !SPECIAL_TOKENS.contains(&lp.token.as_str())
        })
        .cloned()
        .collect()
}

/// How an OpenAI-compatible backend constrains output to a label set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Constraint {
    /// vLLM: `structured_outputs.choice` = labels (vLLM v0.12.0+, replaces
    /// the deprecated `guided_choice`).
    StructuredOutputsChoice,
    /// SGLang: `regex` = `(label1|label2|...)`.
    Regex,
    /// llama.cpp: `grammar` = GBNF rule `root ::= "a" | "b" | ...`.
    Grammar,
}

impl Constraint {
    /// Apply this constraint to an OpenAI-compatible request body.
    pub fn apply(&self, body: &mut Value, labels: &[String]) {
        match self {
            Constraint::StructuredOutputsChoice => {
                body["structured_outputs"] = serde_json::json!({
                    "choice": labels
                });
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

/// How completion scoring works for this backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryStrategy {
    /// vLLM: tokenize the prompt by id count; use echo/prefill.
    Ids,
    /// SGLang: tokenize the prompt by text count; use echo/prefill.
    Count,
    /// llama.cpp: forced constrained generation (no echo support).
    Forced,
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

    // ----- scoring -------------------------------------------------------

    /// Synchronous scoring.
    pub(crate) fn post_score(
        &self,
        messages: &[ChatMessage],
        completion: &str,
    ) -> Result<ScoringResponse> {
        match self.boundary {
            BoundaryStrategy::Ids => self.post_score_echo(messages, completion, true),
            BoundaryStrategy::Count => self.post_score_echo(messages, completion, false),
            BoundaryStrategy::Forced => self.post_score_forced(messages, completion),
        }
    }

    /// Asynchronous scoring.
    pub(crate) async fn apost_score(
        &self,
        messages: &[ChatMessage],
        completion: &str,
    ) -> Result<ScoringResponse> {
        match self.boundary {
            BoundaryStrategy::Ids => self.apost_score_echo(messages, completion, true).await,
            BoundaryStrategy::Count => self.apost_score_echo(messages, completion, false).await,
            BoundaryStrategy::Forced => self.apost_score_forced(messages, completion).await,
        }
    }

    /// Echo/prefill scoring (vLLM, SGLang).
    ///
    /// Uses `/v1/completions` with `echo=true` to recover the model's genuine
    /// per-token logprobs for the label as an unexpected continuation of the
    /// prompt. The `/tokenize` endpoint pinpoints the label-token boundary.
    fn post_score_echo(
        &self,
        messages: &[ChatMessage],
        completion: &str,
        use_ids: bool,
    ) -> Result<ScoringResponse> {
        let prompt = render_prompt(messages);
        let prompt_with_completion = format!("{prompt}{completion}");

        let prompt_len = self.tokenize_count(&prompt)?;
        let total_len = self.tokenize_count(&prompt_with_completion)?;

        let url = format!("{}/completions", self.base_url);
        let mut body = serde_json::json!({
            "model": self.model,
            "prompt": prompt_with_completion,
            "echo": true,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 1,
        });
        for (k, v) in &self.extra_body {
            body[k] = v.clone();
        }

        let response = self
            .client
            .post(&url)
            .headers(build_headers(&self.api_key))
            .json(&body)
            .send()?;
        response.error_for_status_ref()?;
        let data: Value = response.json()?;

        let all = extract_completions_logprobs(&data);
        let _ = use_ids; // both Ids and Count use the same slicing logic
        let completion_lps = if total_len > prompt_len && total_len <= all.len() {
            all[prompt_len..total_len].to_vec()
        } else if prompt_len < all.len() {
            all[prompt_len..].to_vec()
        } else {
            Vec::new()
        };

        if completion_lps.is_empty() {
            anyhow::bail!("score({completion:?}): echo returned no label tokens");
        }

        Ok(ScoringResponse {
            completion: completion.to_string(),
            logprobs: completion_lps,
            raw: data,
        })
    }

    /// Forced constrained generation scoring (llama.cpp).
    ///
    /// Forces `completion` as the only valid choice via the backend's
    /// constraint mechanism and reads back the model's genuine per-token
    /// logprobs (teacher forcing, pre-mask).
    fn post_score_forced(
        &self,
        messages: &[ChatMessage],
        completion: &str,
    ) -> Result<ScoringResponse> {
        let mut body = build_chat_body(
            &self.model,
            messages,
            0.0,
            true,
            1,
            self.max_tokens,
            &self.extra_body,
        );
        let labels = vec![completion.to_string()];
        self.constraint.apply(&mut body, &labels);

        let response = self.post_chat(body)?;
        let lps = filter_special_tokens(response.logprobs.as_deref().unwrap_or_default());

        if lps.is_empty() {
            anyhow::bail!("score({completion:?}): forced generation returned no value tokens");
        }

        Ok(ScoringResponse {
            completion: completion.to_string(),
            logprobs: lps,
            raw: response.raw,
        })
    }

    async fn apost_score_echo(
        &self,
        messages: &[ChatMessage],
        completion: &str,
        use_ids: bool,
    ) -> Result<ScoringResponse> {
        let prompt = render_prompt(messages);
        let prompt_with_completion = format!("{prompt}{completion}");

        let prompt_len = self.atokenize_count(&prompt).await?;
        let total_len = self.atokenize_count(&prompt_with_completion).await?;

        let url = format!("{}/completions", self.base_url);
        let mut body = serde_json::json!({
            "model": self.model,
            "prompt": prompt_with_completion,
            "echo": true,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 1,
        });
        for (k, v) in &self.extra_body {
            body[k] = v.clone();
        }

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

        let all = extract_completions_logprobs(&data);
        let _ = use_ids;
        let completion_lps = if total_len > prompt_len && total_len <= all.len() {
            all[prompt_len..total_len].to_vec()
        } else if prompt_len < all.len() {
            all[prompt_len..].to_vec()
        } else {
            Vec::new()
        };

        if completion_lps.is_empty() {
            anyhow::bail!("ascore({completion:?}): echo returned no label tokens");
        }

        Ok(ScoringResponse {
            completion: completion.to_string(),
            logprobs: completion_lps,
            raw: data,
        })
    }

    async fn apost_score_forced(
        &self,
        messages: &[ChatMessage],
        completion: &str,
    ) -> Result<ScoringResponse> {
        let mut body = build_chat_body(
            &self.model,
            messages,
            0.0,
            true,
            1,
            self.max_tokens,
            &self.extra_body,
        );
        let labels = vec![completion.to_string()];
        self.constraint.apply(&mut body, &labels);

        let response = self.apost_chat(body).await?;
        let lps = filter_special_tokens(response.logprobs.as_deref().unwrap_or_default());

        if lps.is_empty() {
            anyhow::bail!("ascore({completion:?}): forced generation returned no value tokens");
        }

        Ok(ScoringResponse {
            completion: completion.to_string(),
            logprobs: lps,
            raw: response.raw,
        })
    }

    // ----- tokenize ------------------------------------------------------

    /// Count tokens via `/tokenize` endpoint (server base URL without `/v1`).
    fn tokenize_count(&self, text: &str) -> Result<usize> {
        let surl = server_url(&self.base_url);
        let url = format!("{surl}/tokenize");
        let body = serde_json::json!({ "model": self.model, "prompt": text });
        let response = self
            .client
            .post(&url)
            .headers(build_headers(&self.api_key))
            .json(&body)
            .send()?;
        response.error_for_status_ref()?;
        let data: Value = response.json()?;
        Ok(data
            .get("tokens")
            .and_then(|t| t.as_array())
            .map(|a| a.len())
            .unwrap_or(0))
    }

    async fn atokenize_count(&self, text: &str) -> Result<usize> {
        let surl = server_url(&self.base_url);
        let url = format!("{surl}/tokenize");
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
        Ok(data
            .get("tokens")
            .and_then(|t| t.as_array())
            .map(|a| a.len())
            .unwrap_or(0))
    }
}

/// Strip the `/v1` suffix to get the server base URL (for `/tokenize`).
pub(crate) fn server_url(base_url: &str) -> String {
    let url = base_url.trim_end_matches('/');
    url.strip_suffix("/v1").unwrap_or(url).to_string()
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
        return super::base::parse_token_logprob_array(content);
    }
    // Single-object fallback (some servers return logprobs as the array directly).
    if let Some(arr) = lp.as_array() {
        return super::base::parse_token_logprob_array(&Value::Array(arr.clone()));
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
                _context: Option<&str>,
            ) -> anyhow::Result<Vec<$crate::backends::base::Token>> {
                $crate::backends::openai_compat::forced_tokenize(&self.core, text)
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
                _context: Option<&str>,
            ) -> anyhow::Result<Vec<$crate::backends::base::Token>> {
                $crate::backends::openai_compat::forced_tokenize_async(&self.core, text).await
            }
        }
    };
}

/// Tokenize text via empirical forced constrained generation.
///
/// Forces `text` as the only valid label in a constrained `chat()` call and
/// reads back the emitted value tokens. This is necessary because standalone
/// BPE tokenization produces different token boundaries than the model emits
/// under constraint guidance.
pub(crate) fn forced_tokenize(core: &OpenAICompatCore, text: &str) -> Result<Vec<Token>> {
    let messages = vec![ChatMessage::new("user", text)];
    let labels = vec![text.to_string()];
    let body = core.chat_body(&messages, 0.0, Some(&labels), true, 1);
    let response = core.post_chat(body)?;
    let lps = filter_special_tokens(response.logprobs.as_deref().unwrap_or_default());

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

    Ok(tokens)
}

pub(crate) async fn forced_tokenize_async(
    core: &OpenAICompatCore,
    text: &str,
) -> Result<Vec<Token>> {
    let messages = vec![ChatMessage::new("user", text)];
    let labels = vec![text.to_string()];
    let body = core.chat_body(&messages, 0.0, Some(&labels), true, 1);
    let response = core.apost_chat(body).await?;
    let lps = filter_special_tokens(response.logprobs.as_deref().unwrap_or_default());

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

    Ok(tokens)
}
