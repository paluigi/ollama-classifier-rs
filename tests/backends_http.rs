//! HTTP-level backend tests using wiremock.
//!
//! These verify that each backend sends the correct constraint field in its
//! request body and parses responses correctly — the wire-format parity points
//! with the Python implementation.
//!
//! Tests are synchronous (not `#[tokio::test]`) because the backends hold a
//! `reqwest::blocking::Client`, which owns its own tokio runtime and panics if
//! dropped inside an async context. We drive async calls via a runtime created
//! and dropped in a synchronous context.

use std::time::Duration;

use ollama_classifier_rs::backends::base::{ChatMessage, LLMBackend};
use ollama_classifier_rs::backends::{LlamaCppBackend, OllamaBackend, SGLangBackend, VLLMBackend};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Run an async future to completion inside a fresh runtime, so the runtime is
/// dropped before the (blocking-client-bearing) backend. Used for both wiremock
/// setup (`MockServer::start`, `Mock::mount`) and backend calls.
fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    // Reuse a thread-local runtime to avoid creating one per call, which is
    // expensive and can exhaust resources.
    thread_local! {
        static RT: tokio::runtime::Runtime = tokio::runtime::Runtime::new().unwrap();
    }
    RT.with(|rt| rt.block_on(fut))
}

// =========================================================================
// vLLM: structured_outputs.choice constraint
// =========================================================================

#[test]
fn test_vllm_chat_sends_structured_outputs_choice() {
    let server = block_on(MockServer::start());
    block_on(
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer not-needed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "positive"},
                    "logprobs": null
                }]
            })))
            .expect(1)
            .mount(&server),
    );

    let base_url = format!("{}/v1", server.uri());
    let backend = VLLMBackend::with_config("test-model", &base_url);
    let messages = vec![ChatMessage::new("user", "hello")];
    let resp = block_on(backend.achat(
        &messages,
        0.0,
        Some(&["positive".into(), "negative".into()]),
        false,
        0,
    ))
    .unwrap();

    assert_eq!(resp.label, "positive");
    assert!(backend.supports_bare_label_constraint());
}

#[test]
fn test_vllm_chat_body_has_structured_outputs_choice_field() {
    let server = block_on(MockServer::start());
    block_on(
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({
                "structured_outputs": {"choice": ["positive", "negative"]}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"content": "positive"}, "logprobs": null}]
            })))
            .expect(1)
            .named("structured_outputs.choice in body")
            .mount(&server),
    );

    let base_url = format!("{}/v1", server.uri());
    let backend = VLLMBackend::with_config("test-model", &base_url);
    let messages = vec![ChatMessage::new("user", "hi")];
    block_on(backend.achat(
        &messages,
        0.0,
        Some(&["positive".into(), "negative".into()]),
        false,
        0,
    ))
    .unwrap();
}

// =========================================================================
// SGLang: regex constraint
// =========================================================================

#[test]
fn test_sglang_chat_body_has_regex_field() {
    let server = block_on(MockServer::start());
    block_on(
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({
                "regex": "(positive|negative)"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"content": "positive"}, "logprobs": null}]
            })))
            .expect(1)
            .mount(&server),
    );

    let base_url = format!("{}/v1", server.uri());
    let backend = SGLangBackend::with_config("test-model", &base_url);
    let messages = vec![ChatMessage::new("user", "hi")];
    block_on(backend.achat(
        &messages,
        0.0,
        Some(&["positive".into(), "negative".into()]),
        false,
        0,
    ))
    .unwrap();
}

// =========================================================================
// llama.cpp: grammar constraint
// =========================================================================

#[test]
fn test_llamacpp_chat_body_has_grammar_field() {
    let server = block_on(MockServer::start());
    block_on(
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({
                "grammar": "root ::= \"positive\" | \"negative\""
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"content": "positive"}, "logprobs": null}]
            })))
            .expect(1)
            .mount(&server),
    );

    let base_url = format!("{}/v1", server.uri());
    let backend = LlamaCppBackend::with_config("test-model", &base_url);
    let messages = vec![ChatMessage::new("user", "hi")];
    block_on(backend.achat(
        &messages,
        0.0,
        Some(&["positive".into(), "negative".into()]),
        false,
        0,
    ))
    .unwrap();
}

// =========================================================================
// Ollama: native /api/chat with JSON-schema format, logprobs parsing
// =========================================================================

#[test]
fn test_ollama_chat_sends_json_schema_format() {
    let server = block_on(MockServer::start());
    block_on(
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(wiremock::matchers::body_partial_json(json!({
                "format": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": ["positive", "negative"]}
                    },
                    "required": ["label"]
                }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "message": {"role": "assistant", "content": "{\"label\": \"positive\"}"},
                "logprobs": null
            })))
            .expect(1)
            .mount(&server),
    );

    let host = server.uri();
    let backend = OllamaBackend::with_config("llama3.2", &host);
    let messages = vec![ChatMessage::new("user", "hello")];
    let resp = block_on(backend.achat(
        &messages,
        0.0,
        Some(&["positive".into(), "negative".into()]),
        false,
        0,
    ))
    .unwrap();

    // Ollama returns JSON-wrapped content; the label should be extracted.
    assert_eq!(resp.label, "positive");
    assert_eq!(resp.content, "{\"label\": \"positive\"}");
    assert!(!backend.supports_bare_label_constraint());
}

#[test]
fn test_ollama_score_uses_forced_constrained_chat() {
    let server = block_on(MockServer::start());
    block_on(
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(wiremock::matchers::body_partial_json(json!({
                "format": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": ["positive"]}
                    },
                    "required": ["label"]
                },
                "logprobs": 1
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "message": {"role": "assistant", "content": "{\"label\": \"positive\"}"},
                "logprobs": [
                    {"token": "{\"label\": \"", "logprob": -10.0},
                    {"token": "pos", "logprob": -0.5},
                    {"token": "itive", "logprob": -0.3},
                    {"token": "\"}", "logprob": -0.0}
                ]
            })))
            .expect(1)
            .mount(&server),
    );

    let host = server.uri();
    let backend = OllamaBackend::with_config("llama3.2", &host);
    let messages = vec![ChatMessage::new("user", "text")];
    let resp = block_on(backend.ascore(&messages, "positive")).unwrap();

    assert_eq!(resp.completion, "positive");
    // label_token_logprobs should extract the two value tokens
    assert_eq!(resp.logprobs.len(), 2);
    assert_eq!(resp.logprobs[0].token, "pos");
    assert!((resp.logprobs[0].logprob - (-0.5)).abs() < 1e-9);
}

#[test]
fn test_ollama_tokenize_uses_forced_constrained_chat() {
    let server = block_on(MockServer::start());
    block_on(
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(wiremock::matchers::body_partial_json(json!({
                "format": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": ["hello"]}
                    },
                    "required": ["label"]
                },
                "logprobs": 1
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "message": {"role": "assistant", "content": "{\"label\": \"hello\"}"},
                "logprobs": [
                    {"token": "{\"label\": \"", "logprob": -0.0},
                    {"token": "he", "logprob": -0.1},
                    {"token": "llo", "logprob": -0.2},
                    {"token": "\"}", "logprob": -0.0}
                ]
            })))
            .expect(1)
            .mount(&server),
    );

    let host = server.uri();
    let backend = OllamaBackend::with_config("llama3.2", &host);
    let tokens = block_on(backend.atokenize("hello", None)).unwrap();

    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0].text, "he");
    assert_eq!(tokens[1].text, "llo");
}

// =========================================================================
// Shared: response parsing, logprobs flattening
// =========================================================================

#[test]
fn test_chat_response_parses_logprobs_and_flattens_top() {
    let server = block_on(MockServer::start());
    block_on(
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {"content": "cat"},
                    "logprobs": {
                        "content": [{
                            "token": "cat",
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "cat", "logprob": -0.1},
                                {"token": "dog", "logprob": -2.5}
                            ]
                        }]
                    }
                }]
            })))
            .mount(&server),
    );

    let base_url = format!("{}/v1", server.uri());
    let backend = VLLMBackend::with_config("m", &base_url);
    let messages = vec![ChatMessage::new("user", "hi")];
    let resp = block_on(backend.achat(&messages, 0.0, None, true, 5)).unwrap();

    let lps = resp.logprobs.expect("logprobs should be present");
    assert_eq!(lps.len(), 1);
    assert_eq!(lps[0].token, "cat");
    assert!((lps[0].logprob - (-0.1)).abs() < 1e-9);
    // top_logprobs flattened into a map
    assert_eq!(lps[0].top_logprobs.len(), 2);
    assert!((lps[0].top_logprobs["dog"] - (-2.5)).abs() < 1e-9);
}

// =========================================================================
// base_url normalization
// =========================================================================

#[test]
fn test_base_url_trailing_slash_stripped() {
    let backend = VLLMBackend::builder("m", "http://localhost:8000/v1///").build();
    assert_eq!(backend.base_url(), "http://localhost:8000/v1");
}

// =========================================================================
// timeout builder is accepted (smoke test)
// =========================================================================

#[test]
fn test_builder_accepts_timeout() {
    let _backend = VLLMBackend::builder("m", "http://localhost:8000/v1")
        .timeout(Duration::from_secs(30))
        .max_tokens(64)
        .api_key("key")
        .build();

    let _ollama = OllamaBackend::builder("m", "http://localhost:11434")
        .timeout(Duration::from_secs(10))
        .max_tokens(32)
        .build();
}
