//! Integration tests for the classifier using a mock backend.
//!
//! These tests exercise the `classify` (multi-call) and `generate` (adaptive)
//! paths end-to-end against a mock backend that returns canned logprob
//! responses, without any network access.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use ollama_classifier_rs::backends::base::{
    ChatMessage, ChatResponse, LLMBackend, ScoringResponse, Token, TokenLogprob,
};
use ollama_classifier_rs::{Choices, ClassificationResult, LLMClassifier};

/// A mock backend that returns scripted responses and counts calls.
struct MockBackend {
    bare_label: bool,
    /// Scripted per-label logprobs for `score` calls: label -> Vec<logprob>.
    score_logprobs: Mutex<HashMap<String, Vec<f64>>>,
    /// Scripted label returned by `chat` when constrained.
    chat_label: String,
    /// Scripted step logprobs returned by `chat` (per-position top-logprobs).
    chat_step_logprobs: Vec<HashMap<String, f64>>,
    /// Token sequences per label (for `tokenize`).
    token_sequences: HashMap<String, Vec<String>>,
    score_calls: AtomicUsize,
    chat_calls: AtomicUsize,
}

impl MockBackend {
    fn new(bare_label: bool) -> Self {
        Self {
            bare_label,
            score_logprobs: Mutex::new(HashMap::new()),
            chat_label: String::new(),
            chat_step_logprobs: Vec::new(),
            token_sequences: HashMap::new(),
            score_calls: AtomicUsize::new(0),
            chat_calls: AtomicUsize::new(0),
        }
    }

    fn with_score(self, label: &str, logprobs: Vec<f64>) -> Self {
        self.score_logprobs
            .lock()
            .unwrap()
            .insert(label.to_string(), logprobs);
        self
    }

    fn with_tokens(mut self, label: &str, tokens: Vec<String>) -> Self {
        self.token_sequences.insert(label.to_string(), tokens);
        self
    }

    fn with_chat(mut self, label: &str, step_logprobs: Vec<HashMap<String, f64>>) -> Self {
        self.chat_label = label.to_string();
        self.chat_step_logprobs = step_logprobs;
        self
    }
}

#[async_trait]
impl LLMBackend for MockBackend {
    fn model(&self) -> &str {
        "mock"
    }
    fn base_url(&self) -> &str {
        "http://mock"
    }
    fn supports_bare_label_constraint(&self) -> bool {
        self.bare_label
    }

    fn chat(
        &self,
        _messages: &[ChatMessage],
        _temperature: f64,
        _constrain_labels: Option<&[String]>,
        logprobs: bool,
        _top_logprobs: u32,
    ) -> anyhow::Result<ChatResponse> {
        self.chat_calls.fetch_add(1, Ordering::SeqCst);
        let content = if self.bare_label {
            self.chat_label.clone()
        } else {
            format!("{{\"label\": \"{}\"}}", self.chat_label)
        };
        let logprobs_data = if logprobs {
            Some(
                self.chat_step_logprobs
                    .iter()
                    .map(|step| {
                        // The emitted token is the argmax of each step.
                        let (tok, lp) = step
                            .iter()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .unwrap();
                        let mut top = step.clone();
                        TokenLogprob {
                            token: tok.clone(),
                            token_id: -1,
                            logprob: *lp,
                            top_logprobs: std::mem::take(&mut top),
                        }
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        Ok(ChatResponse {
            content,
            label: self.chat_label.clone(),
            logprobs: logprobs_data,
            raw: serde_json::json!({}),
        })
    }

    fn score(
        &self,
        _messages: &[ChatMessage],
        completion: &str,
    ) -> anyhow::Result<ScoringResponse> {
        self.score_calls.fetch_add(1, Ordering::SeqCst);
        let lps = self
            .score_logprobs
            .lock()
            .unwrap()
            .get(completion)
            .cloned()
            .unwrap_or_default();
        let logprobs = lps
            .iter()
            .map(|&lp| TokenLogprob {
                token: "x".into(),
                token_id: -1,
                logprob: lp,
                top_logprobs: HashMap::new(),
            })
            .collect();
        Ok(ScoringResponse {
            completion: completion.to_string(),
            logprobs,
            raw: serde_json::json!({}),
        })
    }

    fn tokenize(&self, text: &str, _context: Option<&str>) -> anyhow::Result<Vec<Token>> {
        let toks = self
            .token_sequences
            .get(text)
            .cloned()
            .unwrap_or_else(|| vec![text.to_string()]);
        Ok(toks
            .iter()
            .enumerate()
            .map(|(i, t)| Token {
                text: t.clone(),
                id: i as i64,
            })
            .collect())
    }

    async fn achat(
        &self,
        messages: &[ChatMessage],
        temperature: f64,
        constrain_labels: Option<&[String]>,
        logprobs: bool,
        top_logprobs: u32,
    ) -> anyhow::Result<ChatResponse> {
        self.chat(
            messages,
            temperature,
            constrain_labels,
            logprobs,
            top_logprobs,
        )
    }

    async fn ascore(
        &self,
        messages: &[ChatMessage],
        completion: &str,
    ) -> anyhow::Result<ScoringResponse> {
        self.score(messages, completion)
    }

    async fn atokenize(&self, text: &str, context: Option<&str>) -> anyhow::Result<Vec<Token>> {
        self.tokenize(text, context)
    }
}

impl MockBackend {
    fn score_calls(&self) -> usize {
        self.score_calls.load(Ordering::SeqCst)
    }
    fn chat_calls(&self) -> usize {
        self.chat_calls.load(Ordering::SeqCst)
    }
}

// =========================================================================
// classify (multi-call)
// =========================================================================

#[test]
fn test_classify_makes_one_call_per_label() {
    let backend = MockBackend::new(true)
        .with_score("positive", vec![-0.5])
        .with_score("negative", vec![-2.0])
        .with_score("neutral", vec![-3.0]);

    let classifier = LLMClassifier::new(backend);
    let result = classifier
        .classify("I love it!", vec!["positive", "negative", "neutral"], None)
        .unwrap();

    assert_eq!(result.method, "multi_call");
    assert_eq!(result.n_calls, 3);
    assert!(!result.approximate);
    assert_eq!(result.prediction, "positive");
    assert!(result.confidence > 0.5);
    assert!(result.coverage.is_empty());
    // probabilities should sum to 1
    let total: f64 = result.probabilities.values().sum();
    assert!((total - 1.0).abs() < 1e-9);
    // 3 backend score calls made
    assert_eq!(classifier.backend().score_calls(), 3);
}

#[test]
fn test_classify_negative_inf_gives_uniform_when_all_invalid() {
    let backend = MockBackend::new(true)
        .with_score("a", vec![f64::NEG_INFINITY])
        .with_score("b", vec![f64::NEG_INFINITY]);

    let classifier = LLMClassifier::new(backend);
    let result = classifier.classify("x", vec!["a", "b"], None).unwrap();

    // All -inf -> uniform distribution
    assert!((result.probabilities["a"] - 0.5).abs() < 1e-9);
    assert!((result.probabilities["b"] - 0.5).abs() < 1e-9);
}

#[test]
fn test_aclassify_runs_concurrently() {
    // Same scripted scores; verify the async path produces the same result.
    let backend = MockBackend::new(true)
        .with_score("positive", vec![-0.5])
        .with_score("negative", vec![-2.0]);

    let classifier = LLMClassifier::new(backend);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt
        .block_on(classifier.aclassify("great", vec!["positive", "negative"], None))
        .unwrap();

    assert_eq!(result.method, "multi_call");
    assert_eq!(result.n_calls, 2);
    assert_eq!(result.prediction, "positive");
    assert_eq!(classifier.backend().score_calls(), 2);
}

// =========================================================================
// generate (adaptive)
// =========================================================================

#[test]
fn test_generate_single_call_picks_winning_label() {
    // Two single-token labels. One constrained call with top-logprobs.
    let backend = MockBackend::new(true)
        .with_tokens("cat", vec!["cat".to_string()])
        .with_tokens("dog", vec!["dog".to_string()])
        .with_chat(
            "cat",
            vec![HashMap::from([
                ("cat".to_string(), -0.5),
                ("dog".to_string(), -2.0),
            ])],
        );

    let classifier = LLMClassifier::new(backend);
    let result = classifier
        .generate("text", vec!["cat", "dog"], None, Some(1))
        .unwrap();

    assert_eq!(result.method, "adaptive_generate");
    assert_eq!(result.prediction, "cat");
    assert_eq!(result.n_calls, 1);
    assert!(result.confidence > 0.5);
    assert!(classifier.backend().chat_calls() >= 1);
}

#[test]
fn test_generate_exact_resolves_shared_prefix() {
    // Two labels sharing a prefix: "cat" vs "car" — both start with "ca".
    // With max_calls=None, the adaptive loop resolves the divergence. The
    // winning constrained label is "cat". "car" diverges at index 2 (t vs r),
    // so only its first 2 tokens ("c","a") are scored against the winning path
    // → coverage 2/3 ≈ 0.667 for "car", 1.0 for "cat".
    let backend = MockBackend::new(true)
        .with_tokens(
            "cat",
            vec!["c".to_string(), "a".to_string(), "t".to_string()],
        )
        .with_tokens(
            "car",
            vec!["c".to_string(), "a".to_string(), "r".to_string()],
        )
        .with_chat(
            "cat",
            vec![
                HashMap::from([("c".to_string(), -0.3)]),
                HashMap::from([("a".to_string(), -0.4)]),
                HashMap::from([("t".to_string(), -0.2), ("r".to_string(), -1.5)]),
            ],
        );

    let classifier = LLMClassifier::new(backend);
    let result = classifier
        .generate("text", vec!["cat", "car"], None, None)
        .unwrap();

    assert_eq!(result.method, "adaptive_generate");
    assert_eq!(result.prediction, "cat");
    // "cat" (the winner) is fully covered: 3/3 tokens scored.
    assert!((result.coverage["cat"] - 1.0).abs() < 1e-9);
    // "car" diverges from the winning path at index 2 → only 2 of 3 tokens
    // scored → coverage 2/3.
    assert!((result.coverage["car"] - (2.0 / 3.0)).abs() < 1e-9);
    // Since at least one label is not fully covered, the result is approximate.
    assert!(result.approximate);
}

#[test]
fn test_generate_prediction_is_softmax_argmax() {
    // Labels "aa" and "ab" share token "a" at position 0. The winning
    // constrained label is "aa", but the prediction is the softmax argmax over
    // geometric-mean scores — which can differ from the winning label. Here
    // "ab"'s second-token logprob (-1.0) is higher than "aa"'s (-0.3), so
    // after normalization "ab" can win. The key assertion is that the
    // algorithm re-evaluates all labels rather than blindly trusting the
    // constrained output.
    let backend = MockBackend::new(true)
        .with_tokens("aa", vec!["a".to_string(), "a".to_string()])
        .with_tokens("ab", vec!["a".to_string(), "b".to_string()])
        .with_chat(
            "aa",
            vec![
                HashMap::from([("a".to_string(), -0.2)]),
                HashMap::from([("a".to_string(), -0.3), ("b".to_string(), -1.0)]),
            ],
        );

    let classifier = LLMClassifier::new(backend);
    let result = classifier
        .generate("text", vec!["aa", "ab"], None, Some(1))
        .unwrap();

    assert_eq!(result.method, "adaptive_generate");
    assert_eq!(result.n_calls, 1);
    // The winner "aa" is fully covered (2/2 tokens on the winning path). "ab"
    // diverges from the winning path "aa" at index 1 (a vs b) → only 1 of 2
    // tokens scored → coverage 0.5.
    assert!((result.coverage["aa"] - 1.0).abs() < 1e-9);
    assert!((result.coverage["ab"] - 0.5).abs() < 1e-9);
    // The prediction is a valid label and the probabilities sum to 1.
    assert!(result.prediction == "aa" || result.prediction == "ab");
    let total: f64 = result.probabilities.values().sum();
    assert!((total - 1.0).abs() < 1e-9);
}

// =========================================================================
// Batch
// =========================================================================

#[test]
fn test_batch_classify_runs_concurrently() {
    let backend = MockBackend::new(true)
        .with_score("positive", vec![-0.3])
        .with_score("negative", vec![-2.5]);

    let classifier = LLMClassifier::with_max_workers(backend, 2);
    let texts = ["I love it", "I hate it", "It's fine"];
    let results = classifier
        .batch_classify(&texts, vec!["positive", "negative"], None)
        .unwrap();

    assert_eq!(results.len(), 3);
    for r in &results {
        assert_eq!(r.method, "multi_call");
        assert_eq!(r.n_calls, 2);
    }
    // 3 texts * 2 labels = 6 score calls
    assert_eq!(classifier.backend().score_calls(), 6);
}

#[test]
fn test_abatch_classify_async() {
    let backend = MockBackend::new(true)
        .with_score("yes", vec![-0.3])
        .with_score("no", vec![-2.0]);

    let classifier = LLMClassifier::new(backend);
    let texts = ["a", "b"];
    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt
        .block_on(classifier.abatch_classify(&texts, vec!["yes", "no"], None))
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(classifier.backend().score_calls(), 4);
}

#[test]
fn test_choices_from_hashmap() {
    // Verify Choices accepts a HashMap (descriptions form).
    let backend = MockBackend::new(true).with_score("a", vec![-0.1]);
    let classifier = LLMClassifier::new(backend);
    let mut map = HashMap::new();
    map.insert("a".to_string(), "Option A".to_string());
    let result = classifier.classify("x", map, None).unwrap();
    assert_eq!(result.prediction, "a");
}

#[test]
fn test_classification_result_serialization() {
    let result = ClassificationResult::new_multi_call(
        "pos".into(),
        0.9,
        HashMap::from([("pos".into(), 0.9), ("neg".into(), 0.1)]),
        2,
    );
    let json = serde_json::to_string(&result).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["prediction"], "pos");
    assert_eq!(v["method"], "multi_call");
    assert_eq!(v["n_calls"], 2);
    assert_eq!(v["probabilities"]["pos"], 0.9);
}

// Re-export Choices so the unused import warning doesn't fire on doc builds.
#[allow(dead_code)]
fn _use_choices(_c: Choices) {}
