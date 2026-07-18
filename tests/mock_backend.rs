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
///
/// Supports two chat-response modes:
/// - **Fixed mode** (via `with_chat`): always returns the same winner and
///   step-logprob map regardless of `constrain_labels`. Used by the original
///   v0.5.0 tests.
/// - **Map mode** (via `with_chat_map`): looks up the winner as
///   `constrain_labels[0]` and returns the per-winner step-logprob map from
///   `chat_step_logprobs_map`. Mirrors the Python `MockBackend` and lets the
///   reproportion regression tests exercise the cluster-resolution path.
struct MockBackend {
    bare_label: bool,
    /// Scripted per-label logprobs for `score` calls: label -> Vec<logprob>.
    score_logprobs: Mutex<HashMap<String, Vec<f64>>>,
    /// Fixed-mode: scripted label returned by `chat`.
    chat_label: String,
    /// Fixed-mode: scripted step logprobs returned by `chat`.
    chat_step_logprobs: Vec<HashMap<String, f64>>,
    /// Map-mode: per-winner step logprob maps. When non-empty, `chat` returns
    /// `constrain_labels[0]` as the winner and looks up its step logprobs here.
    chat_step_logprobs_map: HashMap<String, Vec<HashMap<String, f64>>>,
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
            chat_step_logprobs_map: HashMap::new(),
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

    /// Fixed-mode chat scripting: always returns `label` as the winner with
    /// the given per-position top-logprob map.
    fn with_chat(mut self, label: &str, step_logprobs: Vec<HashMap<String, f64>>) -> Self {
        self.chat_label = label.to_string();
        self.chat_step_logprobs = step_logprobs;
        self
    }

    /// Map-mode chat scripting: the winner is `constrain_labels[0]` and its
    /// step logprobs are looked up in this map by winner name.
    fn with_chat_map(mut self, map: HashMap<String, Vec<HashMap<String, f64>>>) -> Self {
        self.chat_step_logprobs_map = map;
        self
    }

    /// Resolve the winner and step logprobs for a `chat` call.
    ///
    /// In map-mode the winner is `constrain_labels[0]` (matching the Python
    /// `MockBackend`); in fixed-mode the winner is the scripted `chat_label`.
    fn resolve_chat(
        &self,
        constrain_labels: Option<&[String]>,
    ) -> (String, Vec<HashMap<String, f64>>) {
        if !self.chat_step_logprobs_map.is_empty() {
            // Map mode: winner = first constrained label (or first known key).
            let winner = constrain_labels
                .and_then(|l| l.first().cloned())
                .or_else(|| self.chat_step_logprobs_map.keys().next().cloned())
                .unwrap_or_default();
            let step_lps = self
                .chat_step_logprobs_map
                .get(&winner)
                .cloned()
                .unwrap_or_default();
            (winner, step_lps)
        } else {
            // Fixed mode.
            (self.chat_label.clone(), self.chat_step_logprobs.clone())
        }
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
        constrain_labels: Option<&[String]>,
        logprobs: bool,
        _top_logprobs: u32,
    ) -> anyhow::Result<ChatResponse> {
        self.chat_calls.fetch_add(1, Ordering::SeqCst);
        let (winner, step_logprobs) = self.resolve_chat(constrain_labels);
        let content = if self.bare_label {
            winner.clone()
        } else {
            format!("{{\"label\": \"{}\"}}", winner)
        };
        let logprobs_data = if logprobs {
            Some(
                step_logprobs
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
            label: winner,
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
    // winning constrained label is "cat". "car" diverges at index 2 (t vs r).
    // With the v0.6.0 d+1 scoring, "car" is scored up to and including the
    // divergence position → 3/3 tokens scored → coverage 1.0.
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
    // "car" diverges at index 2 but the divergence position is scored too
    // (d+1=3, capped at len 3) → coverage 3/3 = 1.0.
    assert!((result.coverage["car"] - 1.0).abs() < 1e-9);
    // Both labels fully covered → not approximate.
    assert!(!result.approximate);
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
    //
    // With v0.6.0 d+1 scoring, both labels are fully covered (divergence at
    // index 1, d+1=2 = full length).
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
    // Both labels fully covered (d+1 scoring).
    assert!((result.coverage["aa"] - 1.0).abs() < 1e-9);
    assert!((result.coverage["ab"] - 1.0).abs() < 1e-9);
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

// =========================================================================
// TestMaxCallsMonotonicity — regression tests for hierarchical reproportion.
// Ported from ollama-classifier v0.6.0 tests/test_classifier.py.
//
// The original cluster-resolution code mixed logprobs from different
// constraint contexts into a single geometric mean, which could DECREASE
// accuracy as max_calls increased. The fix uses reproportioning:
// supplementary calls only redistribute probability mass *within* a cluster,
// never changing between-group totals.
// =========================================================================

/// Shared fixture for the monotonicity tests.
///
/// Scenario: 3 labels with a real multi-label cluster.
///   A = [s, a]               (2 tokens)
///   B = [s, b1, b2, b3]      (4 tokens, diverges from A at pos 2)
///   C = [s, b1, c2, c3]      (4 tokens, shares [s, b1] with B at pos 1-2)
///
/// After the first call (winner A), B and C share the scored prefix [s, b1]
/// → they form a multi-label cluster → a reproportioning call is made.
fn monotonicity_backend() -> MockBackend {
    let token_sequences: HashMap<String, Vec<String>> = HashMap::from([
        ("A".to_string(), vec!["s".to_string(), "a".to_string()]),
        (
            "B".to_string(),
            vec![
                "s".to_string(),
                "b1".to_string(),
                "b2".to_string(),
                "b3".to_string(),
            ],
        ),
        (
            "C".to_string(),
            vec![
                "s".to_string(),
                "b1".to_string(),
                "c2".to_string(),
                "c3".to_string(),
            ],
        ),
    ]);
    let step_logprobs_map: HashMap<String, Vec<HashMap<String, f64>>> = HashMap::from([
        // Winner A (3-way call): B and C diverge at pos 2.
        (
            "A".to_string(),
            vec![
                HashMap::from([("s".to_string(), -0.2)]),
                HashMap::from([("a".to_string(), -0.1), ("b1".to_string(), -0.8)]),
            ],
        ),
        // Winner B (subset call on {B, C}): fully resolves B.
        (
            "B".to_string(),
            vec![
                HashMap::from([("s".to_string(), -0.2)]),
                HashMap::from([("b1".to_string(), -0.8)]),
                HashMap::from([("b2".to_string(), -0.3)]),
                HashMap::from([("b3".to_string(), -0.3)]),
            ],
        ),
        // Winner C (subset call on {C}): fully resolves C.
        (
            "C".to_string(),
            vec![
                HashMap::from([("s".to_string(), -0.2)]),
                HashMap::from([("b1".to_string(), -0.8)]),
                HashMap::from([("c2".to_string(), -2.0)]),
                HashMap::from([("c3".to_string(), -2.0)]),
            ],
        ),
    ]);
    MockBackend::new(true)
        .with_tokens("A", token_sequences["A"].clone())
        .with_tokens("B", token_sequences["B"].clone())
        .with_tokens("C", token_sequences["C"].clone())
        .with_chat_map(step_logprobs_map)
}

#[test]
fn test_max_calls_does_not_flip_prediction() {
    // generate(max_calls=None) must not produce a worse prediction than
    // generate(max_calls=1). The model's true preference is A > B > C, and
    // greedy constrained generation picks A. Reproportioning must not inflate
    // B's probability above A's.
    for max_calls in [Some(1usize), Some(2), Some(3), None] {
        let classifier = LLMClassifier::new(monotonicity_backend());
        let result = classifier
            .generate("test", vec!["A", "B", "C"], None, max_calls)
            .unwrap();
        assert_eq!(
            result.prediction, "A",
            "max_calls={:?}: expected 'A', got '{}'",
            max_calls, result.prediction
        );
        // Probabilities must always sum to 1.0.
        let total: f64 = result.probabilities.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "max_calls={:?}: probabilities must sum to 1.0 (got {})",
            max_calls,
            total
        );
    }
}

#[test]
fn test_reproportion_preserves_group_mass() {
    // The sum of cluster probabilities is invariant under reproportioning.
    // Concretely: A's probability must not decrease when max_calls grows, and
    // the total P(B)+P(C) must be preserved.
    let clf1 = LLMClassifier::new(monotonicity_backend());
    let r1 = clf1
        .generate("test", vec!["A", "B", "C"], None, Some(1))
        .unwrap();

    let clf2 = LLMClassifier::new(monotonicity_backend());
    let r2 = clf2
        .generate("test", vec!["A", "B", "C"], None, None)
        .unwrap();

    // Both distributions sum to 1.0.
    let total1: f64 = r1.probabilities.values().sum();
    let total2: f64 = r2.probabilities.values().sum();
    assert!((total1 - 1.0).abs() < 1e-10);
    assert!((total2 - 1.0).abs() < 1e-10);

    // A's probability must not decrease under full resolution.
    assert!(
        r2.probabilities["A"] >= r1.probabilities["A"] - 1e-10,
        "A's probability decreased: mc=1={}, mc=None={}",
        r1.probabilities["A"],
        r2.probabilities["A"]
    );

    // Between-group mass P(B)+P(C) is preserved by reproportioning.
    let group1 = r1.probabilities["B"] + r1.probabilities["C"];
    let group2 = r2.probabilities["B"] + r2.probabilities["C"];
    assert!(
        (group2 - group1).abs() < 1e-9,
        "P(B)+P(C) changed: mc=1={}, mc=None={}",
        group1,
        group2
    );
}

#[test]
fn test_single_token_labels_need_no_resolution_calls() {
    // When all labels are single-token, max_calls has no effect — there are
    // no clusters to resolve.
    for max_calls in [Some(1usize), Some(5), None] {
        let classifier = LLMClassifier::new(monotonicity_backend_for_single_token());
        let result = classifier
            .generate(
                "test",
                vec!["positive", "negative", "neutral"],
                None,
                max_calls,
            )
            .unwrap();
        assert_eq!(result.prediction, "positive");
        assert_eq!(result.n_calls, 1, "no resolution calls needed");
        assert!(!result.approximate);
    }
}

/// Helper: a fresh backend for the single-token monotonicity scenario.
fn monotonicity_backend_for_single_token() -> MockBackend {
    let step_logprobs_map: HashMap<String, Vec<HashMap<String, f64>>> = HashMap::from([(
        "positive".to_string(),
        vec![HashMap::from([
            ("positive".to_string(), -0.3),
            ("negative".to_string(), -1.5),
            ("neutral".to_string(), -2.8),
        ])],
    )]);
    MockBackend::new(true)
        .with_tokens("positive", vec!["positive".to_string()])
        .with_tokens("negative", vec!["negative".to_string()])
        .with_tokens("neutral", vec!["neutral".to_string()])
        .with_chat_map(step_logprobs_map)
}

// Re-export Choices so the unused import warning doesn't fire on doc builds.
#[allow(dead_code)]
fn _use_choices(_c: Choices) {}
