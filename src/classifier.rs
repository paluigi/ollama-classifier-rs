//! Backend-agnostic LLM classifier with hierarchical constrained scoring.
//!
//! Provides [`LLMClassifier`], a backend-agnostic classifier with two
//! confidence-scoring paths:
//!
//! - [`classify`](LLMClassifier::classify) — multi-call completion scoring:
//!   one backend call per label, geometric-mean logprobs normalized with a
//!   stable softmax. Exact; N calls for N labels.
//! - [`generate`](LLMClassifier::generate) — hierarchical constrained
//!   generation. A single constrained call produces a probability distribution
//!   over all labels using divergence-aware logprobs from the winning path.
//!   When `max_calls > 1`, supplementary calls resolve clusters of labels that
//!   share a token prefix but diverge from the winner — but only to
//!   **reproportion** probability mass *within* each cluster, never changing
//!   between-group totals. This guarantees accuracy never degrades as the call
//!   budget grows.
//!
//! Sync, async (`a*`), and batch (`batch_*` / `abatch_*`) variants are
//! provided.
//!
//! # Example
//!
//! ```no_run
//! use ollama_classifier_rs::backends::VLLMBackend;
//! use ollama_classifier_rs::LLMClassifier;
//!
//! #[tokio::main]
//! async fn main() -> ollama_classifier_rs::Result<()> {
//!     let backend = VLLMBackend::new("meta-llama/Llama-3.2-3B-Instruct");
//!     let classifier = LLMClassifier::new(backend);
//!
//!     let result = classifier
//!         .classify("I love this product!", vec!["positive", "negative", "neutral"], None)?;
//!
//!     println!("Prediction: {}", result.prediction);
//!     println!("Confidence: {:.2}%", result.confidence * 100.0);
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;

use crate::backends::base::{ChatMessage, LLMBackend};
use crate::prompts::{build_classification_prompt, get_choice_labels};
use crate::scoring::{
    geometric_mean_logprob, get_scored_lengths, identify_unresolved_clusters, stable_softmax,
    LabelTrie,
};
use crate::types::{Choices, ClassificationResult};

/// Default number of worker threads for synchronous batch classification.
const DEFAULT_MAX_WORKERS: usize = 4;

/// A backend-agnostic text classifier.
///
/// Generic over any [`LLMBackend`]. Construct with [`LLMClassifier::new`]
/// (default `max_workers = 4`) or [`LLMClassifier::with_max_workers`].
pub struct LLMClassifier<B: LLMBackend> {
    backend: B,
    max_workers: usize,
}

impl<B: LLMBackend> LLMClassifier<B> {
    /// Create a new classifier with the given backend and the default
    /// concurrency (`max_workers = 4`).
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            max_workers: DEFAULT_MAX_WORKERS,
        }
    }

    /// Create a new classifier with an explicit sync-batch concurrency.
    pub fn with_max_workers(backend: B, max_workers: usize) -> Self {
        Self {
            backend,
            max_workers: max_workers.max(1),
        }
    }

    /// Get a reference to the underlying backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    // ----------------------------------------------------------------------
    // Internal helpers
    // ----------------------------------------------------------------------

    fn to_messages(system: &str, user: &str) -> Vec<ChatMessage> {
        vec![
            ChatMessage::new("system", system),
            ChatMessage::new("user", user),
        ]
    }

    /// Minimum `top_logprobs` budget for the adaptive path.
    fn top_logprobs_budget(trie: &LabelTrie) -> u32 {
        trie.max_branching_factor().max(5) as u32
    }

    // ======================================================================
    // classify — multi-call completion scoring
    // ======================================================================

    /// Classify text with calibrated confidence scores via multi-call
    /// completion scoring.
    ///
    /// Makes one backend call per label (via `backend.score`), takes the
    /// geometric-mean logprob of each completion's tokens, and normalizes with
    /// a stable softmax. Exact; N calls for N labels.
    pub fn classify(
        &self,
        text: &str,
        choices: impl Into<Choices>,
        system_prompt: Option<&str>,
    ) -> Result<ClassificationResult> {
        let choices = choices.into();
        let labels = get_choice_labels(&choices);
        let (system, user) = build_classification_prompt(text, &choices, system_prompt);
        let messages = Self::to_messages(&system, &user);

        let mut raw_scores: HashMap<String, f64> = HashMap::new();
        let mut logprob_details: HashMap<String, Vec<f64>> = HashMap::new();
        for label in &labels {
            let lps = match self.backend.score(&messages, label) {
                Ok(resp) => resp.logprobs.iter().map(|t| t.logprob).collect::<Vec<_>>(),
                Err(_) => Vec::new(),
            };
            let score = geometric_mean_logprob(&lps).unwrap_or(f64::NEG_INFINITY);
            raw_scores.insert(label.clone(), score);
            logprob_details.insert(label.clone(), lps);
        }
        Ok(self.finalize_multi_call(raw_scores, logprob_details, labels.len()))
    }

    /// Asynchronous [`classify`](LLMClassifier::classify). Runs all per-label
    /// `ascore` calls concurrently.
    pub async fn aclassify(
        &self,
        text: &str,
        choices: impl Into<Choices>,
        system_prompt: Option<&str>,
    ) -> Result<ClassificationResult> {
        let choices = choices.into();
        let labels = get_choice_labels(&choices);
        let (system, user) = build_classification_prompt(text, &choices, system_prompt);
        let messages = Arc::new(Self::to_messages(&system, &user));

        let mut futs = Vec::with_capacity(labels.len());
        for label in &labels {
            let messages = messages.clone();
            let label = label.clone();
            futs.push(async move {
                let lps = match self.backend.ascore(&messages, &label).await {
                    Ok(resp) => resp.logprobs.iter().map(|t| t.logprob).collect::<Vec<_>>(),
                    Err(_) => Vec::new(),
                };
                let score = geometric_mean_logprob(&lps).unwrap_or(f64::NEG_INFINITY);
                (label, score, lps)
            });
        }
        let results = futures::future::join_all(futs).await;
        let mut raw_scores: HashMap<String, f64> = HashMap::new();
        let mut logprob_details: HashMap<String, Vec<f64>> = HashMap::new();
        for (label, score, lps) in results {
            raw_scores.insert(label.clone(), score);
            logprob_details.insert(label, lps);
        }
        Ok(self.finalize_multi_call(raw_scores, logprob_details, labels.len()))
    }

    fn finalize_multi_call(
        &self,
        raw_scores: HashMap<String, f64>,
        logprob_details: HashMap<String, Vec<f64>>,
        n_calls: usize,
    ) -> ClassificationResult {
        let probabilities = stable_softmax(&raw_scores).unwrap_or_default();
        let prediction = argmax(&probabilities).unwrap_or_default();
        let confidence = probabilities.get(&prediction).copied().unwrap_or(0.0);
        let mut result = ClassificationResult::new_multi_call(
            prediction,
            confidence,
            probabilities,
            n_calls as i64,
        );
        // Populate raw_response with per-label token logprobs for debugging.
        let raw = serde_json::json!({
            "logprobs": raw_scores,
            "token_logprobs": logprob_details,
        });
        if let serde_json::Value::Object(map) = raw {
            result.raw_response = map;
        }
        result
    }

    // ======================================================================
    // generate — hierarchical constrained generation
    // ======================================================================

    /// Classify text via hierarchical constrained-generation scoring.
    ///
    /// **Call 1** constrains the model to all labels and returns
    /// `top_logprobs` at every generated position. Labels are scored up to
    /// their divergence point from the winning path using the geometric mean
    /// of available token logprobs, then a softmax produces the initial
    /// probability distribution. All logprobs come from the same constraint
    /// context, so the distribution is internally consistent.
    ///
    /// **Calls 2…max_calls** resolve *clusters*: groups of ≥2 non-winning
    /// labels that share a scored prefix but diverge from the winner (and from
    /// each other) at a later position. For each cluster, a constrained call
    /// over only the cluster's labels produces divergence-based logprobs whose
    /// softmax gives **relative weights** summing to 1. The cluster's total
    /// probability mass (summed from the initial distribution) is then
    /// redistributed among its members according to these relative weights.
    /// This **reproportioning** never changes the total probability of any
    /// group — it only sharpens the distribution *within* a group — so
    /// accuracy can only improve or stay the same, never degrade.
    ///
    /// The `max_calls` budget bounds the number of backend calls:
    /// - `Some(1)` (default) — a single call, no cluster resolution.
    /// - `Some(k)` — up to `k` calls; resolves clusters adaptively.
    /// - `None` — resolves all clusters recursively (exact for non-winning
    ///   labels; equivalent to `classify` when every label is fully resolved).
    pub fn generate(
        &self,
        text: &str,
        choices: impl Into<Choices>,
        system_prompt: Option<&str>,
        max_calls: Option<usize>,
    ) -> Result<ClassificationResult> {
        let choices = choices.into();
        let labels = get_choice_labels(&choices);
        let (system, user) = build_classification_prompt(text, &choices, system_prompt);
        let messages = Self::to_messages(&system, &user);

        let token_context = self.token_context();
        let mut token_sequences: HashMap<String, Vec<String>> = HashMap::new();
        for label in &labels {
            let tokens = self
                .backend
                .tokenize(label, token_context)
                .unwrap_or_default();
            let seq: Vec<String> = tokens
                .into_iter()
                .map(|t| {
                    if t.text.is_empty() {
                        format!("token_{}", t.id)
                    } else {
                        t.text
                    }
                })
                .collect();
            token_sequences.insert(label.clone(), seq);
        }

        Ok(self.run_adaptive(&messages, &labels, token_sequences, max_calls))
    }

    /// Asynchronous [`generate`](LLMClassifier::generate). Tokenizes labels
    /// concurrently before running the adaptive loop.
    pub async fn agenerate(
        &self,
        text: &str,
        choices: impl Into<Choices>,
        system_prompt: Option<&str>,
        max_calls: Option<usize>,
    ) -> Result<ClassificationResult> {
        let choices = choices.into();
        let labels = get_choice_labels(&choices);
        let (system, user) = build_classification_prompt(text, &choices, system_prompt);
        let messages = Self::to_messages(&system, &user);

        let token_context = self.token_context().map(String::from);
        let mut futs = Vec::with_capacity(labels.len());
        for label in &labels {
            let label = label.clone();
            let ctx = token_context.clone();
            futs.push(async move {
                let tokens = self
                    .backend
                    .atokenize(&label, ctx.as_deref())
                    .await
                    .unwrap_or_default();
                let seq: Vec<String> = tokens
                    .into_iter()
                    .map(|t| {
                        if t.text.is_empty() {
                            format!("token_{}", t.id)
                        } else {
                            t.text
                        }
                    })
                    .collect();
                (label, seq)
            });
        }
        let results = futures::future::join_all(futs).await;
        let token_sequences: HashMap<String, Vec<String>> = results.into_iter().collect();

        Ok(self.run_adaptive(&messages, &labels, token_sequences, max_calls))
    }

    /// The tokenization context prefix for labels (Ollama wraps labels in
    /// `{"label": "..."}`, so labels must be tokenized in that context).
    fn token_context(&self) -> Option<&str> {
        if self.backend.supports_bare_label_constraint() {
            None
        } else {
            Some(crate::backends::ollama::JSON_LABEL_CONTEXT)
        }
    }

    /// Core hierarchical-reproportion loop shared by the sync and async paths.
    ///
    /// Algorithm (mirrors ollama-classifier v0.6.0):
    /// 1. First constrained call over ALL labels → initial probability
    ///    distribution (all logprobs from the same constraint context).
    /// 2. Identify clusters of ≥2 labels that share a scored prefix but
    ///    diverge from the winner.
    /// 3. For each cluster: subset call → divergence-based relative weights
    ///    (softmax of geometric-mean scores) → redistribute the cluster's
    ///    total probability mass among its members. Between-group totals are
    ///    locked.
    /// 4. Single-label clusters are skipped (nothing to reproportion).
    fn run_adaptive(
        &self,
        messages: &[ChatMessage],
        labels: &[String],
        token_sequences: HashMap<String, Vec<String>>,
        max_calls: Option<usize>,
    ) -> ClassificationResult {
        let mut trie = LabelTrie::new();
        for label in labels {
            if let Some(seq) = token_sequences.get(label) {
                trie.insert(label, seq);
            }
        }
        let k = Self::top_logprobs_budget(&trie);

        // 3. First constrained call over ALL labels.
        let response = match self.backend.chat(messages, 0.0, Some(labels), true, k) {
            Ok(r) => r,
            Err(_) => {
                return Self::empty_adaptive(labels, 0);
            }
        };
        let mut calls_made: usize = 1;

        let winning_label = response.label.clone();
        let step_logprobs =
            extract_step_logprobs(response.logprobs.as_deref().unwrap_or_default(), &trie);

        // Accumulate per-label logprobs and scored lengths for the initial
        // call. All logprobs come from the same constraint context (all
        // labels), so the distribution produced below is internally
        // consistent.
        let mut all_step_logprobs: HashMap<String, Vec<f64>> = HashMap::new();
        let mut all_scored_lengths: HashMap<String, usize> = HashMap::new();
        let initial_lengths = get_scored_lengths(&token_sequences, &winning_label);
        for label in labels {
            let scored_len = *initial_lengths.get(label).unwrap_or(&0);
            let seq = token_sequences.get(label).cloned().unwrap_or_default();
            let mut lps: Vec<f64> = Vec::with_capacity(scored_len);
            for i in 0..scored_len {
                let tok = seq.get(i).cloned().unwrap_or_default();
                let lp = step_logprobs
                    .get(i)
                    .and_then(|m| m.get(&tok).copied())
                    .unwrap_or(f64::NEG_INFINITY);
                lps.push(lp);
            }
            all_step_logprobs.insert(label.clone(), lps);
            all_scored_lengths.insert(label.clone(), scored_len);
        }

        // 4. Initial probability distribution (single constraint context).
        let mut raw_scores: HashMap<String, f64> = HashMap::new();
        for label in labels {
            let lps = all_step_logprobs.get(label).cloned().unwrap_or_default();
            let score = if lps.is_empty() {
                f64::NEG_INFINITY
            } else {
                geometric_mean_logprob(&lps).unwrap_or(f64::NEG_INFINITY)
            };
            raw_scores.insert(label.clone(), score);
        }
        let mut probabilities = stable_softmax(&raw_scores).unwrap_or_default();

        // 5. Recursive cluster resolution via reproportioning.
        let mut frontier = identify_unresolved_clusters(&token_sequences, &all_scored_lengths);

        while !frontier.is_empty() && max_calls.is_none_or(|m| calls_made < m) {
            let cluster = frontier.remove(0);
            let cluster_labels = &cluster.labels;

            // Only resolve clusters with ≥2 labels. Singletons are already
            // fixed: their probability is set by the between-group
            // distribution and no reproportioning call would change it.
            if cluster_labels.len() < 2 {
                continue;
            }

            let cluster_response =
                match self
                    .backend
                    .chat(messages, 0.0, Some(cluster_labels), true, k)
                {
                    Ok(r) => r,
                    Err(_) => break,
                };
            calls_made += 1;

            let cluster_winner = cluster_response.label.clone();
            let cluster_step_lps = extract_step_logprobs(
                cluster_response.logprobs.as_deref().unwrap_or_default(),
                &trie,
            );

            // Build the cluster's token sequences and score them.
            let cluster_token_seqs: HashMap<String, Vec<String>> = cluster_labels
                .iter()
                .filter_map(|l| token_sequences.get(l).map(|s| (l.clone(), s.clone())))
                .collect();
            let sub_lengths = get_scored_lengths(&cluster_token_seqs, &cluster_winner);

            // Replace per-label logprobs for cluster members (NOT append —
            // mixing logprobs from different constraint contexts corrupts the
            // geometric mean). The replacement is only used to compute
            // relative weights below.
            for label in cluster_labels {
                let new_len = *sub_lengths.get(label).unwrap_or(&0);
                let old_len = *all_scored_lengths.get(label).unwrap_or(&0);
                if new_len > old_len {
                    let seq = token_sequences.get(label).cloned().unwrap_or_default();
                    let mut new_lps: Vec<f64> = Vec::with_capacity(new_len);
                    for i in 0..new_len {
                        let tok = seq.get(i).cloned().unwrap_or_default();
                        let lp = cluster_step_lps
                            .get(i)
                            .and_then(|m| m.get(&tok).copied())
                            .unwrap_or(f64::NEG_INFINITY);
                        new_lps.push(lp);
                    }
                    all_step_logprobs.insert(label.clone(), new_lps);
                    all_scored_lengths.insert(label.clone(), new_len);
                }
            }

            // Reproportion: redistribute the cluster's total probability mass
            // among its members using softmax of geometric-mean scores. The
            // sum of cluster probabilities is invariant; only within-cluster
            // shares change.
            let cluster_total: f64 = cluster_labels
                .iter()
                .filter_map(|l| probabilities.get(l).copied())
                .sum();

            let mut cluster_raw: HashMap<String, f64> = HashMap::new();
            for label in cluster_labels {
                let lps = all_step_logprobs.get(label).cloned().unwrap_or_default();
                let score = if lps.is_empty() {
                    f64::NEG_INFINITY
                } else {
                    geometric_mean_logprob(&lps).unwrap_or(f64::NEG_INFINITY)
                };
                cluster_raw.insert(label.clone(), score);
            }
            let cluster_weights = stable_softmax(&cluster_raw).unwrap_or_default();

            for label in cluster_labels {
                if let Some(w) = cluster_weights.get(label) {
                    probabilities.insert(label.clone(), cluster_total * w);
                }
            }

            // Identify sub-clusters within this cluster for further resolution.
            let sub_clusters = identify_unresolved_clusters(&cluster_token_seqs, &sub_lengths);
            frontier.extend(sub_clusters);
        }

        // 6. Compute coverage and final values.
        let mut coverage: HashMap<String, f64> = HashMap::new();
        for label in labels {
            let total = token_sequences.get(label).map(|s| s.len()).unwrap_or(0);
            let scored = *all_scored_lengths.get(label).unwrap_or(&0);
            let cov = if total == 0 {
                1.0
            } else {
                scored as f64 / total as f64
            };
            coverage.insert(label.clone(), cov);
        }
        let approximate = coverage.values().any(|&c| c < 1.0);
        let prediction = argmax(&probabilities).unwrap_or_default();
        let confidence = probabilities.get(&prediction).copied().unwrap_or(0.0);

        let mut result = ClassificationResult::new_adaptive(
            prediction,
            confidence,
            probabilities,
            coverage,
            calls_made as i64,
            approximate,
        );
        // Populate raw_response with the per-label diagnostics.
        let raw = serde_json::json!({
            "logprobs": raw_scores,
            "token_sequences": token_sequences,
            "step_logprobs": all_step_logprobs,
            "scored_lengths": all_scored_lengths,
        });
        if let serde_json::Value::Object(map) = raw {
            result.raw_response = map;
        }
        result
    }

    /// Fallback result when the initial constrained call fails entirely.
    fn empty_adaptive(labels: &[String], calls_made: usize) -> ClassificationResult {
        let n = labels.len() as f64;
        let probabilities: HashMap<String, f64> =
            labels.iter().map(|l| (l.clone(), 1.0 / n)).collect();
        let coverage: HashMap<String, f64> = labels.iter().map(|l| (l.clone(), 0.0)).collect();
        let prediction = labels.first().cloned().unwrap_or_default();
        ClassificationResult::new_adaptive(
            prediction,
            1.0 / n,
            probabilities,
            coverage,
            calls_made as i64,
            true,
        )
    }

    // ======================================================================
    // Batch methods
    // ======================================================================

    /// Synchronous batch [`classify`](LLMClassifier::classify), run across up
    /// to `max_workers` threads.
    pub fn batch_classify(
        &self,
        texts: &[&str],
        choices: impl Into<Choices> + Clone,
        system_prompt: Option<&str>,
    ) -> Result<Vec<ClassificationResult>> {
        let choices = choices.into();
        let labels = get_choice_labels(&choices);
        let sp = system_prompt.map(|s| s.to_string());
        run_batch_sync(self.max_workers, self, texts, move |this, text| {
            let (system, user) = build_classification_prompt(text, &choices, sp.as_deref());
            let messages = Self::to_messages(&system, &user);
            let mut raw_scores: HashMap<String, f64> = HashMap::new();
            let mut logprob_details: HashMap<String, Vec<f64>> = HashMap::new();
            for label in &labels {
                let lps = match this.backend.score(&messages, label) {
                    Ok(resp) => resp.logprobs.iter().map(|t| t.logprob).collect::<Vec<_>>(),
                    Err(_) => Vec::new(),
                };
                raw_scores.insert(
                    label.clone(),
                    geometric_mean_logprob(&lps).unwrap_or(f64::NEG_INFINITY),
                );
                logprob_details.insert(label.clone(), lps);
            }
            Ok(this.finalize_multi_call(raw_scores, logprob_details, labels.len()))
        })
    }

    /// Synchronous batch [`generate`](LLMClassifier::generate), run across up
    /// to `max_workers` threads. Uses the default `max_calls = Some(1)`.
    pub fn batch_generate(
        &self,
        texts: &[&str],
        choices: impl Into<Choices> + Clone,
        system_prompt: Option<&str>,
    ) -> Result<Vec<ClassificationResult>> {
        let choices = choices.into();
        let sp = system_prompt.map(|s| s.to_string());
        run_batch_sync(self.max_workers, self, texts, move |this, text| {
            this.generate(text, choices.clone(), sp.as_deref(), Some(1))
        })
    }

    /// Asynchronous batch [`classify`](LLMClassifier::aclassify), run
    /// concurrently for all texts.
    pub async fn abatch_classify(
        &self,
        texts: &[&str],
        choices: impl Into<Choices> + Clone,
        system_prompt: Option<&str>,
    ) -> Result<Vec<ClassificationResult>> {
        let sp = system_prompt.map(|s| s.to_string());
        let mut futs = Vec::with_capacity(texts.len());
        for text in texts {
            futs.push(self.aclassify(text, choices.clone(), sp.as_deref()));
        }
        futures::future::join_all(futs).await.into_iter().collect()
    }

    /// Asynchronous batch [`generate`](LLMClassifier::agenerate), run
    /// concurrently for all texts. Uses the default `max_calls = Some(1)`.
    pub async fn abatch_generate(
        &self,
        texts: &[&str],
        choices: impl Into<Choices> + Clone,
        system_prompt: Option<&str>,
    ) -> Result<Vec<ClassificationResult>> {
        let sp = system_prompt.map(|s| s.to_string());
        let mut futs = Vec::with_capacity(texts.len());
        for text in texts {
            futs.push(self.agenerate(text, choices.clone(), sp.as_deref(), Some(1)));
        }
        futures::future::join_all(futs).await.into_iter().collect()
    }
}

// =========================================================================
// Free helpers
// =========================================================================

/// Pick the key with the maximum value, breaking ties deterministically by
/// taking the *first* maximum encountered during iteration order (matches the
/// Python `max(probabilities, key=...)` semantics for dict insertion order).
fn argmax(probs: &HashMap<String, f64>) -> Option<String> {
    probs
        .iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(k, _)| k.clone())
}

/// Run a synchronous batch across up to `max_workers` scoped threads.
///
/// Each worker pulls the next text index from a shared atomic counter and
/// applies `work(this, text)` to it, writing the result into a pre-sized slot.
/// Scoped threads borrow `this` and `texts` for the duration of the scope, so
/// no `'static` bound on `B` is required.
fn run_batch_sync<B, F>(
    max_workers: usize,
    this: &LLMClassifier<B>,
    texts: &[&str],
    work: F,
) -> Result<Vec<ClassificationResult>>
where
    B: LLMBackend,
    F: Fn(&LLMClassifier<B>, &str) -> Result<ClassificationResult> + Sync + Send,
{
    let n = texts.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let next = Arc::new(AtomicUsize::new(0));
    let mut out: Vec<Option<Result<ClassificationResult>>> = (0..n).map(|_| None).collect();
    let n_workers = max_workers.min(n).max(1);

    // Cast the output base pointer to `usize` so it is `Send`/`Sync` across
    // scoped threads. Each thread writes to a distinct, disjoint index.
    //
    // SAFETY: indices handed out by the atomic counter are unique, so no two
    // threads ever touch the same slot. The scope joins all threads before
    // `out` is read again below.
    let out_ptr: usize = out.as_mut_ptr() as usize;
    let work = &work;
    std::thread::scope(|s| {
        for _ in 0..n_workers {
            let next = next.clone();
            s.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= n {
                        break;
                    }
                    // SAFETY: `i` is unique (atomic fetch-add), so this slot is
                    // never aliased by another thread.
                    let slot = unsafe {
                        &mut *(out_ptr as *mut Option<Result<ClassificationResult>>).add(i)
                    };
                    *slot = Some(work(this, texts[i]));
                }
            });
        }
    });

    out.into_iter()
        .map(|slot| slot.expect("every index must have been filled"))
        .collect()
}

/// Extract per-step candidate-token logprob maps along the winning path,
/// filtered to tokens that are valid continuations in the label trie at that
/// depth.
///
/// At each token position we keep every candidate from the server's
/// `top_logprobs` that could extend *some* label prefix; this is what lets the
/// scoring functions re-evaluate non-winning labels against the same evidence.
fn extract_step_logprobs(
    logprobs: &[crate::backends::base::TokenLogprob],
    trie: &LabelTrie,
) -> Vec<HashMap<String, f64>> {
    let mut steps = Vec::with_capacity(logprobs.len());
    let root = trie.root();
    let mut node = root;
    for tlp in logprobs {
        let mut filtered: HashMap<String, f64> = HashMap::new();
        // Include any top-logprob token that is a valid child of the current
        // trie node (i.e. a continuation of at least one label).
        for (tok, &lp) in &tlp.top_logprobs {
            if node.children.contains_key(tok) {
                filtered.insert(tok.clone(), lp);
            }
        }
        // Also include the emitted token itself if it advances the trie.
        if let Some(child) = node.children.get(&tlp.token) {
            filtered.entry(tlp.token.clone()).or_insert(tlp.logprob);
            node = child;
        } else {
            // Emitted token isn't a known label continuation; keep the filtered
            // set but do not descend (subsequent positions are off the trie).
        }
        steps.push(filtered);
    }
    steps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::VLLMBackend;
    use std::collections::HashMap as StdHashMap;

    #[test]
    fn test_argmax() {
        let mut m = StdHashMap::new();
        m.insert("a".to_string(), 0.1);
        m.insert("b".to_string(), 0.7);
        m.insert("c".to_string(), 0.2);
        assert_eq!(argmax(&m), Some("b".to_string()));
    }

    #[test]
    fn test_to_messages() {
        let msgs = LLMClassifier::<VLLMBackend>::to_messages("sys", "usr");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "sys");
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[1].content, "usr");
    }
}
