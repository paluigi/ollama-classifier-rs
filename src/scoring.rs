//! Probability and scoring utilities for classification.
//!
//! All length normalization uses geometric mean (not arithmetic mean),
//! applied consistently across both `classify` (multi-call) and `generate`
//! (adaptive) paths. This eliminates the token-count concentration bias that
//! affects longer labels.
//!
//! Mirrors the Python `ollama_classifier.scoring` module function-for-function.

use std::collections::HashMap;

/// Compute the geometric-mean logprob of a token-logprob sequence.
///
/// Filters out `-inf` values; returns the arithmetic mean of the remaining
/// logprobs (`sum(valid) / len(valid)`). Returns `-inf` when the filtered
/// sequence is empty.
///
/// Returns an error on an empty input (matches Python's `ValueError`).
pub fn geometric_mean_logprob(logprobs: &[f64]) -> anyhow::Result<f64> {
    if logprobs.is_empty() {
        anyhow::bail!("Cannot compute geometric mean of empty sequence.");
    }
    let valid: Vec<f64> = logprobs
        .iter()
        .filter(|&&lp| lp > f64::NEG_INFINITY)
        .copied()
        .collect();
    if valid.is_empty() {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(valid.iter().sum::<f64>() / valid.len() as f64)
}

/// Numerically stable softmax over a map of label → logprob.
///
/// Subtracts the maximum *valid* logprob, exponentiates (`-inf` → 0), then
/// normalizes. Returns a uniform `1/n` distribution when there are no valid
/// entries or when the total underflows to zero.
///
/// Returns an error on an empty input (matches Python's `ValueError`).
pub fn stable_softmax(logprobs: &HashMap<String, f64>) -> anyhow::Result<HashMap<String, f64>> {
    if logprobs.is_empty() {
        anyhow::bail!("Cannot compute softmax of empty dict.");
    }
    let n = logprobs.len() as f64;
    let uniform = || logprobs.keys().map(|k| (k.clone(), 1.0 / n)).collect();

    let max_lp = logprobs
        .values()
        .filter(|&&v| v > f64::NEG_INFINITY)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max_lp == f64::NEG_INFINITY {
        return Ok(uniform());
    }

    let mut exp_vals = HashMap::with_capacity(logprobs.len());
    let mut total = 0.0f64;
    for (key, &val) in logprobs {
        let e = if val > f64::NEG_INFINITY {
            (val - max_lp).exp()
        } else {
            0.0
        };
        exp_vals.insert(key.clone(), e);
        total += e;
    }
    if total == 0.0 {
        return Ok(uniform());
    }
    Ok(exp_vals.into_iter().map(|(k, v)| (k, v / total)).collect())
}

/// Index of the first token where `label_tokens` diverges from `winning_tokens`.
///
/// If the two sequences are identical over their overlapping prefix, returns
/// `min(label_tokens.len(), winning_tokens.len())`.
pub fn divergence_point(label_tokens: &[String], winning_tokens: &[String]) -> usize {
    let limit = label_tokens.len().min(winning_tokens.len());
    for i in 0..limit {
        if label_tokens[i] != winning_tokens[i] {
            return i;
        }
    }
    limit
}

/// A node in the [`LabelTrie`].
#[derive(Debug, Default, Clone)]
pub struct TrieNode {
    /// Child nodes keyed by token text.
    pub children: HashMap<String, TrieNode>,
    /// Whether this node marks the end of a label token sequence.
    pub is_terminal: bool,
    /// The label, if this is a terminal node.
    pub label: Option<String>,
}

impl TrieNode {
    /// Recursively compute the maximum number of children at any node in the
    /// subtree rooted here.
    fn max_branching(&self) -> usize {
        let here = self.children.len();
        let below = self
            .children
            .values()
            .map(|c| c.max_branching())
            .max()
            .unwrap_or(0);
        here.max(below)
    }
}

/// Trie built from per-label token sequences.
///
/// Used to (a) compute the maximum branching factor, which sets the
/// `top_logprobs` budget for the adaptive `generate` loop, and (b) decide
/// whether a candidate token is a valid continuation for some label.
#[derive(Debug, Default, Clone)]
pub struct LabelTrie {
    root: TrieNode,
    /// Preserved insertion order of the token sequence for each label, so that
    /// `get_token_sequence` returns the original (post-fallback) token strings.
    sequences: HashMap<String, Vec<String>>,
}

impl LabelTrie {
    /// Create an empty trie.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a label and its token sequence.
    pub fn insert(&mut self, label: &str, tokens: &[String]) {
        self.sequences.insert(label.to_string(), tokens.to_vec());
        let mut node = &mut self.root;
        for tok in tokens {
            node = node.children.entry(tok.clone()).or_default();
        }
        node.is_terminal = true;
        node.label = Some(label.to_string());
    }

    /// Maximum branching factor across the trie. Drives the `top_logprobs`
    /// budget `k = max(branching, 5)`.
    pub fn max_branching_factor(&self) -> usize {
        self.root.max_branching()
    }

    /// Return the stored token sequence for a label.
    pub fn get_token_sequence(&self, label: &str) -> Vec<String> {
        self.sequences.get(label).cloned().unwrap_or_default()
    }

    /// All labels known to the trie.
    pub fn all_labels(&self) -> Vec<String> {
        self.sequences.keys().cloned().collect()
    }

    /// Reference to the root node (used by the classifier to validate
    /// candidate continuation tokens).
    pub fn root(&self) -> &TrieNode {
        &self.root
    }
}

/// A cluster of labels that share a (so-far) unresolved token prefix.
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Labels still unresolved in this cluster.
    pub labels: Vec<String>,
    /// Number of tokens already resolved (scored) for every label in the
    /// cluster before the next constrained call.
    pub resolved_length: usize,
}

/// Per-label geometric-mean logprob score, computed by walking the *winning*
/// label's token path and, for each label, averaging only the logprobs over
/// the prefix it shares with the winner (`[0, divergence_point]`).
///
/// - `token_sequences`: label → its token sequence.
/// - `winning_label`: the label the constrained call actually emitted.
/// - `step_logprobs`: per-position candidate token → logprob maps along the
///   winning path (already filtered to valid trie tokens by the caller).
pub fn score_labels_from_winning_path(
    token_sequences: &HashMap<String, Vec<String>>,
    winning_label: &str,
    step_logprobs: &[HashMap<String, f64>],
) -> HashMap<String, f64> {
    let mut scores = HashMap::new();
    let winning = token_sequences
        .get(winning_label)
        .cloned()
        .unwrap_or_default();

    for (label, seq) in token_sequences {
        let dp = divergence_point(seq, &winning);
        if dp == 0 {
            scores.insert(label.clone(), f64::NEG_INFINITY);
            continue;
        }
        // Walk the winning path and collect the logprob of this label's token
        // at each shared position. Positions where the label's token isn't in
        // the candidate set contribute -inf.
        let mut lps = Vec::with_capacity(dp);
        for (i, tok) in seq.iter().take(dp).enumerate() {
            let lp = step_logprobs
                .get(i)
                .and_then(|m| m.get(tok).copied())
                .unwrap_or(f64::NEG_INFINITY);
            lps.push(lp);
        }
        let score = geometric_mean_logprob(&lps).unwrap_or(f64::NEG_INFINITY);
        scores.insert(label.clone(), score);
    }
    scores
}

/// Per-label number of tokens that would be scored against the winning path
/// (i.e. the divergence-point length for each label).
pub fn get_scored_lengths(
    token_sequences: &HashMap<String, Vec<String>>,
    winning_label: &str,
) -> HashMap<String, usize> {
    let winning = token_sequences
        .get(winning_label)
        .cloned()
        .unwrap_or_default();
    token_sequences
        .iter()
        .map(|(label, seq)| (label.clone(), divergence_point(seq, &winning)))
        .collect()
}

/// Identify groups of labels that remain unresolved (share an unresolved
/// prefix) and so require another constrained call to disambiguate.
///
/// Each returned [`Cluster`] groups labels that coincide on every already-scored
/// token (`resolved_length`) but have not yet fully diverged; one additional
/// constrained call per cluster can push their resolution further.
pub fn identify_unresolved_clusters(
    token_sequences: &HashMap<String, Vec<String>>,
    scored_lengths: &HashMap<String, usize>,
) -> Vec<Cluster> {
    // Group labels by the token prefix they have already been scored on.
    let mut groups: HashMap<Vec<String>, Vec<String>> = HashMap::new();
    for (label, seq) in token_sequences {
        let resolved = scored_lengths.get(label).copied().unwrap_or(0);
        // A label is fully resolved when its scored prefix length reaches its
        // full token length; exclude those.
        if resolved >= seq.len() {
            continue;
        }
        let prefix: Vec<String> = seq.iter().take(resolved).cloned().collect();
        groups.entry(prefix).or_default().push(label.clone());
    }

    groups
        .into_iter()
        .filter(|(_, members)| members.len() > 1)
        .map(|(prefix, labels)| Cluster {
            labels,
            resolved_length: prefix.len(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometric_mean_logprob_basic() {
        let lps = vec![-1.0, -2.0, -3.0];
        let g = geometric_mean_logprob(&lps).unwrap();
        // sum/len = -6/3
        assert!((g - (-2.0)).abs() < 1e-9);
    }

    #[test]
    fn test_geometric_mean_logprob_filters_neg_inf() {
        // [-inf, -2.0] -> valid=[-2.0] -> -2.0/1 = -2.0
        let lps = vec![f64::NEG_INFINITY, -2.0];
        let g = geometric_mean_logprob(&lps).unwrap();
        assert!((g - (-2.0)).abs() < 1e-9);
    }

    #[test]
    fn test_geometric_mean_logprob_all_neg_inf() {
        let lps = vec![f64::NEG_INFINITY, f64::NEG_INFINITY];
        let g = geometric_mean_logprob(&lps).unwrap();
        assert!(g == f64::NEG_INFINITY);
    }

    #[test]
    fn test_geometric_mean_logprob_empty_errors() {
        assert!(geometric_mean_logprob(&[]).is_err());
    }

    #[test]
    fn test_stable_softmax_basic() {
        let mut lp = HashMap::new();
        lp.insert("a".into(), -1.0);
        lp.insert("b".into(), -2.0);
        lp.insert("c".into(), -3.0);
        let p = stable_softmax(&lp).unwrap();
        assert!((p["a"] - 0.66524096).abs() < 1e-6);
        assert!((p["b"] - 0.24472847).abs() < 1e-6);
        assert!((p["c"] - 0.09003057).abs() < 1e-6);
        // sums to 1
        let total: f64 = p.values().sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stable_softmax_all_neg_inf_is_uniform() {
        let mut lp = HashMap::new();
        lp.insert("a".into(), f64::NEG_INFINITY);
        lp.insert("b".into(), f64::NEG_INFINITY);
        lp.insert("c".into(), f64::NEG_INFINITY);
        let p = stable_softmax(&lp).unwrap();
        for v in p.values() {
            assert!((v - (1.0 / 3.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn test_stable_softmax_empty_errors() {
        let lp = HashMap::new();
        assert!(stable_softmax(&lp).is_err());
    }

    #[test]
    fn test_divergence_point_identical() {
        let a = vec!["x".to_string(), "y".to_string()];
        let b = vec!["x".to_string(), "y".to_string()];
        assert_eq!(divergence_point(&a, &b), 2);
    }

    #[test]
    fn test_divergence_point_differs_midway() {
        let a = vec!["x".to_string(), "y".to_string()];
        let b = vec!["x".to_string(), "z".to_string()];
        assert_eq!(divergence_point(&a, &b), 1);
    }

    #[test]
    fn test_divergence_point_different_length() {
        let a = vec!["x".to_string()];
        let b = vec!["x".to_string(), "y".to_string()];
        assert_eq!(divergence_point(&a, &b), 1);
    }

    #[test]
    fn test_label_trie_branching_factor() {
        // root has 2 children ("a","b"); "a" subtree has 1 child
        let mut trie = LabelTrie::new();
        trie.insert("cat", &["a".to_string(), "x".to_string()]);
        trie.insert("dog", &["b".to_string(), "y".to_string()]);
        assert_eq!(trie.max_branching_factor(), 2);
        assert_eq!(
            trie.get_token_sequence("cat"),
            vec!["a".to_string(), "x".to_string()]
        );
        assert_eq!(trie.all_labels().len(), 2);
    }

    #[test]
    fn test_label_trie_branching_factor_shared_prefix() {
        // root has 1 child; that node has 2 children -> branching = 2
        let mut trie = LabelTrie::new();
        trie.insert("cat", &["a".to_string(), "x".to_string()]);
        trie.insert("car", &["a".to_string(), "y".to_string()]);
        assert_eq!(trie.max_branching_factor(), 2);
    }

    #[test]
    fn test_score_labels_from_winning_path() {
        let mut seq = HashMap::new();
        seq.insert("cat".to_string(), vec!["a".to_string(), "x".to_string()]);
        seq.insert("car".to_string(), vec!["a".to_string(), "y".to_string()]);
        seq.insert("dog".to_string(), vec!["b".to_string(), "z".to_string()]);
        // winning path: tokens "a" (-1.0), "x" (-0.5)
        let steps = vec![
            HashMap::from([("a".to_string(), -1.0), ("b".to_string(), -3.0)]),
            HashMap::from([("x".to_string(), -0.5), ("y".to_string(), -2.0)]),
        ];
        let scores = score_labels_from_winning_path(&seq, "cat", &steps);
        // cat shares full path [a,x]: (-1.0 + -0.5)/2 = -0.75
        assert!((scores["cat"] - (-0.75)).abs() < 1e-9);
        // car diverges at index 1: only [a] -> -1.0
        assert!((scores["car"] - (-1.0)).abs() < 1e-9);
        // dog diverges at index 0: no shared tokens -> dp=0 -> -inf
        assert!(scores["dog"] == f64::NEG_INFINITY);
    }

    #[test]
    fn test_get_scored_lengths() {
        let mut seq = HashMap::new();
        seq.insert("cat".to_string(), vec!["a".to_string(), "x".to_string()]);
        seq.insert("car".to_string(), vec!["a".to_string(), "y".to_string()]);
        let lens = get_scored_lengths(&seq, "cat");
        assert_eq!(lens["cat"], 2);
        assert_eq!(lens["car"], 1);
    }

    #[test]
    fn test_identify_unresolved_clusters() {
        let mut seq = HashMap::new();
        seq.insert("cat".to_string(), vec!["a".to_string(), "x".to_string()]);
        seq.insert("car".to_string(), vec!["a".to_string(), "y".to_string()]);
        seq.insert("dog".to_string(), vec!["b".to_string(), "z".to_string()]);
        // scored 1 token each -> cat & car share prefix ["a"], unresolved; dog resolved alone (no cluster)
        let mut scored = HashMap::new();
        scored.insert("cat".to_string(), 1);
        scored.insert("car".to_string(), 1);
        scored.insert("dog".to_string(), 1);
        let clusters = identify_unresolved_clusters(&seq, &scored);
        assert_eq!(clusters.len(), 1);
        let mut members = clusters[0].labels.clone();
        members.sort();
        assert_eq!(members, vec!["car".to_string(), "cat".to_string()]);
        assert_eq!(clusters[0].resolved_length, 1);
    }
}
