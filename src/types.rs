//! Type definitions for ollama-classifier-rs.
//!
//! Mirrors the Python `ollama_classifier.types` module. [`ClassificationResult`]
//! is serialized with the same field names as the Pydantic model so that
//! outputs are interchangeable between the two implementations.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;

/// Result of a classification operation.
///
/// Field-for-field compatible with the Python `ClassificationResult` Pydantic
/// model (the new fields `method`, `approximate`, `coverage`, `n_calls`, and
/// `raw_response` were introduced in the v0.4.0 redesign).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// The predicted class label.
    pub prediction: String,
    /// Confidence score for the prediction (0.0 to 1.0).
    pub confidence: f64,
    /// Probability distribution over all choices.
    pub probabilities: HashMap<String, f64>,
    /// How the result was produced: `"multi_call"` or `"adaptive_generate"`.
    #[serde(default = "default_method")]
    pub method: String,
    /// `true` when the adaptive `generate` path could not fully resolve every
    /// label token sequence within the `max_calls` budget.
    #[serde(default)]
    pub approximate: bool,
    /// Per-label fraction of tokens that were actually scored (only meaningful
    /// for the adaptive `generate` path; empty for `multi_call`).
    #[serde(default)]
    pub coverage: HashMap<String, f64>,
    /// Number of backend calls made to produce this result.
    #[serde(default = "default_n_calls")]
    pub n_calls: i64,
    /// Raw backend response payload, when retained.
    #[serde(default)]
    pub raw_response: Map<String, Value>,
}

fn default_method() -> String {
    "multi_call".to_string()
}

fn default_n_calls() -> i64 {
    1
}

impl ClassificationResult {
    /// Create a result for the multi-call completion-scoring path.
    ///
    /// `n_calls` is set to the number of labels scored and `coverage` is empty
    /// (every label is fully scored by definition).
    pub fn new_multi_call(
        prediction: String,
        confidence: f64,
        probabilities: HashMap<String, f64>,
        n_calls: i64,
    ) -> Self {
        Self {
            prediction,
            confidence,
            probabilities,
            method: "multi_call".to_string(),
            approximate: false,
            coverage: HashMap::new(),
            n_calls,
            raw_response: Map::new(),
        }
    }

    /// Create a result for the adaptive-constrained-generation path.
    pub fn new_adaptive(
        prediction: String,
        confidence: f64,
        probabilities: HashMap<String, f64>,
        coverage: HashMap<String, f64>,
        n_calls: i64,
        approximate: bool,
    ) -> Self {
        Self {
            prediction,
            confidence,
            probabilities,
            method: "adaptive_generate".to_string(),
            approximate,
            coverage,
            n_calls,
            raw_response: Map::new(),
        }
    }
}

/// Choices type — either a simple list of labels, or a map from label to description.
///
/// Accepting `impl Into<Choices>` lets callers pass `Vec<&str>`, `Vec<String>`,
/// or `HashMap<String, String>` directly.
#[derive(Debug, Clone)]
pub enum Choices {
    /// Simple list of choice labels.
    Labels(Vec<String>),
    /// Map from label to description.
    Descriptions(HashMap<String, String>),
}

/// Type alias mirroring the Python `ChoicesType`.
pub type ChoicesType = Choices;

impl Choices {
    /// Extract the choice labels from either format.
    pub fn labels(&self) -> Vec<String> {
        match self {
            Choices::Labels(v) => v.clone(),
            Choices::Descriptions(m) => m.keys().cloned().collect(),
        }
    }

    /// Check if choices is a descriptions map.
    pub fn is_descriptions(&self) -> bool {
        matches!(self, Choices::Descriptions(_))
    }
}

impl From<Vec<String>> for Choices {
    fn from(v: Vec<String>) -> Self {
        Choices::Labels(v)
    }
}

impl From<Vec<&str>> for Choices {
    fn from(v: Vec<&str>) -> Self {
        Choices::Labels(v.into_iter().map(String::from).collect())
    }
}

impl From<HashMap<String, String>> for Choices {
    fn from(m: HashMap<String, String>) -> Self {
        Choices::Descriptions(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_multi_call_serializes_with_python_fields() {
        let mut probs = HashMap::new();
        probs.insert("positive".into(), 0.7);
        probs.insert("negative".into(), 0.3);
        let r = ClassificationResult::new_multi_call("positive".into(), 0.7, probs, 2);
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["prediction"], "positive");
        assert_eq!(json["method"], "multi_call");
        assert_eq!(json["approximate"], false);
        assert_eq!(json["n_calls"], 2);
        assert!(json["coverage"].as_object().unwrap().is_empty());
        assert!(json["raw_response"].as_object().unwrap().is_empty());
    }

    #[test]
    fn test_result_adaptive_carries_coverage() {
        let mut probs = HashMap::new();
        probs.insert("a".into(), 0.6);
        probs.insert("b".into(), 0.4);
        let mut cov = HashMap::new();
        cov.insert("a".into(), 1.0);
        cov.insert("b".into(), 0.5);
        let r = ClassificationResult::new_adaptive("a".into(), 0.6, probs, cov, 3, true);
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["method"], "adaptive_generate");
        assert_eq!(json["approximate"], true);
        assert_eq!(json["coverage"]["b"], 0.5);
    }
}
