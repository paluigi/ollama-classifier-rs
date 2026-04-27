//! Type definitions for ollama-classifier-rs.
//!
//! Mirrors the Python `ollama_classifier.types` module.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of a classification operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// The predicted class label.
    pub prediction: String,
    /// Confidence score for the prediction (0.0 to 1.0).
    pub confidence: f64,
    /// Probability distribution over all choices.
    pub probabilities: HashMap<String, f64>,
}

impl ClassificationResult {
    /// Create a new classification result.
    pub fn new(prediction: String, confidence: f64, probabilities: HashMap<String, f64>) -> Self {
        Self {
            prediction,
            confidence,
            probabilities,
        }
    }
}

/// Choices type — either a simple list of labels, or a map from label to description.
#[derive(Debug, Clone)]
pub enum Choices {
    /// Simple list of choice labels.
    Labels(Vec<String>),
    /// Map from label to description.
    Descriptions(HashMap<String, String>),
}

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
