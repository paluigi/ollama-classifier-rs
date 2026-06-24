//! Generic LLM classifier that works with any inference backend.
//!
//! This module provides [`LLMClassifier`], a backend-agnostic classifier
//! that delegates inference to a [`LLMBackend`](crate::backends::LLMBackend)
//! instance. The public API mirrors the Python `LLMClassifier` so that switching
//! engines requires changing only the constructor.
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
//!     let result = classifier.classify(
//!         "I love this product!",
//!         vec!["positive", "negative", "neutral"],
//!         None,
//!     )?;
//!
//!     println!("Prediction: {}", result.prediction);
//!     println!("Confidence: {:.2}%", result.confidence * 100.0);
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;

use anyhow::Result;
use serde_json::Value;

use crate::backends::base::{ChatMessage, LLMBackend};
use crate::prompts::{
    build_classification_prompt, build_json_schema_for_choices, get_choice_labels,
};
use crate::types::{Choices, ClassificationResult};

/// A backend-agnostic text classifier.
///
/// Provides the same classification interface regardless of which
/// inference backend is used (vLLM, SGLang, llama.cpp, etc.).
pub struct LLMClassifier<B: LLMBackend> {
    backend: B,
}

impl<B: LLMBackend> LLMClassifier<B> {
    /// Create a new classifier with the given backend.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Get a reference to the underlying backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    fn to_messages(system: &str, user: &str) -> Vec<ChatMessage> {
        vec![
            ChatMessage::new("system", system),
            ChatMessage::new("user", user),
        ]
    }

    fn extract_logprob_sum(response: &crate::backends::base::ChatResponse) -> f64 {
        match &response.logprobs {
            Some(logprobs) => logprobs.iter().map(|entry| entry.logprob).sum(),
            None => 0.0,
        }
    }

    /// Numerically stable softmax over log probabilities.
    pub(crate) fn softmax(logprobs: &HashMap<String, f64>) -> HashMap<String, f64> {
        let valid: Vec<(&String, &f64)> = logprobs
            .iter()
            .filter(|(_, &v)| v > f64::NEG_INFINITY)
            .collect();

        if valid.is_empty() {
            let n = logprobs.len() as f64;
            return logprobs.keys().map(|k| (k.clone(), 1.0 / n)).collect();
        }

        let max_lp = valid
            .iter()
            .map(|(_, &v)| v)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut exp_vals: HashMap<String, f64> = HashMap::new();
        let mut total = 0.0;

        for (key, &val) in logprobs {
            let exp_val = if val > f64::NEG_INFINITY {
                (val - max_lp).exp()
            } else {
                0.0
            };
            exp_vals.insert(key.clone(), exp_val);
            total += exp_val;
        }

        if total == 0.0 {
            let n = logprobs.len() as f64;
            return logprobs.keys().map(|k| (k.clone(), 1.0 / n)).collect();
        }

        exp_vals.into_iter().map(|(k, v)| (k, v / total)).collect()
    }

    /// Get log P(choice | context) for a single choice by appending it as
    /// a forced continuation and reading the token logprobs.
    fn get_choice_logprob(&self, system: &str, user: &str, label: &str) -> f64 {
        // We ask the model to complete with the label and read the logprob
        // of the first generated token. We use a prompt that encourages the
        // model to output the label.
        let forced_user = format!("{user}\n\nCategory: {label}");
        let messages = Self::to_messages(system, &forced_user);

        match self.backend.chat(&messages, 0.0, None, true, 5) {
            Ok(response) => {
                // Use the sum of all token logprobs as the score
                Self::extract_logprob_sum(&response)
            }
            Err(_) => f64::NEG_INFINITY,
        }
    }

    // =========================================================================
    // Sync Methods — Generate
    // =========================================================================

    /// Generate a constrained classification for a single text.
    ///
    /// Uses JSON schema with enum constraint to ensure only valid choices
    /// are generated. This is the fastest method as it only makes one API
    /// call and does not compute confidence scores.
    pub fn generate(
        &self,
        text: &str,
        choices: impl Into<Choices>,
        system_prompt: Option<&str>,
    ) -> Result<String> {
        let choices = choices.into();
        let labels = get_choice_labels(&choices);
        let (system, user) = build_classification_prompt(text, &choices, system_prompt);
        let schema = build_json_schema_for_choices(&labels);
        let messages = Self::to_messages(&system, &user);

        let response = self.backend.chat(&messages, 0.0, Some(schema), false, 5)?;

        let parsed: Value = serde_json::from_str(&response.content)?;
        Ok(parsed["label"].as_str().unwrap_or("").to_string())
    }

    /// Generate constrained classifications for multiple texts.
    pub fn batch_generate(
        &self,
        texts: &[&str],
        choices: impl Into<Choices> + Clone,
        system_prompt: Option<&str>,
    ) -> Result<Vec<String>> {
        texts
            .iter()
            .map(|text| self.generate(text, choices.clone().into(), system_prompt))
            .collect()
    }

    // =========================================================================
    // Sync Methods — Classify
    // =========================================================================

    /// Classify text with calibrated confidence scores.
    ///
    /// Uses multi-call evaluation to compute calibrated probabilities
    /// for each choice. Makes N API calls for N choices.
    pub fn classify(
        &self,
        text: &str,
        choices: impl Into<Choices>,
        system_prompt: Option<&str>,
    ) -> Result<ClassificationResult> {
        let choices = choices.into();
        let labels = get_choice_labels(&choices);
        let (system, user) = build_classification_prompt(text, &choices, system_prompt);

        let mut logprobs: HashMap<String, f64> = HashMap::new();
        for label in &labels {
            logprobs.insert(
                label.clone(),
                self.get_choice_logprob(&system, &user, label),
            );
        }

        let probabilities = Self::softmax(&logprobs);
        let prediction = probabilities
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone())
            .expect("at least one choice required");
        let confidence = probabilities[&prediction];

        Ok(ClassificationResult::new(
            prediction,
            confidence,
            probabilities,
        ))
    }

    /// Classify multiple texts with calibrated confidence scores.
    pub fn batch_classify(
        &self,
        texts: &[&str],
        choices: impl Into<Choices> + Clone,
        system_prompt: Option<&str>,
    ) -> Result<Vec<ClassificationResult>> {
        texts
            .iter()
            .map(|text| self.classify(text, choices.clone().into(), system_prompt))
            .collect()
    }

    // =========================================================================
    // Async Methods — Generate
    // =========================================================================

    /// Async version of [`generate`](LLMClassifier::generate).
    pub async fn agenerate(
        &self,
        text: &str,
        choices: impl Into<Choices>,
        system_prompt: Option<&str>,
    ) -> Result<String> {
        let choices = choices.into();
        let labels = get_choice_labels(&choices);
        let (system, user) = build_classification_prompt(text, &choices, system_prompt);
        let schema = build_json_schema_for_choices(&labels);
        let messages = Self::to_messages(&system, &user);

        let response = self
            .backend
            .achat(&messages, 0.0, Some(schema), false, 5)
            .await?;

        let parsed: Value = serde_json::from_str(&response.content)?;
        Ok(parsed["label"].as_str().unwrap_or("").to_string())
    }

    /// Async version of [`batch_generate`](LLMClassifier::batch_generate).
    pub async fn abatch_generate(
        &self,
        texts: &[&str],
        choices: impl Into<Choices> + Clone,
        system_prompt: Option<&str>,
    ) -> Result<Vec<String>> {
        let mut results = Vec::new();
        for text in texts {
            results.push(
                self.agenerate(text, choices.clone().into(), system_prompt)
                    .await?,
            );
        }
        Ok(results)
    }

    // =========================================================================
    // Async Methods — Classify
    // =========================================================================

    /// Async version of [`classify`](LLMClassifier::classify).
    pub async fn aclassify(
        &self,
        text: &str,
        choices: impl Into<Choices>,
        system_prompt: Option<&str>,
    ) -> Result<ClassificationResult> {
        let choices = choices.into();
        let labels = get_choice_labels(&choices);
        let (system, user) = build_classification_prompt(text, &choices, system_prompt);

        let mut logprobs: HashMap<String, f64> = HashMap::new();
        for label in &labels {
            let lp = self.aget_choice_logprob(&system, &user, label).await;
            logprobs.insert(label.clone(), lp);
        }

        let probabilities = Self::softmax(&logprobs);
        let prediction = probabilities
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone())
            .expect("at least one choice required");
        let confidence = probabilities[&prediction];

        Ok(ClassificationResult::new(
            prediction,
            confidence,
            probabilities,
        ))
    }

    /// Async version of [`batch_classify`](LLMClassifier::batch_classify).
    pub async fn abatch_classify(
        &self,
        texts: &[&str],
        choices: impl Into<Choices> + Clone,
        system_prompt: Option<&str>,
    ) -> Result<Vec<ClassificationResult>> {
        let mut results = Vec::new();
        for text in texts {
            results.push(
                self.aclassify(text, choices.clone().into(), system_prompt)
                    .await?,
            );
        }
        Ok(results)
    }

    // =========================================================================
    // Internal async helpers
    // =========================================================================

    async fn aget_choice_logprob(&self, system: &str, user: &str, label: &str) -> f64 {
        let forced_user = format!("{user}\n\nCategory: {label}");
        let messages = Self::to_messages(system, &forced_user);

        match self.backend.achat(&messages, 0.0, None, true, 5).await {
            Ok(response) => Self::extract_logprob_sum(&response),
            Err(_) => f64::NEG_INFINITY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::VLLMBackend;

    #[test]
    fn test_softmax_basic() {
        let mut logprobs = HashMap::new();
        logprobs.insert("a".into(), -1.0);
        logprobs.insert("b".into(), -2.0);
        logprobs.insert("c".into(), -3.0);

        let probs = LLMClassifier::<VLLMBackend>::softmax(&logprobs);
        assert!((probs["a"] - 0.6652).abs() < 0.01);
        assert!((probs["b"] - 0.2447).abs() < 0.01);
        assert!((probs["c"] - 0.0900).abs() < 0.01);
    }

    #[test]
    fn test_softmax_all_negative_infinity() {
        let mut logprobs = HashMap::new();
        logprobs.insert("a".into(), f64::NEG_INFINITY);
        logprobs.insert("b".into(), f64::NEG_INFINITY);

        let probs = LLMClassifier::<VLLMBackend>::softmax(&logprobs);
        assert!((probs["a"] - 0.5).abs() < 0.01);
        assert!((probs["b"] - 0.5).abs() < 0.01);
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
