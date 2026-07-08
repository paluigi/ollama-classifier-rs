//! Basic usage example for ollama-classifier-rs.
//!
//! Run with: `cargo run --example example_usage`
//!
//! Requires a running inference server (Ollama by default). Adjust the backend
//! constructor and model to match your setup.

use ollama_classifier_rs::backends::OllamaBackend;
use ollama_classifier_rs::LLMClassifier;

fn main() -> ollama_classifier_rs::Result<()> {
    let backend = OllamaBackend::new("llama3.2");
    let classifier = LLMClassifier::new(backend);

    let choices = vec!["positive", "negative", "neutral"];

    // --- classify: multi-call completion scoring (exact) ---
    let result = classifier.classify("I love this product!", choices.clone(), None)?;
    println!("=== classify (multi-call) ===");
    println!("Prediction: {}", result.prediction);
    println!("Confidence: {:.2}%", result.confidence * 100.0);
    println!("Method: {} ({} calls)", result.method, result.n_calls);
    println!("Probabilities:");
    for (label, prob) in &result.probabilities {
        println!("  {label}: {:.4}", prob);
    }

    println!();

    // --- generate: adaptive constrained generation (max_calls=1, fast) ---
    let result = classifier.generate("This movie was okay.", choices.clone(), None, Some(1))?;
    println!("=== generate (max_calls=1) ===");
    println!("Prediction: {}", result.prediction);
    println!("Confidence: {:.2}%", result.confidence * 100.0);
    println!("Method: {} ({} calls)", result.method, result.n_calls);
    println!("Approximate: {}", result.approximate);

    Ok(())
}
