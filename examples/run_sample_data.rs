//! Batch classification over a small sample dataset.
//!
//! Run with: `cargo run --example run_sample_data`
//!
//! Requires a running inference server. Demonstrates batch (concurrent)
//! classification with both scoring methods.

use ollama_classifier_rs::backends::OllamaBackend;
use ollama_classifier_rs::LLMClassifier;

const SAMPLE_TEXTS: &[&str] = &[
    "I absolutely love this! Best purchase I've ever made.",
    "Terrible quality, broke after one day. Do not buy.",
    "It's okay, I guess. Nothing special but it works.",
    "The customer service was fantastic and resolved my issue quickly.",
    "Worst experience ever. Complete waste of money.",
];

fn main() -> ollama_classifier_rs::Result<()> {
    let backend = OllamaBackend::new("llama3.2");
    // Use up to 4 concurrent threads for the sync batch.
    let classifier = LLMClassifier::with_max_workers(backend, 4);

    let choices = vec!["positive", "negative", "neutral"];

    println!(
        "=== batch_classify (multi-call, {} texts) ===",
        SAMPLE_TEXTS.len()
    );
    let results = classifier.batch_classify(SAMPLE_TEXTS, choices.clone(), None)?;
    for (text, result) in SAMPLE_TEXTS.iter().zip(results.iter()) {
        println!(
            "[{:<8} conf={:.2}%] {:<50}",
            result.prediction,
            result.confidence * 100.0,
            &text[..text.len().min(50)],
        );
    }

    println!();

    // Async batch generation with max_calls=1 for speed.
    println!(
        "=== abatch_generate (max_calls=1, {} texts) ===",
        SAMPLE_TEXTS.len()
    );
    let rt = tokio::runtime::Runtime::new()?;
    let results = rt.block_on(async {
        classifier
            .abatch_generate(SAMPLE_TEXTS, choices, None)
            .await
    })?;
    for (text, result) in SAMPLE_TEXTS.iter().zip(results.iter()) {
        println!(
            "[{:<8} conf={:.2}% approx={}] {:<42}",
            result.prediction,
            result.confidence * 100.0,
            result.approximate,
            &text[..text.len().min(42)],
        );
    }

    Ok(())
}
