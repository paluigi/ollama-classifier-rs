//! LOCAL-ONLY integration tests against a real SGLang server.
//!
//! Run with: `cargo run --example test_local_ollama`
//! (add to [[example]] in Cargo.toml, or run as `rustc` standalone)
//!
//! Skips gracefully if the server is unreachable.

#[path = "dataset_runner.rs"]
mod dataset_runner;

use std::net::TcpStream;
use std::time::Duration;

use ollama_classifier_rs::backends::SGLangBackend;
use ollama_classifier_rs::LLMClassifier;

const MODEL: &str = "Qwen2.5-3B-Instruct-GGUF";
const HOST: &str = "http://localhost:30000/v1";
const PORT: u16 = 30000;

fn server_reachable() -> bool {
    TcpStream::connect_timeout(
        &format!("localhost:{PORT}").parse().unwrap(),
        Duration::from_secs(2),
    )
    .is_ok()
}

fn assert_valid(result: &ollama_classifier_rs::ClassificationResult, choices: &[&str], method: &str) {
    assert_eq!(result.method, method, "method mismatch");
    assert!(
        choices.contains(&result.prediction.as_str()),
        "prediction {} not in choices",
        result.prediction
    );
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    let total: f64 = result.probabilities.values().sum();
    assert!((total - 1.0).abs() < 1e-6, "probs don't sum to 1: {total}");
}

fn main() {
    if !server_reachable() {
        eprintln!("Skipping: SGLang server not reachable at localhost:{PORT}");
        return;
    }

    let backend = SGLangBackend::with_config(MODEL, HOST);
    let classifier = LLMClassifier::new(backend);

    // 1. classify_basic
    println!("\n=== classify: basic ===");
    let result = classifier
        .classify(
            "The new quantum processor architecture drastically reduces latency.",
            vec!["technology", "sports", "politics", "entertainment"],
            None,
        )
        .unwrap();
    println!("  prediction={}, confidence={:.2}%", result.prediction, result.confidence * 100.0);
    assert_valid(&result, &["technology", "sports", "politics", "entertainment"], "multi_call");
    assert_eq!(result.n_calls, 4);
    assert_eq!(result.prediction, "technology");

    // 2. classify_with_descriptions
    println!("\n=== classify: with descriptions ===");
    use std::collections::HashMap;
    let mut choices = HashMap::new();
    choices.insert("positive".into(), "Text expresses happiness, satisfaction, or approval".into());
    choices.insert("negative".into(), "Text expresses anger, disappointment, or disapproval".into());
    choices.insert("mixed".into(), "Text contains both positive and negative sentiments".into());
    choices.insert("neutral".into(), "Text is factual without strong emotional content".into());
    let result = classifier
        .classify("This restaurant has amazing food but terrible service.", choices, None)
        .unwrap();
    println!("  prediction={}, confidence={:.2}%", result.prediction, result.confidence * 100.0);
    assert!(result.prediction == "negative" || result.prediction == "mixed");

    // 3. classify_custom_prompt
    println!("\n=== classify: custom prompt ===");
    let result = classifier
        .classify(
            "The quarterly earnings exceeded analyst expectations.",
            vec!["bullish", "bearish", "neutral"],
            Some("You are a financial sentiment analyzer. Classify financial news based on market sentiment."),
        )
        .unwrap();
    println!("  prediction={}, confidence={:.2}%", result.prediction, result.confidence * 100.0);
    assert_eq!(result.prediction, "bullish");

    // 4. generate_single_call
    println!("\n=== generate: max_calls=1 ===");
    let result = classifier
        .generate(
            "The team won the championship!",
            vec!["sports", "finance", "science", "politics"],
            None,
            Some(1),
        )
        .unwrap();
    println!("  prediction={}, confidence={:.2}%, approximate={}", result.prediction, result.confidence * 100.0, result.approximate);
    assert_eq!(result.n_calls, 1);
    assert_eq!(result.prediction, "sports");

    // 5. generate_adaptive
    println!("\n=== generate: max_calls=3 ===");
    let result = classifier
        .generate(
            "Stock prices plummeted after the announcement.",
            vec!["sports", "finance", "science", "politics"],
            None,
            Some(3),
        )
        .unwrap();
    println!("  prediction={}, n_calls={}", result.prediction, result.n_calls);
    assert!(result.n_calls >= 1 && result.n_calls <= 3);
    assert_eq!(result.prediction, "finance");

    // 6. generate_exact
    println!("\n=== generate: max_calls=None ===");
    let result = classifier
        .generate(
            "Scientists discovered a new species in the Amazon.",
            vec!["sports", "finance", "science", "politics"],
            None,
            None,
        )
        .unwrap();
    println!("  prediction={}, confidence={:.2}%", result.prediction, result.confidence * 100.0);
    assert!(result.prediction == "science" || result.prediction == "politics");

    // 7. batch_classify
    println!("\n=== batch_classify ===");
    let texts = [
        "The goalkeeper made an incredible save!",
        "The central bank raised interest rates.",
        "The new smartphone features a revolutionary camera.",
    ];
    let results = classifier
        .batch_classify(&texts, vec!["sports", "finance", "technology"], None)
        .unwrap();
    for (text, result) in texts.iter().zip(results.iter()) {
        println!("  {text:?} -> {} ({:.2}%)", result.prediction, result.confidence * 100.0);
    }
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].prediction, "sports");
    assert_eq!(results[1].prediction, "finance");
    assert_eq!(results[2].prediction, "technology");

    // 8. batch_generate
    println!("\n=== batch_generate ===");
    let texts = [
        "The team secured a decisive victory.",
        "Markets rallied on positive economic data.",
        "The software update fixes critical security vulnerabilities.",
    ];
    let results = classifier
        .batch_generate(&texts, vec!["sports", "finance", "technology"], None)
        .unwrap();
    for (text, result) in texts.iter().zip(results.iter()) {
        println!("  {text:?} -> {} ({:.2}%)", result.prediction, result.confidence * 100.0);
    }
    assert_eq!(results[0].prediction, "sports");
    assert_eq!(results[1].prediction, "finance");
    assert_eq!(results[2].prediction, "technology");

    // 9. Dataset
    println!("\n=== dataset ===");
    let csv_path = dataset_runner::run_dataset_and_save_csv(&classifier, "sglang", MODEL)
        .expect("dataset CSV save failed");
    println!("  CSV: {csv_path}");

    println!("\n✓ All SGLang tests passed!");
}
