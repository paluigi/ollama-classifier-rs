//! LOCAL-ONLY integration test against a real Ollama server.
//!
//! Standalone binary that connects to a running Ollama instance and exercises
//! both scoring methods of [`LLMClassifier`]:
//!
//! - `classify()`  — exact multi-call completion scoring (`method = "multi_call"`)
//! - `generate()`  — adaptive constrained generation (`method = "adaptive_generate"`)
//!
//! Prerequisites
//! -------------
//! 1. Ollama runtime installed and running (>= 0.12): https://ollama.com/download
//! 2. Pull the model once: `ollama pull qwen2.5:3b-instruct`
//!
//! The program prints a message and exits early if Ollama is unreachable.

mod dataset_runner;

use std::collections::HashMap;
use std::net::TcpStream;
use std::time::Duration;

use ollama_classifier_rs::backends::{LLMBackend, OllamaBackend};
use ollama_classifier_rs::{ClassificationResult, LLMClassifier};

const MODEL: &str = "qwen2.5:3b-instruct";
const HOST: &str = "http://localhost:11434";
const PORT: u16 = 11434;

// ---------------------------------------------------------------------------
// Skip guard
// ---------------------------------------------------------------------------

fn port_open(host: &str, port: u16) -> bool {
    use std::net::ToSocketAddrs;
    let mut addrs = match (host, port).to_socket_addrs() {
        Ok(it) => it,
        Err(_) => return false,
    };
    match addrs.next() {
        Some(addr) => TcpStream::connect_timeout(&addr, Duration::from_secs(2)).is_ok(),
        None => false,
    }
}

// ---------------------------------------------------------------------------
// Shared validation helper
// ---------------------------------------------------------------------------

fn assert_valid(result: &ClassificationResult, choices: &[&str], method: &str) -> bool {
    let mut ok = true;
    if result.method != method {
        eprintln!("  FAIL: method={:?}, expected {:?}", result.method, method);
        ok = false;
    }
    if !choices.contains(&result.prediction.as_str()) {
        eprintln!(
            "  FAIL: prediction {:?} not in choices {:?}",
            result.prediction, choices
        );
        ok = false;
    }
    if !(0.0..=1.0).contains(&result.confidence) {
        eprintln!("  FAIL: confidence {} not in [0, 1]", result.confidence);
        ok = false;
    }
    let prob_keys: std::collections::HashSet<&str> =
        result.probabilities.keys().map(|s| s.as_str()).collect();
    let choice_set: std::collections::HashSet<&str> = choices.iter().copied().collect();
    if prob_keys != choice_set {
        eprintln!(
            "  FAIL: probability keys {:?} != choices {:?}",
            prob_keys, choice_set
        );
        ok = false;
    }
    let sum: f64 = result.probabilities.values().sum();
    if (sum - 1.0).abs() >= 1e-6 {
        eprintln!("  FAIL: probabilities sum {} != 1.0", sum);
        ok = false;
    }
    ok
}

// ===========================================================================
// Test cases
// ===========================================================================

fn classify_basic<B: LLMBackend>(classifier: &LLMClassifier<B>) -> bool {
    let text = "The new quantum processor architecture drastically reduces latency.";
    let choices = vec!["technology", "sports", "politics", "entertainment"];

    let result = match classifier.classify(text, choices.clone(), None) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  ERROR: {e}");
            return false;
        }
    };

    println!(
        "\n[classify] prediction={}  confidence={:.2}%",
        result.prediction,
        result.confidence * 100.0
    );
    println!("  n_calls={}  method={}", result.n_calls, result.method);

    if !assert_valid(&result, &choices, "multi_call") {
        return false;
    }
    if result.approximate {
        eprintln!("  FAIL: approximate=true, expected false");
        return false;
    }
    if result.n_calls != choices.len() as i64 {
        eprintln!(
            "  FAIL: n_calls={}, expected {}",
            result.n_calls,
            choices.len()
        );
        return false;
    }
    if result.prediction != "technology" {
        eprintln!(
            "  FAIL: prediction={}, expected 'technology'",
            result.prediction
        );
        return false;
    }
    println!("  PASS");
    true
}

fn classify_with_descriptions<B: LLMBackend>(classifier: &LLMClassifier<B>) -> bool {
    let text = "This restaurant has amazing food but terrible service.";
    let mut choices_map = HashMap::new();
    choices_map.insert(
        "positive".to_string(),
        "Text expresses happiness, satisfaction, or approval".to_string(),
    );
    choices_map.insert(
        "negative".to_string(),
        "Text expresses anger, disappointment, or disapproval".to_string(),
    );
    choices_map.insert(
        "mixed".to_string(),
        "Text contains both positive and negative sentiments".to_string(),
    );
    choices_map.insert(
        "neutral".to_string(),
        "Text is factual without strong emotional content".to_string(),
    );
    let labels: Vec<&str> = vec!["positive", "negative", "mixed", "neutral"];

    let result = match classifier.classify(text, choices_map, None) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  ERROR: {e}");
            return false;
        }
    };

    println!(
        "\n[classify+desc] prediction={}  confidence={:.2}%",
        result.prediction,
        result.confidence * 100.0
    );

    if !assert_valid(&result, &labels, "multi_call") {
        return false;
    }
    if result.prediction != "negative" && result.prediction != "mixed" {
        eprintln!(
            "  FAIL: prediction={}, expected 'negative' or 'mixed'",
            result.prediction
        );
        return false;
    }
    println!("  PASS");
    true
}

fn classify_custom_prompt<B: LLMBackend>(classifier: &LLMClassifier<B>) -> bool {
    let text = "The quarterly earnings exceeded analyst expectations.";
    let choices = vec!["bullish", "bearish", "neutral"];
    let system_prompt = "You are a financial sentiment analyzer. \
        Classify financial news based on market sentiment.";

    let result = match classifier.classify(text, choices.clone(), Some(system_prompt)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  ERROR: {e}");
            return false;
        }
    };

    println!(
        "\n[classify+prompt] prediction={}  confidence={:.2}%",
        result.prediction,
        result.confidence * 100.0
    );

    if !assert_valid(&result, &choices, "multi_call") {
        return false;
    }
    if result.prediction != "bullish" {
        eprintln!(
            "  FAIL: prediction={}, expected 'bullish'",
            result.prediction
        );
        return false;
    }
    println!("  PASS");
    true
}

fn generate_single_call<B: LLMBackend>(classifier: &LLMClassifier<B>) -> bool {
    let text = "The team won the championship!";
    let choices = vec!["sports", "finance", "science", "politics"];

    let result = match classifier.generate(text, choices.clone(), None, Some(1)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  ERROR: {e}");
            return false;
        }
    };

    println!(
        "\n[generate max_calls=1] prediction={}  confidence={:.2}%",
        result.prediction,
        result.confidence * 100.0
    );
    println!(
        "  approximate={}  n_calls={}",
        result.approximate, result.n_calls
    );

    if !assert_valid(&result, &choices, "adaptive_generate") {
        return false;
    }
    if result.n_calls != 1 {
        eprintln!("  FAIL: n_calls={}, expected 1", result.n_calls);
        return false;
    }
    if result.prediction != "sports" {
        eprintln!("  FAIL: prediction={}, expected 'sports'", result.prediction);
        return false;
    }
    println!("  PASS");
    true
}

fn generate_adaptive<B: LLMBackend>(classifier: &LLMClassifier<B>) -> bool {
    let text = "Stock prices plummeted after the announcement.";
    let choices = vec!["sports", "finance", "science", "politics"];

    let result = match classifier.generate(text, choices.clone(), None, Some(3)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  ERROR: {e}");
            return false;
        }
    };

    println!(
        "\n[generate max_calls=3] prediction={}  confidence={:.2}%",
        result.prediction,
        result.confidence * 100.0
    );
    println!(
        "  approximate={}  n_calls={}",
        result.approximate, result.n_calls
    );

    if !assert_valid(&result, &choices, "adaptive_generate") {
        return false;
    }
    if result.n_calls < 1 || result.n_calls > 3 {
        eprintln!(
            "  FAIL: n_calls={}, expected 1..=3",
            result.n_calls
        );
        return false;
    }
    if result.prediction != "finance" {
        eprintln!("  FAIL: prediction={}, expected 'finance'", result.prediction);
        return false;
    }
    println!("  PASS");
    true
}

fn generate_exact<B: LLMBackend>(classifier: &LLMClassifier<B>) -> bool {
    let text = "Scientists discovered a new species in the Amazon.";
    let choices = vec!["sports", "finance", "science", "politics"];

    let result = match classifier.generate(text, choices.clone(), None, None) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  ERROR: {e}");
            return false;
        }
    };

    println!(
        "\n[generate max_calls=None] prediction={}  confidence={:.2}%",
        result.prediction,
        result.confidence * 100.0
    );
    println!(
        "  approximate={}  n_calls={}",
        result.approximate, result.n_calls
    );

    if !assert_valid(&result, &choices, "adaptive_generate") {
        return false;
    }
    if result.prediction != "science" && result.prediction != "politics" {
        eprintln!(
            "  FAIL: prediction={}, expected 'science' or 'politics'",
            result.prediction
        );
        return false;
    }
    println!("  PASS");
    true
}

fn batch_classify<B: LLMBackend>(classifier: &LLMClassifier<B>) -> bool {
    let texts = [
        "The goalkeeper made an incredible save!",
        "The central bank raised interest rates.",
        "The new smartphone features a revolutionary camera.",
    ];
    let choices = vec!["sports", "finance", "technology"];
    let expected = ["sports", "finance", "technology"];

    let results = match classifier.batch_classify(&texts, choices.clone(), None) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  ERROR: {e}");
            return false;
        }
    };

    if results.len() != texts.len() {
        eprintln!(
            "  FAIL: got {} results, expected {}",
            results.len(),
            texts.len()
        );
        return false;
    }

    let mut all_ok = true;
    for (i, ((text, result), exp)) in texts
        .iter()
        .zip(results.iter())
        .zip(expected.iter())
        .enumerate()
    {
        println!(
            "\n[batch_classify {}/{}] {:?} -> {} ({:.2}%)",
            i + 1,
            texts.len(),
            text,
            result.prediction,
            result.confidence * 100.0
        );
        if !assert_valid(result, &choices, "multi_call") {
            all_ok = false;
        }
        if result.prediction != *exp {
            eprintln!(
                "  FAIL: prediction={}, expected '{}'",
                result.prediction, exp
            );
            all_ok = false;
        }
    }
    if all_ok {
        println!("  PASS");
    }
    all_ok
}

fn batch_generate<B: LLMBackend>(classifier: &LLMClassifier<B>) -> bool {
    let texts = [
        "The team secured a decisive victory.",
        "Markets rallied on positive economic data.",
        "The software update fixes critical security vulnerabilities.",
    ];
    let choices = vec!["sports", "finance", "technology"];
    let expected = ["sports", "finance", "technology"];

    let results = match classifier.batch_generate(&texts, choices.clone(), None) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  ERROR: {e}");
            return false;
        }
    };

    if results.len() != texts.len() {
        eprintln!(
            "  FAIL: got {} results, expected {}",
            results.len(),
            texts.len()
        );
        return false;
    }

    let mut all_ok = true;
    for (i, ((text, result), exp)) in texts
        .iter()
        .zip(results.iter())
        .zip(expected.iter())
        .enumerate()
    {
        println!(
            "\n[batch_generate {}/{}] {:?} -> {} ({:.2}%)",
            i + 1,
            texts.len(),
            text,
            result.prediction,
            result.confidence * 100.0
        );
        if !assert_valid(result, &choices, "adaptive_generate") {
            all_ok = false;
        }
        if result.prediction != *exp {
            eprintln!(
                "  FAIL: prediction={}, expected '{}'",
                result.prediction, exp
            );
            all_ok = false;
        }
    }
    if all_ok {
        println!("  PASS");
    }
    all_ok
}

fn dataset<B: LLMBackend>(classifier: &LLMClassifier<B>) -> bool {
    println!("\n[9/9] dataset");
    match dataset_runner::run_dataset_and_save_csv(classifier, "ollama", MODEL) {
        Ok(path) => {
            println!("  CSV saved: {path}");
            println!("  PASS");
            true
        }
        Err(e) => {
            eprintln!("  ERROR: {e}");
            false
        }
    }
}

// ===========================================================================
// Main
// ===========================================================================

fn main() {
    println!("=== Local integration test: Ollama ({MODEL}) ===\n");

    if !port_open("localhost", PORT) {
        eprintln!("Ollama server not reachable at localhost:{PORT} — skipping.");
        return;
    }

    let backend = OllamaBackend::with_config(MODEL, HOST);
    let classifier = LLMClassifier::new(backend);

    let mut passed = 0usize;
    let mut failed = 0usize;

    println!("\n--- [1/9] classify_basic ---");
    if classify_basic(&classifier) {
        passed += 1;
    } else {
        failed += 1;
    }

    println!("\n--- [2/9] classify_with_descriptions ---");
    if classify_with_descriptions(&classifier) {
        passed += 1;
    } else {
        failed += 1;
    }

    println!("\n--- [3/9] classify_custom_prompt ---");
    if classify_custom_prompt(&classifier) {
        passed += 1;
    } else {
        failed += 1;
    }

    println!("\n--- [4/9] generate_single_call ---");
    if generate_single_call(&classifier) {
        passed += 1;
    } else {
        failed += 1;
    }

    println!("\n--- [5/9] generate_adaptive ---");
    if generate_adaptive(&classifier) {
        passed += 1;
    } else {
        failed += 1;
    }

    println!("\n--- [6/9] generate_exact ---");
    if generate_exact(&classifier) {
        passed += 1;
    } else {
        failed += 1;
    }

    println!("\n--- [7/9] batch_classify ---");
    if batch_classify(&classifier) {
        passed += 1;
    } else {
        failed += 1;
    }

    println!("\n--- [8/9] batch_generate ---");
    if batch_generate(&classifier) {
        passed += 1;
    } else {
        failed += 1;
    }

    println!("\n--- [9/9] dataset ---");
    if dataset(&classifier) {
        passed += 1;
    } else {
        failed += 1;
    }

    println!("\n=== Results: {passed} passed, {failed} failed ===");
}
