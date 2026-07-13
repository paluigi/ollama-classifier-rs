# ollama-classifier-rs

A Rust port of the Python [`ollama-classifier`](https://github.com/paluigi/ollama-classifier) library (v0.5.0) — a backend-agnostic text classifier that delegates inference to any LLM server and produces calibrated confidence scores.

## Supported Backends

| Backend | Default URL | Constraint mechanism | Bare label? |
|---------|------------|----------------------|-------------|
| **Ollama** | `http://localhost:11434` | Native API, JSON-schema `format` enum | No (JSON-wrapped) |
| **vLLM** | `http://localhost:8000/v1` | OpenAI-compatible, `structured_outputs.choice` (v0.12.0+) | Yes |
| **SGLang** | `http://localhost:30000/v1` | OpenAI-compatible, `regex` | Yes |
| **llama.cpp** | `http://localhost:8080/v1` | OpenAI-compatible, GBNF `grammar` | Yes |

All backends use empirical **forced constrained generation** for tokenization
(forcing the label as the only valid choice and reading back the emitted value
tokens) and echo/prefill (vLLM, SGLang) or forced generation (Ollama, llama.cpp)
for completion scoring. Tokenization results are memoized per label.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ollama-classifier-rs = "0.5"
```

### Basic Usage

```rust
use ollama_classifier_rs::backends::OllamaBackend;
use ollama_classifier_rs::LLMClassifier;

fn main() -> ollama_classifier_rs::Result<()> {
    let backend = OllamaBackend::new("llama3.2");
    let classifier = LLMClassifier::new(backend);

    // Multi-call classification with calibrated probabilities (N calls for N labels)
    let result = classifier.classify(
        "I love this product!",
        vec!["positive", "negative", "neutral"],
        None,
    )?;

    println!("Prediction: {}", result.prediction);
    println!("Confidence: {:.2}%", result.confidence * 100.0);
    println!("Method: {} ({} calls)", result.method, result.n_calls);
    Ok(())
}
```

### Adaptive Generation (`generate`)

`generate` uses constrained generation with an adaptive trie-based scoring
algorithm. The `max_calls` budget controls the accuracy/cost trade-off:

```rust
// Single fast call — partial/approximate scoring
let result = classifier.generate(
    "This movie was okay.",
    vec!["positive", "negative", "neutral"],
    None,
    Some(1),
)?;

// Fully exact — resolves all ambiguity (up to N calls)
let result = classifier.generate(
    "This movie was okay.",
    vec!["positive", "negative", "neutral"],
    None,
    None,
)?;

if result.approximate {
    println!("Scoring is partial (max_calls budget reached)");
}
println!("Coverage: {:?}", result.coverage);
```

### Using Different Backends

```rust
use ollama_classifier_rs::backends::{VLLMBackend, SGLangBackend, LlamaCppBackend};
use ollama_classifier_rs::LLMClassifier;

// vLLM
let backend = VLLMBackend::new("meta-llama/Llama-3.2-3B-Instruct");
let classifier = LLMClassifier::new(backend);

// SGLang
let backend = SGLangBackend::new("meta-llama/Llama-3.2-3B-Instruct");

// llama.cpp (llama-server)
let backend = LlamaCppBackend::new("model.gguf");
```

### Custom Configuration (Builder Pattern)

```rust
use ollama_classifier_rs::backends::VLLMBackend;
use std::time::Duration;

let backend = VLLMBackend::builder("meta-llama/Llama-3.2-3B-Instruct", "http://localhost:8000/v1")
    .api_key("your-api-key")
    .timeout(Duration::from_secs(60))
    .max_tokens(128)
    .build();
```

### Choices with Descriptions

```rust
use std::collections::HashMap;

let mut choices = HashMap::new();
choices.insert("positive".to_string(), "The text expresses a positive sentiment.".to_string());
choices.insert("negative".to_string(), "The text expresses a negative sentiment.".to_string());
choices.insert("neutral".to_string(), "The text is neutral or objective.".to_string());

let result = classifier.classify("This movie was okay.", choices, None)?;
```

### Async Usage

```rust
use ollama_classifier_rs::backends::VLLMBackend;
use ollama_classifier_rs::LLMClassifier;

#[tokio::main]
async fn main() -> ollama_classifier_rs::Result<()> {
    let backend = VLLMBackend::new("meta-llama/Llama-3.2-3B-Instruct");
    let classifier = LLMClassifier::new(backend);

    let result = classifier
        .aclassify("I love this product!", vec!["positive", "negative", "neutral"], None)
        .await?;

    println!("Prediction: {}", result.prediction);
    Ok(())
}
```

### Batch Classification

Batch methods process multiple texts concurrently:

```rust
let texts = ["I love it!", "I hate it.", "It was okay."];

// Sync batch (uses up to max_workers threads)
let results = classifier.batch_classify(
    &texts,
    vec!["positive", "negative", "neutral"],
    None,
)?;

// Async batch (all texts concurrently)
let results = classifier
    .abatch_classify(&texts, vec!["positive", "negative", "neutral"], None)
    .await?;
```

Control sync-batch concurrency with `LLMClassifier::with_max_workers`:

```rust
let classifier = LLMClassifier::with_max_workers(backend, 8);
```

## Classification Methods

| Method | Calls | Description |
|--------|-------|-------------|
| [`classify`](LLMClassifier::classify) | N (one per label) | Multi-call completion scoring with geometric-mean normalization. Exact. |
| [`generate`](LLMClassifier::generate) | ≤ `max_calls` | Adaptive trie-masked constrained generation. `max_calls=1` is fast/approximate; `None` is exact. |
| `batch_*` | N or ≤max_calls per text | Concurrent batch variants of the above. |
| `a*` (async) | Same | Async versions of all methods. |

### `ClassificationResult`

All methods return a `ClassificationResult`:

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | `String` | The predicted class label. |
| `confidence` | `f64` | Confidence score for the prediction (0.0–1.0). |
| `probabilities` | `HashMap<String, f64>` | Probability distribution over all choices. |
| `method` | `String` | `"multi_call"` or `"adaptive_generate"`. |
| `approximate` | `bool` | `true` if `generate` couldn't fully resolve every label within `max_calls`. |
| `coverage` | `HashMap<String, f64>` | Per-label fraction of tokens scored (adaptive path only). |
| `n_calls` | `i64` | Number of backend calls made. |

## License

MIT
