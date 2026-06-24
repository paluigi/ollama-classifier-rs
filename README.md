# ollama-classifier-rs

A Rust port of the Python [`ollama-classifier`](https://github.com/paluigi-moltis/ollama-classifier) library — a backend-agnostic text classifier that delegates inference to any LLM server via the OpenAI-compatible chat completions API.

## Supported Backends

| Backend | Default URL | Description |
|---------|------------|-------------|
| **vLLM** | `http://localhost:8000/v1` | High-throughput serving engine |
| **SGLang** | `http://localhost:30000/v1` | Fast structured generation serving |
| **llama.cpp** | `http://localhost:8080/v1` | Lightweight local inference via `llama-server` |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ollama-classifier-rs = "0.1"
```

### Basic Usage

```rust
use ollama_classifier_rs::backends::VLLMBackend;
use ollama_classifier_rs::LLMClassifier;

fn main() -> ollama_classifier_rs::Result<()> {
    let backend = VLLMBackend::new("meta-llama/Llama-3.2-3B-Instruct");
    let classifier = LLMClassifier::new(backend);

    // Fast single-call classification (no confidence scores)
    let label = classifier.generate(
        "I love this product!",
        vec!["positive", "negative", "neutral"],
        None,
    )?;
    println!("Label: {label}");

    // Classification with calibrated confidence scores
    let result = classifier.classify(
        "I love this product!",
        vec!["positive", "negative", "neutral"],
        None,
    )?;
    println!("Prediction: {}", result.prediction);
    println!("Confidence: {:.2}%", result.confidence * 100.0);

    Ok(())
}
```

### Using Different Backends

```rust
use ollama_classifier_rs::backends::{SGLangBackend, LlamaCppBackend};
use ollama_classifier_rs::LLMClassifier;

// SGLang
let backend = SGLangBackend::with_config(
    "meta-llama/Llama-3.2-3B-Instruct",
    "http://localhost:30000/v1",
);
let classifier = LLMClassifier::new(backend);

// llama.cpp
let backend = LlamaCppBackend::new("model.gguf");
let classifier = LLMClassifier::new(backend);
```

### Custom Configuration (Builder Pattern)

```rust
use ollama_classifier_rs::backends::VLLMBackend;
use std::time::Duration;

let backend = VLLMBackend::builder("meta-llama/Llama-3.2-3B-Instruct")
    .api_key("your-api-key")
    .timeout(Duration::from_secs(60))
    .max_tokens(128)
    .build();
```

### Choices with Descriptions

```rust
use std::collections::HashMap;

let mut choices = HashMap::new();
choices.insert("positive".into(), "The text expresses a positive sentiment.".into());
choices.insert("negative".into(), "The text expresses a negative sentiment.".into());
choices.insert("neutral".into(), "The text is neutral or objective.".into());

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

## Classification Methods

| Method | Calls | Description |
|--------|-------|-------------|
| `generate` | 1 | Fast single-call with JSON schema constraint. Returns only the label. |
| `classify` | N | Multi-call with softmax-calibrated probabilities. N calls for N choices. |
| `batch_*` | N or N×C | Process multiple texts sequentially. |
| `a*` (async) | Same | Async versions of all methods. Requires a Tokio runtime. |

## License

MIT
