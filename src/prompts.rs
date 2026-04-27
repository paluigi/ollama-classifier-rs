//! Prompt building utilities for classification.
//!
//! Mirrors the Python `ollama_classifier.prompts` module.

use serde_json::{json, Value};

use crate::types::Choices;

/// Default system prompt for classification.
const DEFAULT_SYSTEM_PROMPT: &str = "You are a precise text classifier. \
     Your task is to classify the given text into exactly one of the provided categories. \
     Respond with only the category label, nothing else.";

/// Build the system and user prompts for classification.
pub fn build_classification_prompt(
    text: &str,
    choices: &Choices,
    system_prompt: Option<&str>,
) -> (String, String) {
    let choices_text = format_choices(choices);
    let system = system_prompt.unwrap_or(DEFAULT_SYSTEM_PROMPT).to_string();

    let user = format!(
        "Classify the following text into one of these categories:\n\n\
         {choices_text}\n\n\
         Text to classify:\n\
         {text}\n\n\
         Respond with only the category label."
    );

    (system, user)
}

/// Format choices for inclusion in the prompt.
fn format_choices(choices: &Choices) -> String {
    match choices {
        Choices::Labels(labels) => labels
            .iter()
            .map(|l| format!("- {l}"))
            .collect::<Vec<_>>()
            .join("\n"),
        Choices::Descriptions(map) => map
            .iter()
            .map(|(label, desc)| format!("- {label}: {desc}"))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Extract the choice labels from either format.
pub fn get_choice_labels(choices: &Choices) -> Vec<String> {
    choices.labels()
}

/// Build a JSON schema that constrains output to the given choices.
///
/// Produces a schema like:
/// ```json
/// {
///   "type": "object",
///   "properties": {
///     "label": { "type": "string", "enum": ["positive", "negative"] }
///   },
///   "required": ["label"]
/// }
/// ```
pub fn build_json_schema_for_choices(labels: &[String]) -> Value {
    json!({
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": labels,
            }
        },
        "required": ["label"],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_format_choices_labels() {
        let choices = Choices::Labels(vec!["positive".into(), "negative".into(), "neutral".into()]);
        let formatted = format_choices(&choices);
        assert!(formatted.contains("- positive"));
        assert!(formatted.contains("- negative"));
        assert!(formatted.contains("- neutral"));
    }

    #[test]
    fn test_format_choices_descriptions() {
        let mut map = HashMap::new();
        map.insert(
            "positive".into(),
            "The text expresses a positive sentiment.".into(),
        );
        map.insert(
            "negative".into(),
            "The text expresses a negative sentiment.".into(),
        );
        let choices = Choices::Descriptions(map);
        let formatted = format_choices(&choices);
        assert!(formatted.contains("- positive:"));
        assert!(formatted.contains("- negative:"));
    }

    #[test]
    fn test_build_json_schema() {
        let labels = vec!["positive".into(), "negative".into()];
        let schema = build_json_schema_for_choices(&labels);
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["label"]["type"], "string");
        assert_eq!(schema["required"], json!(["label"]));
    }

    #[test]
    fn test_build_classification_prompt_default_system() {
        let choices = Choices::Labels(vec!["positive".into(), "negative".into()]);
        let (system, user) = build_classification_prompt("Great product!", &choices, None);
        assert!(system.contains("text classifier"));
        assert!(user.contains("Great product!"));
        assert!(user.contains("- positive"));
    }

    #[test]
    fn test_build_classification_prompt_custom_system() {
        let choices = Choices::Labels(vec!["yes".into(), "no".into()]);
        let (system, _) = build_classification_prompt("test", &choices, Some("Custom prompt"));
        assert_eq!(system, "Custom prompt");
    }
}
