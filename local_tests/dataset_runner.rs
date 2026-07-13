//! Shared dataset evaluation helper for local_tests.
//!
//! Contains the test dataset and a runner that classifies every entry with
//! both `classify()` and `generate()`, then writes results to a timestamped
//! CSV.
//!
//! A `"none_of_the_above"` choice is appended to the category list so the
//! model can express that no category fits — its probability is reported in
//! the `prob_none` column.

use std::io::Write;

use ollama_classifier_rs::{Choices, ClassificationResult, LLMClassifier};
use ollama_classifier_rs::backends::base::LLMBackend;

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DatasetEntry {
    pub id: u32,
    pub text: String,
    pub ambiguity_level: String,
    pub primary_category: Option<String>,
    pub secondary_category: Option<String>,
}

pub fn categories() -> Vec<String> {
    vec![
        "technology".into(),
        "food_cooking".into(),
        "sports_fitness".into(),
        "finance_investing".into(),
        "travel_tourism".into(),
    ]
}

pub fn dataset() -> Vec<DatasetEntry> {
    vec![
        DatasetEntry { id: 1, text: "The new smartphone features an upgraded octa-core processor and 12GB of RAM for seamless multitasking.".into(), ambiguity_level: "clear".into(), primary_category: Some("technology".into()), secondary_category: None },
        DatasetEntry { id: 2, text: "I need to update my operating system because some software applications are crashing on startup.".into(), ambiguity_level: "clear".into(), primary_category: Some("technology".into()), secondary_category: None },
        DatasetEntry { id: 3, text: "Cloud computing allows businesses to scale their server infrastructure dynamically based on demand.".into(), ambiguity_level: "clear".into(), primary_category: Some("technology".into()), secondary_category: None },
        DatasetEntry { id: 4, text: "Whisk the egg whites until stiff peaks form before gently folding them into the cake batter.".into(), ambiguity_level: "clear".into(), primary_category: Some("food_cooking".into()), secondary_category: None },
        DatasetEntry { id: 5, text: "This local Italian restaurant serves authentic wood-fired Neapolitan pizza with fresh basil and mozzarella.".into(), ambiguity_level: "clear".into(), primary_category: Some("food_cooking".into()), secondary_category: None },
        DatasetEntry { id: 6, text: "Slow-roasting garlic in olive oil at a low temperature produces a sweet, spreadable paste.".into(), ambiguity_level: "clear".into(), primary_category: Some("food_cooking".into()), secondary_category: None },
        DatasetEntry { id: 7, text: "The striker scored a stunning hat-trick in the second half to secure a victory for his team.".into(), ambiguity_level: "clear".into(), primary_category: Some("sports_fitness".into()), secondary_category: None },
        DatasetEntry { id: 8, text: "Proper hydration and stretching before running a marathon are essential to prevent muscle cramps.".into(), ambiguity_level: "clear".into(), primary_category: Some("sports_fitness".into()), secondary_category: None },
        DatasetEntry { id: 9, text: "Our local basketball league is looking for new referees to officiate the upcoming weekend games.".into(), ambiguity_level: "clear".into(), primary_category: Some("sports_fitness".into()), secondary_category: None },
        DatasetEntry { id: 10, text: "Diversifying your investment portfolio across stocks, bonds, and real estate helps mitigate risk.".into(), ambiguity_level: "clear".into(), primary_category: Some("finance_investing".into()), secondary_category: None },
        DatasetEntry { id: 11, text: "The central bank decided to raise interest rates to curb rising inflation across the country.".into(), ambiguity_level: "clear".into(), primary_category: Some("finance_investing".into()), secondary_category: None },
        DatasetEntry { id: 12, text: "Opening a high-yield savings account is a simple way to earn interest on your emergency fund.".into(), ambiguity_level: "clear".into(), primary_category: Some("finance_investing".into()), secondary_category: None },
        DatasetEntry { id: 13, text: "We spent the afternoon exploring the historic ruins of Rome and taking photos of the Colosseum.".into(), ambiguity_level: "clear".into(), primary_category: Some("travel_tourism".into()), secondary_category: None },
        DatasetEntry { id: 14, text: "Remember to check the visa requirements and passport validity before booking your flights abroad.".into(), ambiguity_level: "clear".into(), primary_category: Some("travel_tourism".into()), secondary_category: None },
        DatasetEntry { id: 15, text: "The boutique hotel offers stunning ocean views and is located just steps from the sandy beach.".into(), ambiguity_level: "clear".into(), primary_category: Some("travel_tourism".into()), secondary_category: None },
        DatasetEntry { id: 16, text: "Backpacking through Southeast Asia is an affordable way for students to experience diverse cultures.".into(), ambiguity_level: "clear".into(), primary_category: Some("travel_tourism".into()), secondary_category: None },
        DatasetEntry { id: 17, text: "This new smart air fryer connects to your home Wi-Fi, allowing you to monitor cooking progress from an app.".into(), ambiguity_level: "mildly_ambiguous".into(), primary_category: Some("technology".into()), secondary_category: Some("food_cooking".into()) },
        DatasetEntry { id: 18, text: "I bought a premium smartwatch to track my daily steps, heart rate variability, and GPS routes during morning jogs.".into(), ambiguity_level: "mildly_ambiguous".into(), primary_category: Some("technology".into()), secondary_category: Some("sports_fitness".into()) },
        DatasetEntry { id: 19, text: "While wandering the streets of Paris, I stumbled upon a tiny bakery serving the most incredible butter croissants.".into(), ambiguity_level: "mildly_ambiguous".into(), primary_category: Some("food_cooking".into()), secondary_category: Some("travel_tourism".into()) },
        DatasetEntry { id: 20, text: "I used a digital kitchen scale and a specialized molecular gastronomy calculator to measure the sodium alginate for this recipe.".into(), ambiguity_level: "mildly_ambiguous".into(), primary_category: Some("food_cooking".into()), secondary_category: Some("technology".into()) },
        DatasetEntry { id: 21, text: "The professional football player signed a multi-million dollar contract extension, making him the highest-paid athlete this season.".into(), ambiguity_level: "mildly_ambiguous".into(), primary_category: Some("sports_fitness".into()), secondary_category: Some("finance_investing".into()) },
        DatasetEntry { id: 22, text: "The cycling team utilized wind-tunnel data and advanced computational fluid dynamics software to optimize their riding postures.".into(), ambiguity_level: "mildly_ambiguous".into(), primary_category: Some("sports_fitness".into()), secondary_category: Some("technology".into()) },
        DatasetEntry { id: 23, text: "The sudden surge in cryptocurrency trading caused several online brokerage platforms to experience temporary server outages.".into(), ambiguity_level: "mildly_ambiguous".into(), primary_category: Some("finance_investing".into()), secondary_category: Some("technology".into()) },
        DatasetEntry { id: 24, text: "Many digital nomads set up offshore bank accounts to optimize their tax liabilities while moving between different countries.".into(), ambiguity_level: "mildly_ambiguous".into(), primary_category: Some("finance_investing".into()), secondary_category: Some("travel_tourism".into()) },
        DatasetEntry { id: 25, text: "Budgeting for a year-long trip around the world requires calculating daily accommodation costs and saving thousands in advance.".into(), ambiguity_level: "mildly_ambiguous".into(), primary_category: Some("travel_tourism".into()), secondary_category: Some("finance_investing".into()) },
        DatasetEntry { id: 26, text: "The culinary tourism package includes guided street food tours and private cooking classes with local chefs in Tokyo.".into(), ambiguity_level: "mildly_ambiguous".into(), primary_category: Some("travel_tourism".into()), secondary_category: Some("food_cooking".into()) },
        DatasetEntry { id: 27, text: "Rising grain prices and supply chain disruptions are forcing local artisan bakeries to increase the cost of a sourdough loaf.".into(), ambiguity_level: "mildly_ambiguous".into(), primary_category: Some("food_cooking".into()), secondary_category: Some("finance_investing".into()) },
        DatasetEntry { id: 28, text: "The software company introduced a new subscription model for its cloud services, aiming to boost recurring software-as-a-service revenues.".into(), ambiguity_level: "mildly_ambiguous".into(), primary_category: Some("technology".into()), secondary_category: Some("finance_investing".into()) },
        DatasetEntry { id: 29, text: "Our amateur soccer club is traveling to Spain next month to participate in an international friendly tournament.".into(), ambiguity_level: "mildly_ambiguous".into(), primary_category: Some("sports_fitness".into()), secondary_category: Some("travel_tourism".into()) },
        DatasetEntry { id: 30, text: "Investing in premium sports memorabilia, like game-worn jerseys, has become a highly lucrative alternative asset class.".into(), ambiguity_level: "mildly_ambiguous".into(), primary_category: Some("finance_investing".into()), secondary_category: Some("sports_fitness".into()) },
        DatasetEntry { id: 31, text: "This article reviews the engineering behind elite running shoes, comparing the energy-return polymer plates with smart embedded pressure sensors.".into(), ambiguity_level: "highly_ambiguous".into(), primary_category: Some("technology".into()), secondary_category: Some("sports_fitness".into()) },
        DatasetEntry { id: 32, text: "Mobile banking apps are leveraging decentralized blockchain protocols and biometric authentication to secure financial transactions.".into(), ambiguity_level: "highly_ambiguous".into(), primary_category: Some("technology".into()), secondary_category: Some("finance_investing".into()) },
        DatasetEntry { id: 33, text: "A comprehensive guide to exploring the night markets of Taiwan, focusing on the history of regional street food and how to navigate the crowded stalls.".into(), ambiguity_level: "highly_ambiguous".into(), primary_category: Some("food_cooking".into()), secondary_category: Some("travel_tourism".into()) },
        DatasetEntry { id: 34, text: "An analysis of the global coffee bean futures market, discussing how climate change impacts crop yields and the final retail price of espresso.".into(), ambiguity_level: "highly_ambiguous".into(), primary_category: Some("food_cooking".into()), secondary_category: Some("finance_investing".into()) },
        DatasetEntry { id: 35, text: "Hiking the Pacific Crest Trail: A detailed breakdown of the physical conditioning required for high-altitude trekking and the logistics of navigating national parks.".into(), ambiguity_level: "highly_ambiguous".into(), primary_category: Some("sports_fitness".into()), secondary_category: Some("travel_tourism".into()) },
        DatasetEntry { id: 36, text: "A sports nutritionist's guide to meal prepping, detailing exactly what macro-nutrients to eat before high-intensity interval training to maximize muscle recovery.".into(), ambiguity_level: "highly_ambiguous".into(), primary_category: Some("sports_fitness".into()), secondary_category: Some("food_cooking".into()) },
        DatasetEntry { id: 37, text: "Agriculture technology is evolving rapidly, with automated indoor hydroponic systems using AI sensors to deliver nutrients to crops without soil.".into(), ambiguity_level: "highly_ambiguous".into(), primary_category: Some("technology".into()), secondary_category: Some("food_cooking".into()) },
        DatasetEntry { id: 38, text: "Analyzing the economic impact of international tourism on developing island nations, specifically tracking foreign currency exchange and hotel industry revenues.".into(), ambiguity_level: "highly_ambiguous".into(), primary_category: Some("finance_investing".into()), secondary_category: Some("travel_tourism".into()) },
        DatasetEntry { id: 39, text: "The chemical structure of DNA consists of two long chains of nucleotides twisted into a double helix.".into(), ambiguity_level: "out_of_scope".into(), primary_category: None, secondary_category: None },
        DatasetEntry { id: 40, text: "William Shakespeare's tragedy 'Hamlet' explores themes of revenge, madness, and moral corruption in the Danish court.".into(), ambiguity_level: "out_of_scope".into(), primary_category: None, secondary_category: None },
    ]
}

/// The "none of the above" choice appended so the model can reject all
/// categories.
pub const NONE_CHOICE: &str = "none_of_the_above";

/// Run the dataset through `classify()` and `generate()`, save to CSV.
///
/// Returns the path to the generated CSV file.
pub fn run_dataset_and_save_csv<B: LLMBackend>(
    classifier: &LLMClassifier<B>,
    backend_name: &str,
    llm_name: &str,
) -> std::io::Result<String> {
    let cats = categories();
    let entries = dataset();

    // Choices presented to the classifier: real categories + none_of_the_above
    let choices: Vec<String> = cats
        .iter()
        .cloned()
        .chain(std::iter::once(NONE_CHOICE.to_string()))
        .collect();

    let timestamp = chrono_like_timestamp();
    let csv_path = format!("{backend_name}_{timestamp}.csv");

    let mut file = std::fs::File::create(&csv_path)?;

    // Header
    let mut header = vec![
        "id", "text", "ambiguity_level", "primary_category", "secondary_category",
        "backend", "llm", "api", "prediction", "confidence",
    ];
    for c in &cats {
        header.push(Box::leak(format!("prob_{c}").into_boxed_str()));
    }
    header.push("prob_none");
    writeln!(file, "{}", header.join(","))?;

    for entry in &entries {
        for api_name in &["classify", "generate"] {
            let result = if *api_name == "classify" {
                classifier.classify(&entry.text, choices.clone(), None)
            } else {
                classifier.generate(&entry.text, choices.clone(), None, Some(1))
            };

            match result {
                Ok(r) => {
                    write_csv_row(
                        &mut file, entry, backend_name, llm_name, api_name, &r, &cats,
                    )?;
                }
                Err(e) => {
                    eprintln!("  Error on entry {} ({}): {e}", entry.id, api_name);
                }
            }
        }
    }

    println!("\n  CSV saved: {csv_path}");
    Ok(csv_path)
}

fn write_csv_row<W: Write>(
    file: &mut W,
    entry: &DatasetEntry,
    backend_name: &str,
    llm_name: &str,
    api_name: &str,
    result: &ClassificationResult,
    cats: &[String],
) -> std::io::Result<()> {
    let mut fields: Vec<String> = vec![
        entry.id.to_string(),
        csv_quote(&entry.text),
        entry.ambiguity_level.clone(),
        entry.primary_category.clone().unwrap_or_default(),
        entry.secondary_category.clone().unwrap_or_default(),
        backend_name.into(),
        llm_name.into(),
        api_name.into(),
        result.prediction.clone(),
        format!("{:.6}", result.confidence),
    ];

    for c in cats {
        let prob = result.probabilities.get(c).copied().unwrap_or(0.0);
        fields.push(format!("{prob:.6}"));
    }
    let prob_none = result.probabilities.get(NONE_CHOICE).copied().unwrap_or(0.0);
    fields.push(format!("{prob_none:.6}"));

    writeln!(file, "{}", fields.join(","))
}

fn csv_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

/// Generate a timestamp string like "20260712000000" without external deps.
fn chrono_like_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = now.as_secs();
    // Simple conversion (not accounting for leap seconds etc. — close enough for a filename)
    let days = secs / 86400;
    let remainder = secs % 86400;
    let h = remainder / 3600;
    let m = (remainder % 3600) / 60;
    let s = remainder % 60;
    // Days since 1970-01-01 → approximate Y-M-D
    let (year, month, day) = days_to_ymd(days as i64);
    format!("{year:04}{month:02}{day:02}{h:02}{m:02}{s:02}")
}

fn days_to_ymd(mut days: i64) -> (i64, i64, i64) {
    // Algorithm from Howard Hinnant: https://howardhinnant.github.io/date_algorithms.html
    days += 719468;
    let era = if days >= 0 { days } else { days - 146096 } / 146097;
    let doe = days - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (y + if m <= 2 { 1 } else { 0 }, m, d)
}
