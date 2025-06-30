# mcj-ml-pipelines
Repository to compile Data/ML related stuff, we might de-structure this repo in future if it's become too bulky or multi-purpose.

# Cyberbullying Detection & AI Coach Project

This project is for detecting cyberbullying in social media posts and testing AI-driven interventions aimed at reducing harmful behavior among teens.

---

## 📌 Part 1: Data Preparation & Annotation

### 1.1. Clean and Preprocess Raw Dataset

- **File:** `model_dev.ipynb`
- **Steps:**
  - Pulls dataset from GitHub (`dataset.txt`)
  - Extracts text-label pairs (bullying or not)
  - Cleans text (lowercase, removes punctuation, whitespace)
  - Converts labels to binary (True → 1, False → 0)
  - Uses TF-IDF and Logistic Regression to train a simple classifier
  - Evaluates with classification report and confusion matrix
  - Saves cleaned posts into `cleaned_posts.json` for annotation

### 1.2. Web Annotation Interface

- **File:** `annotator_app.py`
- **How to run:**
  ```bash
  streamlit run annotator_app.py
  ```
- **Features:**
  - Loads 25 posts from `cleaned_posts.json`
  - Allows students to label posts for bullying, self-harm, severity
  - Confidence slider and notes field
  - Annotations saved into `annotations.json`
  - Cohen’s Kappa used to calculate inter-annotator agreement

### 1.3. Fix and View Annotations

- **File:** `fix_annotations.py`
- **Usage:**
  ```bash
  python fix_annotations.py
  ```
- Converts raw `annotations.json` (newline JSON) into valid JSON array → `annotations_viewable.json`

---

## 📌 Part 2: Research Design & Bias Analysis

- **File:** `research_design.py`
- **Functions:**
  - `design_validation_study()`: Outlines an A/B/C test research plan with ethics, metrics, and sample size.
  - `analyze_algorithm_bias(predictions, demographics)`: Analyzes fairness across race, gender, age, and platform using accuracy, precision, and recall.

---

## 📌 Part 3: Data Quality & Edge Cases

- **File:** `data_quality_and_edge_cases.py`
- **Functions:**
  - `detect_annotation_quality_issues()`: Detects fatigue, bias, low variation in labeling, and systematic disagreement.
  - `handle_edge_cases()`: Provides strict guidelines on sarcasm, cultural misinterpretation, accessibility, and language variation.

---

## 📌 Part 4: Intervention Design

- **File:** `intervention_design.py`
- **Functions:**
  - `design_intervention_messages()`: Generates compassionate messages for bullying and self-harm cases.
  - `measure_intervention_effectiveness()`: Tracks success with short- and long-term metrics, including tone shift and positive engagement. Addresses ethical safeguards and privacy.

---

## 📁 Output Files

| File Name               | Description                                   |
|------------------------|-----------------------------------------------|
| `cleaned_posts.json`   | Cleaned posts used for annotation             |
| `annotations.json`     | Raw user annotations (NDJSON format)          |
| `annotations_viewable.json` | Converted annotations as JSON array     |
| `model_dev.ipynb`      | Main notebook for training and exporting data |
| `annotator_app.py`     | Streamlit annotation UI                       |
| `fix_annotations.py`   | JSON formatting script                        |
| `research_design.py`   | Study design and bias detection               |
| `data_quality_and_edge_cases.py` | Quality checks and edge handling  |
| `intervention_design.py` | Message and metric logic                    |

---

---

## ⚖️ Ethics & Privacy

- No personal data is stored or linked.
- All analysis uses anonymous or hashed IDs.
- Evaluation thresholds designed with mental health expert consultation.

---

## 📌 Notes

- All results are simulated using sample data.
- Expandable to real-world data and scalable annotation systems.

---
