# mcj-ml-pipelines
Repository to compile Data/ML related stuff, we might de-structure this repo in future if it's become too bulky or multi-purpose.

# Cyberbullying Detection & AI Coach Project

This project is for detecting cyberbullying in social media posts and testing AI-driven interventions aimed at reducing harmful behavior among teens.

---

## 🔹 Project Structure

| File | Description |
|------|-------------|
| `annotator_app.py` | Streamlit interface for annotators to label posts (bullying or not) - part 1|
| `annotations.json` | Collected raw annotations from users - part 1 |
| `annotations_viewable.json` | Viewable/readable version of annotations - part 1 |
| `cleaned_posts.json` | Input dataset with preprocessed social media posts part 1 |
| `fix_annotations.py` | Script to clean and validate annotation data part 1 |
| `research_design.py` | Experimental framework for A/B testing the AI coach - part 2|
| `data_quality_and_edge_cases.py` | Tools for detecting annotation issues and creating labeling guidelines part 3|
| `intervention_design.py` | Design and evaluation metrics for AI intervention messages - part 4 |
| `model_dev.ipynb` | Notebook for training/testing machine learning models for part 1 |
| `README.md` | Project overview and documentation |

---

## 🧠 Project Overview

### Part 1: Annotation Interface & Cleanup
- Developed a labeling app using Streamlit.
- Annotators label social media posts with optional comments.
- Metadata cleaned and normalized for analysis.

### Part 2: Research Design & Algorithm Bias
- Designed an A/B/C test to evaluate impact of intervention timing.
- Bias audit implemented across gender, race, age, and platform dimensions.

### Part 3: Data Quality & Edge Cases
- Tools to detect annotator fatigue, random labeling, and annotation disagreements.
- Defined culturally sensitive, inclusive annotation guidelines for sarcasm, slang, and multilingual posts.

### Part 4: AI Coach & Effectiveness Metrics
- Developed compassionate, age-appropriate intervention templates.
- Defined measurable short-term and long-term impact metrics with privacy and ethical safeguards.

---

## ✅ How to Run

1. **Annotation App**  
   ```bash
   streamlit run annotator_app.py
   ```

2. **Fix Annotations**  
   ```bash
   python fix_annotations.py
   ```

3. **Run Bias Analysis**  
   Check `research_design.py` for `analyze_algorithm_bias()` function.

4. **Detect Label Quality Issues**  
   Use `data_quality_and_edge_cases.py`.

5. **Test Intervention Metrics**  
   See `intervention_design.py` for structure and sample interventions.

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
