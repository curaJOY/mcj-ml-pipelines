# Cyberbullying-detection

# CuraJOY Cyberbullying Detection Challenge Submission

**Rose Njuguna**

CuraJOY Impact Fellowship Program's Cyberbullying Detection Challenge.

---

### Table of Contents

1.  [Solution Overview](#solution-overview)
2.  [Repository Structure](#repository-structure)
3.  [Setup and Installation](#setup-and-installation)
4.  [How to Run](#how-to-run)
5.  [Model Performance Summary](#model-performance-summary)
6.  [Fellowship Challenge Parts](#fellowship-challenge-parts)

---

### Solution Overview

My approach is a hybrid, multi-stage architecture designed for both accuracy and efficiency. It consists of:

1.  A lightweight **Logistic Regression Triage Model** to filter obvious cases.
2.  A **DistilBERT-based Deep Learning Model** for nuanced, contextual analysis.
3.  An **LLM-based Agent** for providing reasoned classifications on the most ambiguous edge cases.

This approach prioritizes minimizing false positives while maintaining high recall for genuine cyberbullying incidents. For a full breakdown, please see the attached `submission_report.pdf`.

---

### Repository Structure

```
.
├── README.md
├── requirements.txt
├── submission_report.pdf
├── data/
├── notebooks/
├── src/
└── fellowship_challenge_parts/
```

- **`submission_report.pdf`**: The detailed 5-page report outlining the methodology, findings, and ethical considerations.
- **`src/`**: Contains the final, cleaned Python scripts for the ML pipeline (preprocessing, training, prediction).
- **`notebooks/`**: Jupyter notebooks detailing the exploratory data analysis and model development process.
- **`fellowship_challenge_parts/`**: Contains the Python solutions to the four timed fellowship challenge questions, including the Streamlit annotation app and research design documents.

---

### Setup and Installation

To reproduce this project, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd curajoy-challenge
    ```
2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare Data**: Place the provided dataset `cyberbullying_data.csv` into the `data` directory.

---

### How to Run

1.  **To run the full training pipeline:**

    ```bash
    python src/train_model.py
    ```

    This will preprocess the data from `data`, save the cleaned data, train the model, and save the final model artifact.

2.  **To explore the development process:**
    Open and run the Jupyter Notebooks in the `notebooks/` directory.

3.  **To launch the Annotation Tool (Part 1):**
    ```bash
    streamlit run challenge_parts/part_1_annotation.py
    ```

---

### Model Performance Summary

| Metric    | Score | Notes                                       |
| --------- | ----- | ------------------------------------------- |
| Precision | 0.95  | High precision to minimize false positives. |
| Recall    | 0.92  | High recall to catch most bullying cases.   |
| F1-Score  | 0.93  | Balanced performance.                       |

_Full evaluation details and a confusion matrix are available in the `submission_report.pdf`._

---

### Fellowship Challenge Parts

The `challenge_parts/` directory contains my responses to the four timed sections of the challenge. These are implemented as "documentation-as-code" in Python files, providing clear, structured designs for:

- A Streamlit-based data annotation tool.
- An A/B test research design for intervention validation.
- Functions for data quality and bias analysis.
- Templates for compassionate AI intervention messages.
