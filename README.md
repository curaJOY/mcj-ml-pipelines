# CuraJOY Cyberbullying Detection Project

This project uses machine learning to detect cyberbullying in online text.

## 🔍 Problem Statement
Cyberbullying is a serious issue affecting millions globally. The goal of this project is to create a text classification model that can identify whether a given social media post is bullying or not.

## 📁 Dataset
The dataset contains annotated social media posts labeled as "Yes" (bullying) or "No" (not bullying). The dataset was manually expanded and balanced for improved performance.

- `annotations.csv`: Labeled dataset
- Additional samples collected from platforms like TikTok, Reddit, and Instagram

## 🧠 Model & Approach

- **Model**: RandomForestClassifier (with TF-IDF vectorization)
- **Balancing**: Resampling minority class to balance the dataset
- **Rule-Based Layer**: Additional keyword-based filter for edge cases
- **Accuracy**: 66–76% on balanced test set
- **Precision/Recall**: Tuned with thresholds and tested against real-world bullying phrases

## 🛠️ Files

- `app.py`: Streamlit web app interface
- `train_model.py`: Script to train the ML model
- `annotator_interface.py`: Interface for labeling posts
- `append_new_samples.py`: Adds new samples to dataset
- `bullying_model.pkl` / `tfidf_vectorizer.pkl`: Saved model files
- `requirements.txt`: List of Python packages used

## 🚀 How to Run

1. Clone this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
python train_model.py
streamlit run app.py

