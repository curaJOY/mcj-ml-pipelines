# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

def train_triage_model():
    """
    Loads processed data, trains a TF-IDF + Logistic Regression model,
    evaluates it, and saves the final pipeline.
    """
    # --- 1. Load Data ---
    print("Loading processed data...")
    try:
        df = pd.read_csv('data/cleaned_data.csv')
    except FileNotFoundError:
        print("Error: `data/cleaned_data.csv` not found.")
        print("Please run `src/preprocess.py` first.")
        return

    # Handle potential missing values in the target column
    df.dropna(subset=['bullying_label', 'processed_text'], inplace=True)
    
    X = df['processed_text']
    y = df['bullying_label'] # Assuming the label column is named 'bullying_label'

    # --- 2. Split Data ---
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 3. Define and Train a Pipeline ---
    print("Training the Triage Model (TF-IDF + Logistic Regression)...")
    
    # The pipeline bundles preprocessing (vectorization) and the estimator
    triage_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
    ])
    
    triage_pipeline.fit(X_train, y_train)

    # --- 4. Evaluate the Model ---
    print("\n--- Model Evaluation ---")
    y_pred = triage_pipeline.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Bullying', 'Bullying']))
    
    # --- 5. Save the Model ---
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'triage_model_pipeline.joblib')
    joblib.dump(triage_pipeline, model_path)
    print(f"\nModel saved successfully to `{model_path}`")

if __name__ == "__main__":
    train_triage_model()