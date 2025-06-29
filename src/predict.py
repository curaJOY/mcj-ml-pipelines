import joblib
import argparse
from preprocess import clean_text, normalize_emojis
import os

def predict(text: str):
    """
    Loads the trained model and predicts if the input text is cyberbullying.
    
    Args:
        text (str): The input text to classify.
        
    Returns:
        A tuple containing the predicted label (0 or 1) and the confidence score.
    """
    model_path = 'models/triage_model_pipeline.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run `src/train.py` first.")

    # Load the trained pipeline
    pipeline = joblib.load(model_path)
    
    # Preprocess the input text using the same functions
    processed_text = clean_text(normalize_emojis(text))
    
    # The pipeline handles TF-IDF transformation automatically
    # We need to pass the text inside a list or iterable
    prediction = pipeline.predict([processed_text])[0]
    confidence_scores = pipeline.predict_proba([processed_text])[0]
    
    # Get the confidence of the predicted class
    confidence = confidence_scores[prediction]
    
    return prediction, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cyberbullying Detection CLI")
    parser.add_argument("text", type=str, help="The text to classify.")
    
    args = parser.parse_args()
    
    try:
        prediction, confidence = predict(args.text)
        label_map = {0: "Not Cyberbullying", 1: "Cyberbullying"}
        
        print(f"\nInput Text: '{args.text}'")
        print("---")
        print(f"Prediction: {label_map[prediction]}")
        print(f"Confidence: {confidence:.2f}")

    except FileNotFoundError as e:
        print(f"Error: {e}")