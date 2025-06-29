import pandas as pd
import re
from typing import List

# A simple mapping for crucial emojis
EMOJI_MAP = {
    "😊": " _positive_emoji_ ",
    "🙂": " _positive_emoji_ ",
    "😂": " _laughing_emoji_ ",
    "🤣": " _laughing_emoji_ ",
    "💀": " _skull_emoji_ ", # Often used for intense laughter
    "😠": " _angry_emoji_ ",
    "😡": " _angry_emoji_ ",
    "😭": " _crying_emoji_ ",
    "😢": " _crying_emoji_ ",
}

def clean_text(text: str) -> str:
    """
    Applies basic text cleaning:
    - Removes URLs
    - Removes user mentions (e.g., @username)
    - Converts to lowercase
    - Removes special characters except for basic punctuation
    """
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,?!_]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_emojis(text: str) -> str:
    """
    Replaces common emojis with descriptive text tokens.
    """
    for emoji, token in EMOJI_MAP.items():
        text = text.replace(emoji, token)
    return text

def preprocess_pipeline(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Runs the full preprocessing pipeline on a DataFrame.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()
    
    # Apply emoji normalization first to preserve them
    df_processed['processed_text'] = df_processed[text_column].apply(normalize_emojis)
    
    # Then apply text cleaning
    df_processed['processed_text'] = df_processed['processed_text'].apply(clean_text)
    
    return df_processed

if __name__ == '__main__':
    # Example usage if you run this script directly
    print("Loading raw data...")
    try:
        df = pd.read_csv('data/cyberbullying_data.csv')
    except FileNotFoundError:
        print("Error: `data/cyberbullying_data.csv` not found.")
        print("Please place the dataset in the correct directory.")
        exit()

    print("Running preprocessing pipeline...")
    df_processed = preprocess_pipeline(df, text_column='text')

    # Save the processed data
    df_processed.to_csv('data/cleaned_data.csv', index=False)
    print("Preprocessing complete. Cleaned data saved to `data/processed/cleaned_data.csv`")
    
    print("\n--- Preprocessing Example ---")
    original_text = "Hope you have a great day at your new school! 😊 (Just kidding, @someuser everyone there will hate you too)"
    processed_text = clean_text(normalize_emojis(original_text))
    print(f"Original: {original_text}")
    print(f"Processed: {processed_text}")