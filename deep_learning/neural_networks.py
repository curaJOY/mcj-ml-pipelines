import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_analysis'))
from data_preprocessing import CyberbullyingPreprocessor


def load_and_prepare_data(max_words: int = 5000, max_len: int = 60, test_size: float = 0.2, random_state: int = 42):
    """Load raw texts & labels, perform basic cleaning, tokenize and pad sequences."""
    preprocessor = CyberbullyingPreprocessor()
    texts, labels = preprocessor.load_dataset()
    if not texts:
        raise RuntimeError("Dataset could not be loaded – aborting neural network training.")

    # Basic cleaning (URL/mentions/hashtags handling)
    cleaned_texts = [preprocessor.basic_text_cleaning(t) for t in texts]

    # Tokenization
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(cleaned_texts)
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    labels_np = np.array(labels, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels_np, test_size=test_size, random_state=random_state, stratify=labels_np
    )

    return tokenizer, X_train, X_test, y_train, y_test


def build_mlp_model(vocab_size: int, embed_dim: int = 64, max_len: int = 60) -> Sequential:
    """Simple MLP over embeddings (Embedding → Flatten → Dense)."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_lstm_model(vocab_size: int, embed_dim: int = 128, max_len: int = 60) -> Sequential:
    """Bidirectional LSTM model."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def evaluate_model(model: tf.keras.Model, X_test, y_test, model_name: str):
    """Evaluate the trained model on test data and print metrics."""
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"   Accuracy : {acc:.3f}")
    print(f"   Precision: {prec:.3f}")
    print(f"   Recall   : {rec:.3f}")
    print(f"   F1-Score : {f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Cyberbullying", "Cyberbullying"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}



def main():
    print("Starting Neural Network Models for Cyberbullying Detection (MLP & LSTM)")
    print("=" * 70)

    MAX_WORDS = 5000  # Vocabulary size for tokenizer
    MAX_LEN = 60      # Max sequence length (padded / truncated)

    tokenizer, X_train, X_test, y_train, y_test = load_and_prepare_data(
        max_words=MAX_WORDS, max_len=MAX_LEN
    )
    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # --- MLP MODEL ---
    mlp_model = build_mlp_model(vocab_size=vocab_size, max_len=MAX_LEN)
    print("\nTraining MLP model…")
    mlp_model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2
    )
    mlp_metrics = evaluate_model(mlp_model, X_test, y_test, "MLP")

    # --- LSTM MODEL ---
    lstm_model = build_lstm_model(vocab_size=vocab_size, max_len=MAX_LEN)
    print("\nTraining Bidirectional LSTM model…")
    lstm_model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2
    )
    lstm_metrics = evaluate_model(lstm_model, X_test, y_test, "Bidirectional LSTM")

    print("\nSummary of Neural Network Models:")
    results_df = pd.DataFrame([mlp_metrics, lstm_metrics], index=["MLP", "Bi-LSTM"])
    print(results_df)

    print("\nNeural network training completed – models ready for evaluation and comparison.")


if __name__ == "__main__":
    main() 