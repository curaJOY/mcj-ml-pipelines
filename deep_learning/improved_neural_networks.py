import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_analysis'))
from data_preprocessing import CyberbullyingPreprocessor


def focal_loss(gamma=2., alpha=0.25):
    """Focal Loss implementation for handling class imbalance."""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        focal_loss = weight * cross_entropy
        return K.mean(focal_loss)
    return focal_loss_fixed


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

    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"Class weights: {class_weight_dict}")
    print(f"Training set distribution: {np.bincount(y_train)}")

    return tokenizer, X_train, X_test, y_train, y_test, class_weight_dict


def build_improved_mlp_model(vocab_size: int, embed_dim: int = 64, max_len: int = 60) -> Sequential:
    """Improved MLP with better regularization and architecture."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    
    # Use focal loss to handle imbalance
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"]
    )
    return model


def build_improved_lstm_model(vocab_size: int, embed_dim: int = 128, max_len: int = 60) -> Sequential:
    """Improved Bidirectional LSTM with better architecture."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len),
        Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)),
        Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=False)),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    
    # Use focal loss to handle imbalance
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"]
    )
    return model


def find_optimal_threshold(model: tf.keras.Model, X_val, y_val):
    """Find optimal threshold that maximizes F1-score."""
    y_pred_prob = model.predict(X_val, verbose=0).ravel()
    
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_pred_prob >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
    return best_threshold


def evaluate_model_with_threshold(model: tf.keras.Model, X_test, y_test, model_name: str, threshold: float = 0.5):
    """Evaluate the trained model with custom threshold."""
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{model_name} Performance (threshold={threshold:.3f}):")
    print(f"   Accuracy : {acc:.3f}")
    print(f"   Precision: {prec:.3f}")
    print(f"   Recall   : {rec:.3f}")
    print(f"   F1-Score : {f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Cyberbullying", "Cyberbullying"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "threshold": threshold}


def main():
    print("Starting Improved Neural Networks with Class Imbalance Handling")
    print("=" * 70)

    MAX_WORDS = 5000  # Vocabulary size for tokenizer
    MAX_LEN = 60      # Max sequence length (padded / truncated)

    tokenizer, X_train, X_test, y_train, y_test, class_weight_dict = load_and_prepare_data(
        max_words=MAX_WORDS, max_len=MAX_LEN
    )
    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)

    # Split training data to get validation set for threshold optimization
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # Early stopping on F1-score would be ideal, but we'll use val_loss with patience
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    results = []

    # --- IMPROVED MLP MODEL ---
    print("\n" + "="*50)
    print("Training Improved MLP model with Focal Loss...")
    mlp_model = build_improved_mlp_model(vocab_size=vocab_size, max_len=MAX_LEN)
    
    mlp_history = mlp_model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        class_weight=class_weight_dict,  # Apply class weights
        callbacks=[early_stop],
        verbose=1
    )
    
    # Find optimal threshold
    optimal_threshold_mlp = find_optimal_threshold(mlp_model, X_val, y_val)
    mlp_metrics = evaluate_model_with_threshold(mlp_model, X_test, y_test, "Improved MLP", optimal_threshold_mlp)
    results.append(mlp_metrics)

    # --- IMPROVED LSTM MODEL ---
    print("\n" + "="*50)
    print("Training Improved Bidirectional LSTM with Focal Loss...")
    lstm_model = build_improved_lstm_model(vocab_size=vocab_size, max_len=MAX_LEN)
    
    lstm_history = lstm_model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        class_weight=class_weight_dict,  # Apply class weights
        callbacks=[early_stop],
        verbose=1
    )
    
    # Find optimal threshold
    optimal_threshold_lstm = find_optimal_threshold(lstm_model, X_val, y_val)
    lstm_metrics = evaluate_model_with_threshold(lstm_model, X_test, y_test, "Improved Bi-LSTM", optimal_threshold_lstm)
    results.append(lstm_metrics)

    # --- COMPARISON WITH ORIGINAL ---
    print("\n" + "="*70)
    print("COMPARISON: Improved Deep Learning vs Traditional ML")
    print("="*70)
    
    results_df = pd.DataFrame(results, index=["Improved MLP", "Improved Bi-LSTM"])
    print(results_df)
    
    print(f"\nTraditional ML Baseline (Logistic Regression):")
    print(f"   F1-Score: 0.432")
    print(f"   Accuracy: 84.4%")
    print(f"   Precision: 40.0%")
    print(f"   Recall: 47.1%")

    print("\nImproved Neural Network Training completed!")
    print("Key improvements implemented:")
    print("✅ Focal Loss for class imbalance")
    print("✅ Class weights during training")
    print("✅ Optimal threshold tuning")
    print("✅ Better model architectures")
    print("✅ Enhanced regularization")


if __name__ == "__main__":
    main() 