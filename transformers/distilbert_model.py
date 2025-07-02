import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup,
    set_seed
)
from torch.optim import AdamW
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_analysis'))
from data_preprocessing import CyberbullyingPreprocessor


class CyberbullyingDataset(Dataset):
    """Custom Dataset for cyberbullying detection with BERT tokenization."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTCyberbullyingClassifier(nn.Module):
    """BERT-based cyberbullying classifier with custom head."""
    
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2, dropout_rate=0.3):
        super(BERTCyberbullyingClassifier, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Custom classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the classifier head."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # Shape: (batch_size, hidden_size)
        
        # Apply dropout and classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance."""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def load_and_prepare_transformer_data(test_size=0.2, val_size=0.1, random_state=42):
    """Load and prepare data for transformer models."""
    preprocessor = CyberbullyingPreprocessor()
    texts, labels = preprocessor.load_dataset()
    
    if not texts:
        raise RuntimeError("Dataset could not be loaded")
    
    # Clean texts
    cleaned_texts = [preprocessor.basic_text_cleaning(t) for t in texts]
    
    # Convert to numpy arrays
    texts_np = np.array(cleaned_texts)
    labels_np = np.array(labels, dtype=np.int32)
    
    # First split: train + val, test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts_np, labels_np, test_size=test_size, random_state=random_state, stratify=labels_np
    )
    
    # Second split: train, val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for the reduced dataset size
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"Dataset splits:")
    print(f"  Train: {len(X_train)} samples ({np.bincount(y_train)})")
    print(f"  Val:   {len(X_val)} samples ({np.bincount(y_val)})")
    print(f"  Test:  {len(X_test)} samples ({np.bincount(y_test)})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_weighted_sampler(labels):
    """Create a weighted sampler to handle class imbalance."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def train_transformer_model(
    model, train_loader, val_loader, 
    num_epochs=5, learning_rate=2e-5, 
    device='cpu', patience=3
):
    """Train the transformer model with early stopping."""
    
    # Loss function and optimizer
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    model.to(device)
    
    best_val_f1 = 0
    patience_counter = 0
    train_losses = []
    val_f1_scores = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_f1 = evaluate_transformer_model(model, val_loader, device, verbose=False)
        val_f1_scores.append(val_f1)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_cyberbullying_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_cyberbullying_model.pth'))
    return model, train_losses, val_f1_scores


def evaluate_transformer_model(model, data_loader, device, verbose=True):
    """Evaluate the transformer model."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    
    if verbose:
        print(f"\nTransformer Model Performance:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, 
                                    target_names=["Non-Cyberbullying", "Cyberbullying"]))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_predictions))
    
    return f1


def main():
    print("Starting Advanced Transformer Models for Cyberbullying Detection")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    set_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_transformer_data()
    
    # Initialize tokenizer and model
    MODEL_NAME = 'distilbert-base-uncased'  # Smaller, faster than BERT
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    
    print(f"Using model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = CyberbullyingDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = CyberbullyingDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = CyberbullyingDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    # Create data loaders with weighted sampling for training
    train_sampler = create_weighted_sampler(y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = BERTCyberbullyingClassifier(model_name=MODEL_NAME, num_classes=2)
    
    print(f"\nModel Architecture:")
    print(f"  Base Model: {MODEL_NAME}")
    print(f"  Hidden Size: {model.config.hidden_size}")
    print(f"  Custom Classification Head: {model.config.hidden_size} → 256 → 64 → 2")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    print(f"\n{'='*50}")
    print("Training Advanced Transformer Model...")
    print(f"{'='*50}")
    
    trained_model, train_losses, val_f1_scores = train_transformer_model(
        model, train_loader, val_loader,
        num_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        patience=3
    )
    
    # Final evaluation on test set
    print(f"\n{'='*50}")
    print("Final Evaluation on Test Set")
    print(f"{'='*50}")
    
    test_f1 = evaluate_transformer_model(trained_model, test_loader, device)
    
    # Comparison with previous results
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    comparison_data = {
        'Model': ['Logistic Regression', 'Original MLP', 'Original Bi-LSTM', 'DistilBERT'],
        'F1-Score': [0.432, 0.111, 0.111, test_f1],
        'Status': ['Traditional ML Best', 'Neural Network Baseline', 'Neural Network Baseline', 'Transformer Model']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Calculate improvement
    improvement = ((test_f1 - 0.432) / 0.432) * 100
    print(f"\nTransformer vs Traditional ML:")
    print(f"  F1-Score Improvement: {improvement:+.1f}%")
    
    if test_f1 > 0.432:
        print("🎉 SUCCESS: Transformer model outperforms traditional ML!")
    else:
        print("📊 Traditional ML still competitive - consider ensemble approaches")
    
    print(f"\nAdvanced Transformer Training Completed!")
    print("✅ DistilBERT fine-tuning")
    print("✅ Focal Loss for class imbalance") 
    print("✅ Weighted sampling")
    print("✅ Custom classification head")
    print("✅ Learning rate scheduling")
    print("✅ Early stopping with validation F1")


if __name__ == "__main__":
    main() 