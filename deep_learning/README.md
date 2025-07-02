# Deep Learning Neural Networks Approach

## 📊 Overview
This folder contains neural network implementations for cyberbullying detection, including Multi-Layer Perceptrons (MLPs) and Bidirectional LSTMs. While theoretically more powerful, these models underperformed compared to traditional ML due to dataset limitations.

## 🎯 Methodology

### Neural Network Architectures

#### 1. Multi-Layer Perceptron (MLP)
- **Architecture**: Embedding → Flatten → Dense(128) → Dense(64) → Dense(32) → Output
- **Regularization**: Dropout layers (0.5, 0.3, 0.2)
- **Activation**: ReLU for hidden layers, Sigmoid for output

#### 2. Bidirectional LSTM
- **Architecture**: Embedding → Bi-LSTM(128) → Bi-LSTM(64) → Dense(64) → Dense(32) → Output
- **Regularization**: Dropout and recurrent dropout (0.3 each)
- **Sequence Processing**: Bidirectional for context understanding

### Technical Implementation
- **Vocabulary Size**: 5,000 most frequent words
- **Sequence Length**: 60 tokens (padded/truncated)
- **Embedding Dimension**: 64 (MLP), 128 (LSTM)
- **Loss Function**: Focal Loss (α=0.25, γ=2.0) for class imbalance
- **Optimizer**: Adam with learning rate 1e-3

### Class Imbalance Handling
- **Focal Loss**: Specialized loss function for imbalanced datasets
- **Class Weights**: Balanced weights applied during training
- **Threshold Optimization**: Post-training threshold tuning for F1 maximization

## 📉 Results

### Performance Summary
| Model | F1-Score | Accuracy | Status | Key Issues |
|-------|----------|----------|--------|------------|
| **Improved MLP** | 0.111 | - | Underperformed | Insufficient training data |
| **Improved Bi-LSTM** | 0.111 | - | Underperformed | Sequential patterns not critical |
| **Original MLP** | 0.111 | - | Baseline | Over-architecture for task |
| **Original Bi-LSTM** | 0.111 | - | Baseline | Limited sequence understanding |

### Comparison to Best Traditional ML
- **Traditional ML (Logistic Regression)**: F1 = 0.432
- **Deep Learning Models**: F1 = 0.111
- **Performance Gap**: Deep learning is **74% worse** than traditional ML

## ❌ Why Deep Learning Failed

### 1. Insufficient Training Data
- **Dataset Size**: Only 674 samples total
- **Training Set**: ~470 samples after splits
- **Rule of Thumb**: Neural networks need 10,000+ samples minimum
- **Result**: Models couldn't learn meaningful patterns

### 2. Task Complexity Mismatch
- **Cyberbullying Detection**: Primarily keyword/pattern-based
- **Neural Networks**: Designed for complex semantic understanding
- **Actual Need**: Simple offensive word detection (coefficient: 1.705)
- **Conclusion**: Over-engineered solution for linear separable problem

### 3. Limited Sequential Patterns
- **LSTM Assumption**: Sequential dependencies are important
- **Reality**: Text length ~13-14 words, mostly independent patterns
- **Key Features**: Single words ("liberal", "bitch", "side") not sequences
- **Result**: LSTM advantages not applicable to this task

### 4. Embedding Learning Challenges
- **Requirement**: Large vocabulary needs substantial data
- **Reality**: 5,000 vocabulary with 674 samples
- **Problem**: Can't learn meaningful word representations
- **Traditional ML Advantage**: Uses pre-engineered features

### 5. Overfitting Despite Regularization
- **Multiple Dropout Layers**: 0.5, 0.3, 0.2 dropout rates
- **Early Stopping**: Implemented with patience
- **Class Weights**: Balanced for minority class
- **Still Failed**: Fundamental data size limitation

## 🔬 Technical Analysis

### Model Architecture Issues
```python
# Over-complex for the task:
Embedding(5000, 128) → Bi-LSTM(128) → Bi-LSTM(64) → Dense(64) → Dense(32)

# vs. What actually works:
LinearModel(1018_engineered_features) → LogisticRegression
```

### Loss Function Analysis
- **Focal Loss**: Good in theory for imbalance
- **Reality**: Added complexity without benefit
- **Traditional ML**: Simple balanced class weights more effective

### Threshold Optimization
- **Post-training tuning**: Attempted to maximize F1
- **Limited Impact**: Underlying model too weak
- **Traditional ML**: Natural threshold already optimal

## 💡 Key Learnings

### When Neural Networks Fail
1. **Small Datasets**: <10,000 samples favor traditional ML
2. **Simple Tasks**: Linear separable problems don't need complexity
3. **Feature Engineering**: Well-engineered features > raw embeddings
4. **Class Imbalance**: Simple balancing often better than complex losses

### Dataset-Specific Insights
- **Text Characteristics**: Short (~13 words), pattern-based
- **Key Signals**: Offensive words, political terms
- **No Complex Semantics**: Sarcasm/context not in current dataset
- **Linear Separability**: Traditional ML sufficient

## 🚀 Usage

```python
from improved_neural_networks import build_improved_mlp_model, build_improved_lstm_model

# Load and prepare data
tokenizer, X_train, X_test, y_train, y_test, class_weights = load_and_prepare_data()

# Build and train MLP
mlp_model = build_improved_mlp_model(vocab_size=5000, max_len=60)
mlp_model.fit(X_train, y_train, class_weight=class_weights, epochs=30)

# Build and train LSTM
lstm_model = build_improved_lstm_model(vocab_size=5000, max_len=60)
lstm_model.fit(X_train, y_train, class_weight=class_weights, epochs=30)
```

## 📁 Files
- `neural_networks.py` - Basic MLP and LSTM implementations
- `improved_neural_networks.py` - Enhanced models with focal loss and regularization
- `README.md` - This documentation

## 🎯 When Deep Learning Would Work

### Required Conditions
1. **Larger Dataset**: 10,000+ samples minimum
2. **Complex Semantics**: Context-dependent understanding needed
3. **Sequential Patterns**: Long-range dependencies important
4. **Balanced Classes**: Or sophisticated imbalance handling

### Alternative Approaches
1. **Pre-trained Embeddings**: Word2Vec, GloVe for better representations
2. **Transfer Learning**: Fine-tune pre-trained language models
3. **Data Augmentation**: Synthetic sample generation
4. **Ensemble Methods**: Combine with traditional ML

## 🔄 Next Steps
For neural networks to be viable:
1. **Collect More Data**: 10x current dataset size
2. **Focus on Context**: Conversation threads, user history
3. **Advanced Architectures**: Attention mechanisms, transformers
4. **Hybrid Approaches**: Combine with traditional ML strengths 