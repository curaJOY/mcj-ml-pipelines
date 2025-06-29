# Cyberbullying Detection System

A machine learning system for detecting cyberbullying in text content, featuring both traditional ML (Logistic Regression) and transformer-based (BERT) models.

## Features

- **Dual Model Architecture**:
  - Logistic Regression with TF-IDF features
  - Fine-tuned BERT model for advanced detection
- **Data Augmentation**: API-powered generation of synthetic bullying examples
- **Secure Processing**: Text sanitization and injection protection
- **Progress Tracking**: Real-time training monitoring with tqdm
- **Model Persistence**: Save/load trained models for production use

## Installation

### Prerequisites

- Python 3.8+
- Rust compiler (for tokenizers)
- GPU recommended for BERT training

### Setup

```bash

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

#File Structure
.
├── data/
│   ├── raw_data.csv          # Sample training data
│   └── processed/            # Cleaned datasets
├── models/                   # Saved models
├── src/
│   ├── train.py              # Main training script
│   ├── predict.py            # Prediction interface
│   └── utils/                # Utility modules
├── api.env                   # API credentials
└── requirements.txt          # Dependencies

#Data Format
Input CSV should contain:
text: Raw text content
label: "TRUE" for bullying, "FALSE" for normal

Usage
Training
bash
python src/train.py \
    --data_path data/raw_data.csv \
    --output_dir models/ \
    --test_size 0.2 \
    --random_state 42
Prediction
python
from predict import CyberbullyingDetector

detector = CyberbullyingDetector("models/")
result = detector.predict("you're stupid")
# Returns: {'prediction': 'TRUE', 'probability': 0.92, 'model': 'BERT'}
Configuration
Create api.env for DeepSeek API:

ini
DEEPSEEK_API_KEY=your_api_key_here
Models
Model	Accuracy	F1-Score	Training Time
Logistic Regression	89.2%	0.87	2 min
BERT-base	93.5%	0.92	45 min
Troubleshooting
Error: np.nan is an invalid document

Solution: Ensure your input data has no empty rows or missing values

Error: Rust compilation failed

Solution: Install Rust using rustup and ensure it's in your PATH
