import pandas as pd
import numpy as np
import re
import os
import time
import io
import chardet
import joblib
import torch
import requests
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from importlib.metadata import version
from tqdm import tqdm  # Added for progress bars

# Version validation
TRANSFORMERS_VERSION = version('transformers')
if int(TRANSFORMERS_VERSION.split('.')[0]) < 4:
    raise ImportError("transformers v4.0+ required")

# NLP and ML imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EvalPrediction
)
from torch.utils.data import Dataset, DataLoader

# Security imports
from dotenv import load_dotenv
from requests.exceptions import RequestException

# Initialize NLTK data with progress
print("Initializing NLTK data...")
nltk_resources = ['corpora/stopwords', 'corpora/wordnet', 'corpora/omw-1.4']
for resource in tqdm(nltk_resources, desc="NLTK Resources"):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], quiet=True)

class SecureAPIHandler:
    def __init__(self):
        self._validate_environment()
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self._validate_key()
        self.last_request_time = 0
        self.request_interval = 1.0
        self.max_retries = 3
        self.timeout = 30  # seconds
        
    def _validate_environment(self):
        """Validate all required environment variables"""
        env_path = Path(__file__).parent / 'api.env'
        if not env_path.exists():
            raise FileNotFoundError(f"Missing API configuration: {env_path}")
        
        load_dotenv(env_path, override=True)
        
        if 'DEEPSEEK_API_KEY' not in os.environ:
            raise ValueError("DEEPSEEK_API_KEY not found in environment")
    
    def _validate_key(self):
        """Validate API key structure"""
        if not self.api_key or not isinstance(self.api_key, str):
            raise ValueError(
                "Invalid API key configuration. Key must be a non-empty string."
            )
        
        if not self.api_key.startswith('sk-'):
            raise ValueError(
                "API key must start with 'sk-'. "
                "Please check your api.env file."
            )
        
        if len(self.api_key) < 20:
            raise ValueError(
                "API key appears too short. "
                "Please verify your key is complete."
            )
    
    def make_api_request(self, payload: Dict) -> Dict:
        """Make secure API request with rate limiting and retries"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "CyberbullyingDetection/1.0"
        }
        
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        
        for attempt in tqdm(range(self.max_retries), desc="API Attempts", leave=False):
            try:
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Validate response structure
                response_json = response.json()
                if not all(k in response_json for k in ['choices', 'usage']):
                    raise ValueError("Invalid API response structure")
                
                self.last_request_time = time.time()
                return response_json
                
            except RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"API request failed after {self.max_retries} attempts: {str(e)}"
                    ) from e
                time.sleep(2 ** attempt)  # Exponential backoff

class DataProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.api_handler = SecureAPIHandler()
        self._toxic_keywords = {
            'kill', 'die', 'hate', 'ugly', 'stupid', 'worthless'
        }
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Robust data loading with enhanced NaN handling"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        print("\nLoading data...")
        # Detect file encoding
        with open(filepath, 'rb') as f:
            raw_data = f.read(10000)
            encoding = chardet.detect(raw_data)['encoding']
        
        # Try multiple encodings
        encodings_to_try = [encoding, 'utf-8', 'latin1', 'cp1252'] if encoding else ['utf-8', 'latin1']
        
        for enc in tqdm(encodings_to_try, desc="Trying encodings"):
            try:
                df = pd.read_csv(
                    filepath,
                    encoding=enc,
                    on_bad_lines='warn',
                    dtype={'text': str, 'label': str}
                )
                
                # Validate dataframe structure
                required_columns = {'text', 'label'}
                if not required_columns.issubset(df.columns):
                    raise ValueError(f"Data must contain columns: {required_columns}")
                
                # Enhanced cleaning with progress
                print("Cleaning data...")
                with tqdm(total=6, desc="Cleaning steps") as pbar:
                    # Convert all text to string first
                    df['text'] = df['text'].astype(str)
                    pbar.update(1)
                    
                    df = df.dropna(subset=['text', 'label'])
                    pbar.update(1)
                    
                    # Remove empty strings after conversion
                    df = df[df['text'].str.strip().astype(bool)]
                    pbar.update(1)
                    
                    df['label'] = df['label'].str.upper().map({'TRUE': 'TRUE', 'FALSE': 'FALSE'})
                    pbar.update(1)
                    
                    df = df[df['label'].isin(['TRUE', 'FALSE'])]
                    pbar.update(1)
                    
                    # Ensure no NaN slipped through
                    df = df[df['text'].notna() & df['label'].notna()]
                    pbar.update(1)
                
                # Preprocess text with progress bar
                print("Preprocessing text...")
                tqdm.pandas(desc="Text preprocessing")
                df['processed_text'] = df['text'].progress_apply(self.preprocess_text)
                
                # Final validation
                df = df[df['processed_text'].str.len() > 0]
                
                if df.empty:
                    raise ValueError("No valid data remaining after cleaning")
                
                print(f"\nSuccessfully loaded {len(df)} records")
                return df
                
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                continue
        
        raise ValueError(f"Failed to load {filepath} with tried encodings: {encodings_to_try}")

    def preprocess_text(self, text: str) -> str:
        """More robust text preprocessing with NaN handling"""
        try:
            # Force conversion to string (handles numpy.nan, None, etc.)
            text = str(text)
            
            # Early return for empty strings
            if not text.strip():
                return ""
            
            # Security and normalization
            text = re.sub(r'[^\w\s]', '', text.lower().strip())
            text = text.replace('\n', ' ').replace('\r', '')
            
            # Token processing
            tokens = text.split()
            tokens = [
                self.lemmatizer.lemmatize(word) 
                for word in tokens 
                if word not in self.stop_words and len(word) > 2
            ]
            
            return ' '.join(tokens) if tokens else ""
            
        except Exception as e:
            print(f"\nText preprocessing error: {str(e)}")
            return ""
    
    def augment_with_deepseek(self, texts: List[str], labels: List[str], num_samples: int = 3) -> pd.DataFrame:
        """Secure data augmentation with rate limiting and validation"""
        augmented = []
        valid_pairs = [
            (t, l) for t, l in zip(texts, labels) 
            if l == "TRUE" and isinstance(t, str) and len(t.strip()) > 0
        ][:20]  # Safety limit
        
        print("Augmenting data via API...")
        for text, label in tqdm(valid_pairs, desc="Generating augmentations"):
            prompt = f"""Generate {num_samples} cyberbullying variations:
            Original: "{text[:200]}"  # Truncate to prevent abuse
            
            Rules:
            1. Maintain harmful intent
            2. Different phrasing
            3. Maximum 25 words
            4. Avoid personal information
            """
            
            payload = {
                "model": "deepseek-chat",
                "messages": [{
                    "role": "user", 
                    "content": prompt
                }],
                "temperature": 0.7,
                "max_tokens": 300,
                "stop": ["\n\n"]
            }
            
            try:
                response = self.api_handler.make_api_request(payload)
                content = response['choices'][0]['message']['content']
                
                # Secure parsing and validation
                variations = []
                for v in content.split('\n'):
                    v = v.strip('"').strip()
                    if 10 <= len(v) <= 500:  # Length validation
                        # Content validation
                        if any(word in v.lower() for word in self._toxic_keywords):
                            v = re.sub(r'[^\w\s]', '', v)  # Further sanitization
                            variations.append(v)
                
                augmented.extend([
                    {"text": v, "label": "TRUE"} 
                    for v in variations[:num_samples]  # Enforce sample limit
                ])
                
            except Exception as e:
                print(f"Secure augmentation failed: {str(e)}")
                continue
                
        return pd.DataFrame(augmented)

class CyberbullyingDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = [1 if label == "TRUE" else 0 for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate inputs
        if len(self.texts) != len(self.labels):
            raise ValueError("Texts and labels must have same length")
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def compute_metrics(pred: EvalPrediction) -> Dict:
    """Improved metrics computation with error handling"""
    try:
        preds = np.argmax(pred.predictions, axis=1)
        report = classification_report(
            pred.label_ids,
            preds,
            output_dict=True,
            zero_division=0
        )
        return {
            'f1': report['weighted avg']['f1-score'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'accuracy': report['accuracy']
        }
    except Exception as e:
        print(f"Metrics computation error: {str(e)}")
        return {'f1': 0.0}

def train_models(data_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    """Secure training pipeline with enhanced validation and progress tracking"""
    try:
        print("\n" + "="*50)
        print("Starting Training Pipeline")
        print("="*50 + "\n")
        
        # Validate output directory
        print("Validating output directory...")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"Cannot write to {output_dir}")
        
        processor = DataProcessor()
        
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        df = processor.load_data(data_path)
        
        # Data augmentation with safety checks
        bullying_samples = df[df['label'] == "TRUE"]
        augmentation_sample = min(20, len(bullying_samples))
        
        if augmentation_sample > 0:
            print(f"\nAugmenting with {augmentation_sample} samples...")
            augmented = processor.augment_with_deepseek(
                bullying_samples['text'].sample(augmentation_sample, random_state=random_state),
                ["TRUE"] * augmentation_sample
            )
            expanded_df = pd.concat([df, augmented], ignore_index=True)
        else:
            expanded_df = df
        
        # Split data with stratification
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            expanded_df['processed_text'],
            expanded_df['label'],
            test_size=test_size,
            random_state=random_state,
            stratify=expanded_df['label']
        )
        
        # 1. Train Logistic Regression
        print("\n" + "="*50)
        print("Training Logistic Regression Model")
        print("="*50)
        
        print("\nVectorizing text...")
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(X_train)
        
        print("Training classifier...")
        lr = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        with tqdm(total=1000, desc="Logistic Regression Training") as pbar:
            lr.fit(X_train_tfidf, y_train)
            pbar.update(1000)
        
        # Save models with atomic writes
        print("\nSaving models...")
        temp_lr_path = output_dir / 'lr_model.temp.pkl'
        joblib.dump(lr, temp_lr_path)
        temp_lr_path.replace(output_dir / 'lr_model.pkl')
        
        temp_tfidf_path = output_dir / 'tfidf_vectorizer.temp.pkl'
        joblib.dump(tfidf, temp_tfidf_path)
        temp_tfidf_path.replace(output_dir / 'tfidf_vectorizer.pkl')
        
        # 2. Train Transformer model
        print("\n" + "="*50)
        print("Training Transformer Model")
        print("="*50)
        
        print("\nInitializing BERT model...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2
        )
        
        print("Creating datasets...")
        train_dataset = CyberbullyingDataset(X_train.tolist(), y_train.tolist(), tokenizer)
        test_dataset = CyberbullyingDataset(X_test.tolist(), y_test.tolist(), tokenizer)
        
        print("Setting up training...")
        training_args = TrainingArguments(
            output_dir=str(output_dir / 'bert_model'),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=str(output_dir / 'logs'),
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=random_state,
            report_to="none",
            logging_steps=100
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        print("\nStarting BERT training...")
        trainer.train()
        
        # Save model safely
        print("\nSaving BERT model...")
        final_model_dir = output_dir / 'bert_model'
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print(f"Models saved to: {output_dir}")
        print("="*50)
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("Starting cyberbullying detection training...")
        train_models(
            data_path='data/raw_data.csv',
            output_dir='models',
            test_size=0.2,
            random_state=42
        )
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        exit(1)