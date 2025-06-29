import joblib
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.preprocessing import DataProcessor
import os
import matplotlib.pyplot as plt
import hashlib
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='predictions.log'
)

class SecureModelLoader:
    """Safely loads and validates models"""
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._validate_path()

    def _validate_path(self):
        """Verify model directory structure"""
        required_files = {
            'lr_model.pkl',
            'tfidf_vectorizer.pkl',
            'bert_model/config.json',
            'bert_model/pytorch_model.bin',
            'bert_model/tokenizer/vocab.txt'
        }
        
        missing = []
        for f in required_files:
            if not os.path.exists(os.path.join(self.model_dir, f)):
                missing.append(f)
        
        if missing:
            raise FileNotFoundError(
                f"Missing model files: {', '.join(missing)}"
            )

    def load_models(self):
        """Secure model loading with validation"""
        try:
            # Load with explicit allowlist
            lr_model = joblib.load(
                os.path.join(self.model_dir, 'lr_model.pkl')
            )
            
            tfidf = joblib.load(
                os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(self.model_dir, 'bert_model/tokenizer')
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(
                os.path.join(self.model_dir, 'bert_model')
            )
            
            return lr_model, tfidf, tokenizer, model
            
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            raise

class CyberbullyingDetector:
    def __init__(self, model_dir: str):
        self.processor = DataProcessor()
        self.model_loader = SecureModelLoader(model_dir)
        
        try:
            (self.lr_model, 
             self.tfidf, 
             self.tokenizer, 
             self.bert_model) = self.model_loader.load_models()
            
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.bert_model.to(self.device)
            
        except Exception as e:
            logging.critical(f"Initialization failed: {str(e)}")
            raise

    def predict(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Secure prediction pipeline"""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")
            
        try:
            # Preprocess with injection protection
            processed_text = self.processor.preprocess_text(text)
            
            # Get predictions
            lr_prob = self._logistic_regression_predict(processed_text)
            bert_prob = self._bert_predict(text)
            
            # Secure ensemble
            ensemble_prob = (lr_prob + bert_prob) / 2
            prediction = (
                "Cyberbullying" if ensemble_prob > threshold 
                else "Non-Cyberbullying"
            )
            
            # Generate visualization
            vis_path = self._generate_visualization(
                text, lr_prob, bert_prob, ensemble_prob, prediction
            )
            
            return {
                "text": text[:500],  # Truncate for safety
                "prediction": prediction,
                "confidence": round(ensemble_prob, 4),
                "probabilities": {
                    "logistic_regression": round(lr_prob, 4),
                    "bert": round(bert_prob, 4),
                    "ensemble": round(ensemble_prob, 4)
                },
                "threshold": threshold,
                "visualization": vis_path
            }
            
        except Exception as e:
            logging.error(f"Prediction failed for text: {text[:100]}... - {str(e)}")
            raise

    def _logistic_regression_predict(self, text: str) -> float:
        """Secure LR prediction"""
        try:
            features = self.tfidf.transform([text])
            return self.lr_model.predict_proba(features)[0][1]
        except Exception as e:
            logging.error(f"LR prediction failed: {str(e)}")
            raise

    def _bert_predict(self, text: str) -> float:
        """Secure BERT prediction"""
        try:
            inputs = self.tokenizer(
                text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                return torch.softmax(outputs.logits, dim=1)[0][1].item()
                
        except Exception as e:
            logging.error(f"BERT prediction failed: {str(e)}")
            raise

    def _generate_visualization(self, text: str, lr_prob: float, 
                              bert_prob: float, ensemble_prob: float,
                              prediction: str) -> str:
        """Secure visualization generation"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Secure display text
            display_text = (text[:50] + '...') if len(text) > 50 else text
            display_text = display_text.replace('\n', ' ').strip()
            
            # Create plot
            models = ['Logistic Regression', 'BERT', 'Ensemble']
            probs = [lr_prob, bert_prob, ensemble_prob]
            colors = ['#1f77b4', '#2ca02c', '#d62728']  # Colorblind-friendly
            
            bars = plt.bar(models, probs, color=colors)
            plt.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
            plt.ylim(0, 1)
            plt.ylabel('Probability Score')
            plt.title(
                f'Cyberbullying Detection Analysis\n'
                f'Text: "{display_text}" → Prediction: {prediction}',
                pad=20
            )
            
            # Annotate bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2., 
                    height/2,
                    f'{height:.2f}',
                    ha='center', 
                    va='center',
                    color='white',
                    fontweight='bold'
                )
            
            plt.tight_layout()
            
            # Secure file naming
            text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
            os.makedirs('results/predictions', exist_ok=True)
            vis_path = f'results/predictions/pred_{text_hash}.png'
            
            plt.savefig(vis_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            return vis_path
            
        except Exception as e:
            logging.error(f"Visualization failed: {str(e)}")
            raise

def main():
    """Secure prediction pipeline"""
    try:
        detector = CyberbullyingDetector('models')
        
        test_cases = [
            "You're such a worthless piece of trash",
            "I really enjoyed your presentation today!",
            "Nobody likes you, why don't you just leave?",
            "That outfit looks... interesting on you",
            "Let's work together on this project"
        ]
        
        results = []
        for text in test_cases:
            try:
                result = detector.predict(text)
                results.append(result)
                
                print(f"\nAnalysis for: {text[:60]}...")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Visualization saved to: {result['visualization']}")
                
            except Exception as e:
                print(f"Error processing: {text[:30]}... - {str(e)}")
                continue
                
        return results
        
    except Exception as e:
        logging.critical(f"System failure: {str(e)}")
        raise

if __name__ == "__main__":
    main()