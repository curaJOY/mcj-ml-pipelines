# -*- coding: utf-8 -*-
import re
import string
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

class CyberbullyingPreprocessor:
    """
    Comprehensive preprocessing pipeline for cyberbullying detection
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.offensive_words = [
            'bitch', 'btch', 'b*tch', 'shit', 'fuck', 'damn', 'ass', 'slut', 
            'whore', 'stupid', 'idiot', 'gay', 'retard', 'retarded', 'loser',
            'faggot', 'fag', 'homo', 'dyke', 'cunt', 'pussy'
        ]
        self.stats = {}
    
    def load_dataset(self, file_path=None):
        """Load and parse the cyberbullying dataset"""
        if file_path is None:
            # Try different possible locations for dataset.txt
            import os
            possible_paths = [
                'dataset.txt',  # Current directory
                '../dataset.txt',  # Parent directory (for models in subfolders)
                '../../dataset.txt',  # Two levels up
                os.path.join(os.path.dirname(__file__), '..', 'dataset.txt')  # Relative to this file
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    file_path = path
                    break
            else:
                print(f"Error: dataset.txt not found in any of the expected locations: {possible_paths}")
                return [], []
        
        texts = []
        labels = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove outer braces
            content = content.strip()
            if content.startswith('{') and content.endswith('}'):
                content = content[1:-1]
            
            # Parse line by line
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Handle complex cases with nested quotes
                if ': True,' in line or ': False,' in line:
                    if ': True,' in line:
                        parts = line.split(': True,')
                        boolean_val = True
                    else:
                        parts = line.split(': False,')
                        boolean_val = False
                    
                    if len(parts) >= 2:
                        text_part = parts[0].strip()
                        # Remove surrounding quotes
                        if text_part.startswith('"') and text_part.endswith('"'):
                            text_part = text_part[1:-1]
                        elif text_part.startswith("'") and text_part.endswith("'"):
                            text_part = text_part[1:-1]
                        
                        if text_part and len(text_part) > 1:
                            texts.append(text_part)
                            labels.append(boolean_val)
            
            print(f"Successfully loaded {len(texts)} samples from {file_path}")
            return texts, labels
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return [], []
    
    def basic_text_cleaning(self, text):
        """Basic text cleaning while preserving important features"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace but preserve structure
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove URLs but keep the fact that they existed
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, ' URL ', text)
        
        # Remove usernames but keep the mention pattern
        text = re.sub(r'@\w+', ' MENTION ', text)
        
        # Remove hashtags but keep the tag pattern
        text = re.sub(r'#\w+', ' HASHTAG ', text)
        
        return text
    
    def extract_features(self, texts):
        """Extract comprehensive features from texts"""
        features = []
        
        for text in texts:
            text_features = {}
            
            # Basic length features
            text_features['char_count'] = len(text)
            text_features['word_count'] = len(text.split())
            text_features['avg_word_length'] = sum(len(word) for word in text.split()) / max(1, len(text.split()))
            
            # Punctuation features
            text_features['exclamation_count'] = text.count('!')
            text_features['question_count'] = text.count('?')
            text_features['dot_count'] = text.count('.')
            text_features['comma_count'] = text.count(',')
            
            # Capitalization features
            text_features['upper_case_count'] = sum(1 for c in text if c.isupper())
            text_features['upper_case_ratio'] = text_features['upper_case_count'] / max(1, len(text))
            text_features['all_caps_words'] = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
            
            # Offensive language features
            text_lower = text.lower()
            text_features['offensive_word_count'] = sum(1 for word in self.offensive_words if word in text_lower)
            text_features['has_offensive_language'] = 1 if text_features['offensive_word_count'] > 0 else 0
            
            # Special character features
            text_features['digit_count'] = sum(1 for c in text if c.isdigit())
            text_features['special_char_count'] = sum(1 for c in text if c in string.punctuation)
            
            # Emotional indicators
            text_features['has_laughter'] = 1 if any(laugh in text_lower for laugh in ['lol', 'haha', 'lmao', 'rofl']) else 0
            text_features['has_crying'] = 1 if any(cry in text_lower for cry in ['cry', 'sob', 'boo']) else 0
            
            # Pattern features
            text_features['repeated_chars'] = len(re.findall(r'(.)\1{2,}', text))  # 3+ repeated chars
            text_features['has_ellipsis'] = 1 if '...' in text else 0
            
            features.append(text_features)
        
        return pd.DataFrame(features)
    
    def create_bag_of_words(self, texts, max_features=5000):
        """Create bag of words representation with smart feature selection"""
        # Tokenize and count words
        word_counts = Counter()
        for text in texts:
            # Simple tokenization
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts.update(words)
        
        # Get top features
        top_words = [word for word, count in word_counts.most_common(max_features)]
        
        # Create BOW matrix
        bow_features = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            word_vector = [words.count(word) for word in top_words]
            bow_features.append(word_vector)
        
        bow_df = pd.DataFrame(bow_features, columns=[f'bow_{word}' for word in top_words])
        return bow_df, top_words
    
    def preprocess_full_pipeline(self, texts, labels, test_size=0.2, random_state=42):
        """Complete preprocessing pipeline"""
        # Store original stats
        self.stats['total_samples'] = len(texts)
        self.stats['cyberbullying_samples'] = sum(labels)
        self.stats['class_ratio'] = sum(labels) / len(labels)
        
        # Clean texts
        print("Cleaning texts...")
        cleaned_texts = [self.basic_text_cleaning(text) for text in texts]
        
        # Extract linguistic features
        linguistic_features = self.extract_features(cleaned_texts)
        
        # Create bag of words
        bow_features, vocabulary = self.create_bag_of_words(cleaned_texts, max_features=1000)
        
        # Combine all features
        print("Combining feature sets...")
        all_features = pd.concat([linguistic_features, bow_features], axis=1)
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Train-test split
        print("Creating train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            all_features, encoded_labels, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=encoded_labels
        )
        
        # Store preprocessing info
        self.stats['features_count'] = all_features.shape[1]
        self.stats['train_samples'] = len(X_train)
        self.stats['test_samples'] = len(X_test)
        self.stats['vocabulary_size'] = len(vocabulary)
        
        print("Preprocessing complete!")
        print(f"   Total features: {self.stats['features_count']}")
        print(f"   Train samples: {self.stats['train_samples']}")
        print(f"   Test samples: {self.stats['test_samples']}")
        print(f"   Vocabulary size: {self.stats['vocabulary_size']}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': all_features.columns.tolist(),
            'vocabulary': vocabulary,
            'cleaned_texts': cleaned_texts,
            'original_texts': texts
        }
    
    def save_preprocessor(self, filepath='cyberbullying_preprocessor.pkl'):
        """Save the preprocessor for later use"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_preprocessor(filepath='cyberbullying_preprocessor.pkl'):
        """Load a saved preprocessor"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def print_stats(self):
        """Print preprocessing statistics"""
        print("\nPreprocessing Statistics:")
        print("-" * 40)
        for key, value in self.stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        print("-" * 40)

def main():
    """Main preprocessing execution"""
    print("CYBERBULLYING DETECTION - DATA PREPROCESSING")
    print("="*60)
    print("Setting up preprocessing pipeline...")
    
    # Initialize preprocessor
    preprocessor = CyberbullyingPreprocessor()
    
    # Load dataset
    texts, labels = preprocessor.load_dataset()
    
    if not texts:
        print("Failed to load dataset. Cannot proceed.")
        return
    
    # Run full preprocessing pipeline
    processed_data = preprocessor.preprocess_full_pipeline(texts, labels)
    
    # Print statistics
    preprocessor.print_stats()
    
    # Save processed data
    print("\nSaving processed data...")
    pd.DataFrame(processed_data['X_train']).to_csv('X_train.csv', index=False)
    pd.DataFrame(processed_data['X_test']).to_csv('X_test.csv', index=False)
    pd.DataFrame(processed_data['y_train']).to_csv('y_train.csv', index=False)
    pd.DataFrame(processed_data['y_test']).to_csv('y_test.csv', index=False)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print("Key outputs:")
    print("   - Feature-engineered training and test sets")
    print("   - Linguistic features (punctuation, capitalization, etc.)")
    print("   - Bag-of-words features (top 1000 words)")
    print("   - Saved preprocessor for future use")
    print("\nReady for Phase 2: Model Development!")
    
    return processed_data

if __name__ == "__main__":
    processed_data = main() 