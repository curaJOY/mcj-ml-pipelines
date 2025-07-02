# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_analysis'))
from data_preprocessing import CyberbullyingPreprocessor

class CyberbullyingTraditionalML:
    """Traditional ML models for cyberbullying detection"""
    
    def __init__(self):
        self.preprocessor = CyberbullyingPreprocessor()
        self.models = {}
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load dataset and run preprocessing pipeline"""
        print("Loading and preprocessing dataset...")
        
        # Load raw data
        texts, labels = self.preprocessor.load_dataset()
        if not texts:
            raise ValueError("Failed to load dataset")
        
        # Run preprocessing pipeline
        processed_data = self.preprocessor.preprocess_full_pipeline(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Extract data from dictionary
        self.X_train = processed_data['X_train']
        self.X_test = processed_data['X_test']
        self.y_train = processed_data['y_train']
        self.y_test = processed_data['y_test']
        
        print("Preprocessing complete!")
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Class distribution - Cyberbullying: {sum(self.y_train)}/{len(self.y_train)} ({sum(self.y_train)/len(self.y_train)*100:.1f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_traditional_models(self):
        """Train traditional machine learning models"""
        print("\nTraining Traditional ML Models...")
        
        # Define models with class balancing
        models_config = {
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            ),
            'SVM (RBF)': SVC(
                kernel='rbf', random_state=42, probability=True, class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'Naive Bayes': MultinomialNB()
        }
        
        # Train and evaluate each model
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Store model and results
            self.models[name] = model
            self.results[name] = metrics
            
            print(f"{name} - F1: {metrics['f1']:.3f}, Accuracy: {metrics['accuracy']:.3f}")
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary')
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None
            
        return metrics
    
    def evaluate_models(self):
        """Comprehensive model evaluation and comparison"""
        print("\nModel Evaluation and Comparison")
        print("=" * 50)
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(3)
        
        # Sort by F1 score (important for imbalanced dataset)
        results_df = results_df.sort_values('f1', ascending=False)
        
        print("\nModel Performance Summary:")
        print(results_df.to_string())
        
        # Best model
        best_model_name = results_df.index[0]
        best_f1 = results_df.loc[best_model_name, 'f1']
        
        print(f"\nBest Model: {best_model_name} (F1: {best_f1:.3f})")
        
        return results_df
    
    def plot_model_comparison(self):
        """Create visualization comparing model performance"""
        print("\nCreating performance visualization...")
        
        # Prepare data for plotting
        results_df = pd.DataFrame(self.results).T
        
        # Metrics to plot
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cyberbullying Detection - Traditional ML Model Comparison', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            # Create bar plot
            models = results_df.index
            values = results_df[metric]
            
            bars = ax.bar(models, values, color='skyblue', alpha=0.8)
            ax.set_title(f'{metric.upper()} Comparison', fontweight='bold')
            ax.set_ylabel(metric.upper())
            ax.set_ylim(0, 1.0)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if not pd.isna(value):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Highlight best performing model
            best_idx = values.idxmax()
            best_bar_idx = list(models).index(best_idx)
            bars[best_bar_idx].set_color('lightcoral')
        
        plt.tight_layout()
        plt.savefig('traditional_ml_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualization saved as 'traditional_ml_comparison.png'")
    
    def perform_detailed_analysis(self, best_model_name):
        """Perform detailed analysis of the best model"""
        print(f"\nDetailed Analysis of {best_model_name}")
        print("=" * 50)
        
        # Get best model
        best_model = self.models[best_model_name]
        
        # Make predictions
        y_pred = best_model.predict(self.X_test)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Non-Cyberbullying', 'Cyberbullying']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
        # Feature importance analysis
        if hasattr(best_model, 'feature_importances_'):
            print(f"\nTop 15 Most Important Features:")
            feature_names = self.X_train.columns
            importances = best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(feature_importance.head(15).to_string(index=False))
        
        elif hasattr(best_model, 'coef_'):
            print(f"\nTop 15 Most Important Features (by coefficient magnitude):")
            feature_names = self.X_train.columns
            coefs = np.abs(best_model.coef_[0])
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'abs_coefficient': coefs
            }).sort_values('abs_coefficient', ascending=False)
            
            print(feature_importance.head(15).to_string(index=False))
    
    def perform_cross_validation(self, best_model_name):
        """Perform cross-validation analysis"""
        print(f"\nCross-Validation Analysis for {best_model_name}")
        print("=" * 50)
        
        best_model = self.models[best_model_name]
        
        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(best_model, self.X_train, self.y_train, cv=5, scoring='f1')
        
        print("5-Fold Cross-Validation F1 Scores:")
        for i, score in enumerate(cv_scores, 1):
            print(f"   Fold {i}: {score:.3f}")
        
        print(f"\nCross-Validation Summary:")
        print(f"   Mean F1: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
        print(f"   Min F1: {cv_scores.min():.3f}")
        print(f"   Max F1: {cv_scores.max():.3f}")

def main():
    """Main execution function for traditional ML models"""
    print("Starting Cyberbullying Detection - Traditional ML")
    print("=" * 60)
    
    # Initialize developer
    developer = CyberbullyingTraditionalML()
    
    try:
        # Load and preprocess data
        developer.load_and_preprocess_data()
        
        # Train traditional ML models
        developer.train_traditional_models()
        
        # Evaluate and compare models
        results_df = developer.evaluate_models()
        
        # Create visualizations
        developer.plot_model_comparison()
        
        # Perform detailed analysis
        best_model_name = results_df.index[0]
        developer.perform_detailed_analysis(best_model_name)
        
        # Cross-validation analysis
        developer.perform_cross_validation(best_model_name)
        
        print("\nTraditional ML analysis completed successfully!")
        print("Generated: traditional_ml_comparison.png")
        print("Results show traditional ML effectiveness for cyberbullying detection")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 