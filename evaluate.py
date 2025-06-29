import joblib
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.preprocessing import DataProcessor

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png')
    plt.close()

def plot_metrics(report, model_name):
    metrics = ['precision', 'recall', 'f1-score']
    classes = ['Non-Bullying', 'Bullying']
    
    data = {cls: [report[cls][metric] for metric in metrics] for cls in classes}
    df = pd.DataFrame(data, index=metrics)
    
    ax = df.plot(kind='bar', figsize=(10, 6), rot=0)
    plt.title(f'{model_name} Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(title='Class')
    plt.tight_layout()
    plt.savefig(f'results/{model_name.lower()}_metrics.png')
    plt.close()

def evaluate_models(data_path, model_dir):
    # Load data
    processor = DataProcessor()
    df = processor.load_data(data_path)
    X_test, y_test = df['text'], df['label']
    
    # Load models
    lr = joblib.load(os.path.join(model_dir, 'lr_model.pkl'))
    tfidf = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, 'bert_model/tokenizer'))
    bert_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_dir, 'bert_model'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model.to(device)
    
    # Prepare test data
    X_test_processed = X_test.apply(processor.preprocess_text)
    X_test_tfidf = tfidf.transform(X_test_processed)
    
    # Evaluate Logistic Regression
    y_pred_lr = lr.predict(X_test_tfidf)
    lr_report = classification_report(y_test, y_pred_lr, output_dict=True)
    
    # Evaluate BERT
    test_dataset = CyberbullyingDataset(X_test.tolist(), y_test.tolist(), tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    bert_model.eval()
    y_true_bert, y_pred_bert = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = bert_model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            y_true_bert.extend(batch['labels'].tolist())
            y_pred_bert.extend(preds.tolist())
    
    y_pred_bert = ['TRUE' if p == 1 else 'FALSE' for p in y_pred_bert]
    bert_report = classification_report(y_test, y_pred_bert, output_dict=True)
    
    # Ensemble evaluation
    y_pred_ensemble = []
    for i in range(len(X_test)):
        lr_prob = lr.predict_proba(X_test_tfidf[i:i+1])[0][1]
        bert_input = tokenizer(
            X_test.iloc[i],
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            bert_output = bert_model(**bert_input)
            bert_prob = torch.softmax(bert_output.logits, dim=1)[0][1].item()
        
        ensemble_prob = (lr_prob + bert_prob) / 2
        y_pred_ensemble.append("TRUE" if ensemble_prob > 0.5 else "FALSE")
    
    ensemble_report = classification_report(y_test, y_pred_ensemble, output_dict=True)
    
    # Create visualizations
    os.makedirs('results', exist_ok=True)
    
    # Confusion matrices
    plot_confusion_matrix(y_test, y_pred_lr, ['Non-Bullying', 'Bullying'], 'Logistic Regression CM')
    plot_confusion_matrix(y_test, y_pred_bert, ['Non-Bullying', 'Bullying'], 'BERT CM')
    plot_confusion_matrix(y_test, y_pred_ensemble, ['Non-Bullying', 'Bullying'], 'Ensemble CM')
    
    # Metrics comparison
    plot_metrics(lr_report, 'Logistic Regression')
    plot_metrics(bert_report, 'BERT')
    plot_metrics(ensemble_report, 'Ensemble')
    
    # Generate comprehensive report
    report = f"""
    Cyberbullying Detection Evaluation Report
    =======================================
    
    Logistic Regression:
    - Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}
    - Precision (Bullying): {lr_report['TRUE']['precision']:.4f}
    - Recall (Bullying): {lr_report['TRUE']['recall']:.4f}
    - F1 (Bullying): {lr_report['TRUE']['f1-score']:.4f}
    
    BERT Model:
    - Accuracy: {accuracy_score(y_test, y_pred_bert):.4f}
    - Precision (Bullying): {bert_report['TRUE']['precision']:.4f}
    - Recall (Bullying): {bert_report['TRUE']['recall']:.4f}
    - F1 (Bullying): {bert_report['TRUE']['f1-score']:.4f}
    
    Ensemble Model:
    - Accuracy: {accuracy_score(y_test, y_pred_ensemble):.4f}
    - Precision (Bullying): {ensemble_report['TRUE']['precision']:.4f}
    - Recall (Bullying): {ensemble_report['TRUE']['recall']:.4f}
    - F1 (Bullying): {ensemble_report['TRUE']['f1-score']:.4f}
    """
    
    with open('results/evaluation_report.txt', 'w') as f:
        f.write(report)
    
    print("Evaluation completed. Results saved in 'results/' directory.")

if __name__ == "__main__":
    evaluate_models('data/raw_data.csv', 'models')