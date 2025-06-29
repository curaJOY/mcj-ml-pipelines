import pandas as pd
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# 1. Load the dataset
df = pd.read_csv("annotations.csv")

# 2. Map 'Yes'/'No' to 1/0 for target
df['bullying_label'] = df['bullying'].map({'Yes': 1, 'No': 0})

# 3. Balance the dataset by upsampling the minority class
df_majority = df[df['bullying_label'] == 0]
df_minority = df[df['bullying_label'] == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Balanced class distribution:")
print(df_balanced['bullying_label'].value_counts())

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Then apply before vectorizing:
df['clean_text'] = df['text'].apply(preprocess_text)


# 4. Clean text and vectorize
df_balanced['clean_text'] = df_balanced['text'].astype(str).str.lower()

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df_balanced['clean_text'])
y = df_balanced['bullying_label']

# 5. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Initialize and train RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# 7. Predict probabilities and apply threshold
y_probs = model.predict_proba(X_test)[:, 1]
threshold = 0.3
y_pred = (y_probs >= threshold).astype(int)

# 8. Evaluate
print("\nModel Performance on Test Data:")
print("Accuracy: ", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall:   ", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1 Score: ", round(f1_score(y_test, y_pred, zero_division=0), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 9. Save model and vectorizer
joblib.dump(model, "bullying_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
