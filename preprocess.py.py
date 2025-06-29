import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 1. Load annotations
annotations = pd.read_csv(r"C:\Users\Paulp\Downloads\cyberbullying_project\annotations.csv")

# 2. Load original post texts
SAMPLE_POSTS = [
    {"id": 1, "text": "Hope you have fun at the dance! 😊 (JK nobody will dance with you)"},
    {"id": 2, "text": "Bro you're literally killing me with these memes 😂💀"},
    {"id": 3, "text": "Nobody cares about your opinion, just stop talking."},
    {"id": 4, "text": "Great job on your presentation! Really inspiring stuff."},
    {"id": 5, "text": "You’re such a loser, can’t believe you even tried."},
    {"id": 6, "text": "Had the best day with friends! Feeling blessed 😄"},
    {"id": 7, "text": "Wow, you really mess things up every single time."},
    {"id": 8, "text": "Keep up the hard work, you’re making progress every day!"},
    {"id": 9, "text": "You’re pathetic, just disappear already."},
    {"id": 10, "text": "Thanks for the support, really appreciate it everyone!"},
]

posts_df = pd.DataFrame(SAMPLE_POSTS)
df = pd.merge(annotations, posts_df, on="id")

# 3. Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)         # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)        # Remove punctuation & emojis
    text = re.sub(r"\s+", " ", text).strip()    # Remove extra spaces
    return text

df["clean_text"] = df["text"].apply(clean_text)

# 4. Convert labels to 0/1
df["label"] = df["bullying"].map({"Yes": 1, "No": 0})

# 5. TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Data ready for modeling!")
