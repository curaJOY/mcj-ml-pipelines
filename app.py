import streamlit as st
import joblib
import pandas as pd

# Load saved model and vectorizer
model = joblib.load("bullying_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define rule-based offensive keywords
offensive_keywords = [
    "stupid", "dumb", "idiot", "pathetic", "worthless", "loser", "i'm going to kill you", "bitch", "fat", "dumb",
    "shut up", "irrelevant", "nobody cares", "you're a fool", "kill yourself", "you're ugly", "fuck you", "pig", "dick", "weirdo",
    "hate you", "nigga", "hoe", "whore", "jumped him/her", "fag", "faggot", "pussy", "you should die", "gonna buy you", "freak show",
    "pmo", "you are annoying", "you piss me off", "fatass", "You’re an embarrassment to everyone.", "Your existence is pointless.",
    "You’ll never be good at anything.", "You ruin everything you touch.", "you are a cow", "dumbfuck"
]

# Rule-based function
def rule_based_check(text):
    lowered = text.lower()
    return any(word in lowered for word in offensive_keywords)

# Streamlit UI setup
st.set_page_config(page_title="Cyberbullying Detector")
st.title("🔍 Cyberbullying Detection Tool")

# Text input
user_input = st.text_area("Enter text to analyze:")

# Predict only if button is pressed and input is not empty
if st.button("Analyze", key="analyze_btn") and user_input.strip():
    if rule_based_check(user_input):
        prediction = 1
        confidence = 0.95
        st.markdown("### 🔴 Prediction: **Bullying** (rule-based)")
        st.write(f"Confidence: {round(confidence * 100)}%")
    else:
        X_vec = vectorizer.transform([user_input])
        prediction = model.predict(X_vec)[0]
        confidence = model.predict_proba(X_vec)[0][prediction]
        if prediction == 1:
            st.markdown("### 🔴 Prediction: **Bullying**")
        else:
            st.markdown("### 🟢 Prediction: **Not Bullying**")
        st.write(f"Confidence: {round(confidence * 100)}%")
elif st.button("Analyze", key="warning_btn"):
    st.warning("⚠️ Please enter some text to analyze.")
