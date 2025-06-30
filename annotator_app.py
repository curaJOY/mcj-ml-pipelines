"""
Build a web-based annotation tool for students to label cyberbullying data
"""

import streamlit as st
import pandas as pd
import json
from typing import Dict, List
from sklearn.metrics import cohen_kappa_score


# Sample social media posts (realistic but sanitized)
# SAMPLE_POSTS = [
#     {
#         "id": 1,
#         "text": "Hope you have fun at the dance! 😊 (JK nobody will dance with you)",
#         "platform": "instagram",
#         "context": "Comment on user's post about school dance",
#         "user_age": "teen"
#     },
#     {
#         "id": 2,
#         "text": "Bro you're literally killing me with these memes 😂💀",
#         "platform": "discord",
#         "context": "Response to funny meme in group chat",
#         "user_age": "teen"
#     }
# ]

with open("cleaned_posts.json", "r") as f:
    SAMPLE_POSTS = [json.loads(line) for line in f]

SAMPLE_POSTS = SAMPLE_POSTS[:25]  # limit to 25 for easier testing



def create_annotation_interface():
    """
    Streamlit interface for student annotators

    Features:
    1. Display post with full context
    2. Multi-label classification (bullying, self-harm, severity)
    3. Confidence scoring
    4. Notes field for edge cases
    5. Progress tracking
    """

    annotator_name = st.text_input("Enter your name or ID:", key="annotator_name")

    if not annotator_name:
        st.warning("Please enter your name or ID to continue.")
        st.stop()

    # Optional: Agreement checker
    if st.button("🔍 Check Annotator Agreement"):
        try:
            annotations = []
            with open("annotations.json", "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if 'annotator' in obj:
                            annotations.append(obj)
                    except json.JSONDecodeError:
                        continue


            score = calculate_inter_annotator_agreement(annotations)
            st.info(f"🧪 Inter-Annotator Agreement Score (Cohen's Kappa): {score:.2f}")
        except Exception as e:
            st.error(f"Couldn't calculate agreement: {e}")



    
    
    if "post_index" not in st.session_state:
        st.session_state.post_index = 0
    
    index = st.session_state.post_index
    total = len(SAMPLE_POSTS)
    
    if index >= total:
        st.success("✅ All posts annotated. You can close the app.")
        st.stop()
    
    post = SAMPLE_POSTS[index]
    
    st.title("Cyberbullying Annotation Tool")
    
    st.markdown(f"### Post {index + 1} of {total}")
    st.write(f"**Text:** {post['text']}")
    # st.write(f"**Platform:** {post['platform']}")
    # st.write(f"**Context:** {post['context']}")
    # st.write(f"**User Age:** {post['user_age']}")
    
    bullying = st.radio("Is the post bullying?", ("Yes", "No"))
    self_harm = st.radio("Is the post self-harm?", ("Yes", "No"))
    severity = st.selectbox("Select severity level:", ["Low", "Medium", "High"])
    confidence = st.slider("Confidence Score (%)", 0, 50, 100)
    notes = st.text_area("Notes (optional):")
    
    if st.button("Submit"):
        annotation = {
            "annotator": annotator_name,
            "id": post["id"],
            "text": post["text"],
            "bullying": bullying,
            "self_harm": self_harm,
            "severity": severity,
            "confidence": confidence,
            "notes": notes
        }

        with open("annotations.json", "a") as f:
            f.write(json.dumps(annotation) + "\n")

        st.success("✅ Annotation saved!")
        st.session_state.post_index += 1
        st.rerun()


def calculate_inter_annotator_agreement(annotations: List[Dict]) -> float:
    """
    Calculate Cohen's Kappa for two annotators on the 'bullying' label.
    Handles duplicate entries gracefully.
    """
    df = pd.DataFrame(annotations)

    # Keep only the latest annotation per (id, annotator) pair
    df = df.drop_duplicates(subset=["id", "annotator"], keep="last")

    # Pivot the data for Cohen's Kappa
    pivot = df.pivot(index='id', columns='annotator', values='bullying')

    # Drop rows where either annotator did not annotate
    pivot = pivot.dropna()

    if pivot.shape[1] < 2:
        raise ValueError("Not enough unique annotators to compute agreement.")

    # Convert 'Yes'/'No' to binary
    mapping = {"Yes": 1, "No": 0}
    annotator1 = pivot.iloc[:, 0].map(mapping).tolist()
    annotator2 = pivot.iloc[:, 1].map(mapping).tolist()

    return cohen_kappa_score(annotator1, annotator2)

    


if __name__ == "__main__":
    create_annotation_interface()
