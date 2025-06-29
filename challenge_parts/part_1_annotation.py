import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Tuple
from sklearn.metrics import cohen_kappa_score
import numpy as np

SAMPLE_POSTS = [
    {
        "id": 1, "text": "Hope you have fun at the dance! 😊 (JK nobody will dance with you)",
        "platform": "instagram", "context": "Comment on user's post about school dance", "user_age": "teen"
    },
    {
        "id": 2, "text": "Bro you're literally killing me with these memes 😂💀",
        "platform": "discord", "context": "Response to funny meme in group chat", "user_age": "teen"
    },
    {
        "id": 3, "text": "Why are you even on this team? You can't even play.",
        "platform": "tiktok", "context": "Comment on a video of a user playing basketball.", "user_age": "teen"
    },
    {
        "id": 4, "text": "u should just give up already, no one cares",
        "platform": "twitter", "context": "Reply to a user's tweet about their art project.", "user_age": "teen"
    },
    {
        "id": 5, "text": "Honestly, your argument isn't very strong. Have you considered this other perspective?",
        "platform": "reddit", "context": "Reply in a debate subreddit.", "user_age": "young_adult"
    }
]

def create_annotation_interface():
    """
    Builds a Streamlit interface for student annotators to label cyberbullying data.
    """
    st.title("CuraJOY Cyberbullying Annotation Tool")
    st.markdown("Your work helps us build a safer online world for everyone. Please read each post carefully before labeling.")

    # --- Session State for Progress Tracking ---
    if 'current_post_index' not in st.session_state:
        st.session_state.current_post_index = 0
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []

    # --- Progress Bar ---
    total_posts = len(SAMPLE_POSTS)
    progress = st.session_state.current_post_index / total_posts
    st.progress(progress)
    st.write(f"Post {st.session_state.current_post_index + 1} of {total_posts}")

    # --- Display Post ---
    if st.session_state.current_post_index < total_posts:
        post = SAMPLE_POSTS[st.session_state.current_post_index]

        st.subheader("Post to Annotate")
        with st.container(border=True):
            st.markdown(f"**Text:** `{post['text']}`")
            st.markdown(f"**Platform:** `{post['platform']}`")
            st.markdown(f"**Context:** `{post['context']}`")
            st.markdown(f"**Reported User Age Group:** `{post['user_age']}`")

        # --- Annotation Form ---
        with st.form(key=f"annotation_form_{post['id']}"):
            st.subheader("1. Classification Labels")
            is_bullying = st.checkbox("This is a form of bullying or harassment.")
            is_self_harm = st.checkbox("This post indicates risk of self-harm or suicide.")

            st.subheader("2. Severity (if bullying)")
            severity = st.select_slider(
                "Rate the severity of the bullying:",
                options=['Not Applicable', 'Mild Teasing', 'Moderate Harassment', 'Severe Attack'],
                value='Not Applicable'
            )

            st.subheader("3. Confidence Score")
            confidence = st.slider(
                "How confident are you in your labels?",
                min_value=0, max_value=100, value=75, step=5,
                help="100% means you are completely certain."
            )

            st.subheader("4. Annotator Notes")
            notes = st.text_area(
                "Add notes for complex cases (e.g., sarcasm, cultural context).",
                key=f"notes_{post['id']}"
            )

            # --- Submission Button ---
            submitted = st.form_submit_button("Submit & Next Post")

            if submitted:
                annotation = {
                    "post_id": post['id'],
                    "annotator_id": "annotator_01", # This would be dynamic in a real app
                    "is_bullying": is_bullying,
                    "is_self_harm": is_self_harm,
                    "severity": severity,
                    "confidence": confidence / 100.0,
                    "notes": notes
                }
                st.session_state.annotations.append(annotation)
                st.session_state.current_post_index += 1
                st.rerun() # Rerun the script to show the next post

    else:
        st.success("Annotation complete! Thank you for your contribution.")
        st.subheader("Your Annotations:")
        st.json(st.session_state.annotations)

        # --- Option to Download Annotations ---
        if st.session_state.annotations:
            df = pd.DataFrame(st.session_state.annotations)
            st.dataframe(df)
            st.download_button(
                label="Download annotations as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='annotations.csv',
                mime='text/csv',
            )

def calculate_inter_annotator_agreement(annotations: List[Dict]) -> Tuple[float, str]:
    """
    Implements Cohen's Kappa to measure inter-annotator agreement for the 'is_bullying' label.
    Assumes a list of annotations where each dict has 'post_id', 'annotator_id', and a label.
    """
    df = pd.DataFrame(annotations)
    if 'annotator_id' not in df.columns or df['annotator_id'].nunique() < 2:
        return 0.0, "Agreement requires at least two annotators' data."

    # Pivot table to align annotations by post_id --> compare first 2 annotators for simplicity (Cohen's Kappa)
    annotator_ids = df['annotator_id'].unique()
    anno1_id, anno2_id = annotator_ids[0], annotator_ids[1]

    anno1_labels = df[df['annotator_id'] == anno1_id].set_index('post_id')['is_bullying']
    anno2_labels = df[df['annotator_id'] == anno2_id].set_index('post_id')['is_bullying']

    # Align labels and drop posts not annotated by both
    comparison_df = pd.concat([anno1_labels, anno2_labels], axis=1, keys=[anno1_id, anno2_id]).dropna()

    if len(comparison_df) < 5: #min number of items to compare
        return 0.0, "Not enough commonly annotated items to calculate a meaningful score."

    kappa_score = cohen_kappa_score(comparison_df[anno1_id], comparison_df[anno2_id])
    
    
    interpretation = "Agreement Score: "
    if kappa_score < 0.2: interpretation += "Slight"
    elif kappa_score < 0.4: interpretation += "Fair"
    elif kappa_score < 0.6: interpretation += "Moderate"
    elif kappa_score < 0.8: interpretation += "Substantial"
    else: interpretation += "Almost Perfect"

    return kappa_score, interpretation

#run the app: `streamlit run part_1_annotation.py`
if __name__ == '__main__':
    create_annotation_interface()