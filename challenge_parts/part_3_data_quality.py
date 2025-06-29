import pandas as pd
from typing import List, Dict

def detect_annotation_quality_issues(annotations: pd.DataFrame) -> List[str]:
    """
    Identifies problematic patterns in annotation data.
    Assumes `annotations` DataFrame has 'annotator_id', 'post_id', 'timestamp', 'confidence', and labels.
    """
    issues = []
    
    # --- Check for Annotator Fatigue ---
    annotations['timestamp'] = pd.to_datetime(annotations['timestamp'])
    annotations = annotations.sort_values(by=['annotator_id', 'timestamp'])
    annotations['time_diff'] = annotations.groupby('annotator_id')['timestamp'].diff().dt.total_seconds()
    
    # Flag annotators with consistently very fast annotation times
    avg_time = annotations.groupby('annotator_id')['time_diff'].mean()
    fast_annotators = avg_time[avg_time < 3.0].index.tolist() # Less than 3 seconds per post
    if fast_annotators:
        issues.append(f"Potential 'random clicking' or fatigue from annotators: {fast_annotators}. Average time per post is too low.")

    # --- Check for Systematic Disagreement ---
    # Find posts where annotators consistently disagree
    agreement_df = annotations.groupby('post_id')['is_bullying'].agg(['mean', 'count'])
    # Disagreement = mean is close to 0.5 and has multiple annotators
    disagreement_posts = agreement_df[(agreement_df['mean'] > 0.4) & (agreement_df['mean'] < 0.6) & (agreement_df['count'] > 2)]
    if not disagreement_posts.empty:
        issues.append(f"Systematic disagreement on post_ids: {disagreement_posts.index.tolist()}. These posts may be ambiguous and need clearer guidelines.")

    # --- Check for Low Confidence ---
    low_confidence_annotators = annotations.groupby('annotator_id')['confidence'].mean()
    low_conf = low_confidence_annotators[low_confidence_annotators < 0.5].index.tolist()
    if low_conf:
        issues.append(f"Annotators consistently reporting low confidence: {low_conf}. May indicate need for more training.")
        
    return issues

def handle_edge_cases() -> Dict[str, str]:
    """
    Creates clear guidelines for annotators on how to handle difficult cases.
    """
    guidelines = {
        "sarcasm_detection":
            "Guideline: Classify as bullying if the sarcasm is used to mock, exclude, or belittle a specific person or their identity. If it's general, self-deprecating, or clearly part of friendly banter (e.g., between known friends), classify as non-bullying. Look for negating phrases like 'JK', 'just kidding', or contradictory emojis. **When in doubt, use the 'Notes' field to explain your reasoning.**",
        
        "cultural_sensitivity":
            "Guideline: Be aware that some phrases or terms may be harmless in one culture but offensive in another (e.g., AAVE, regional slang). If you encounter unfamiliar language, use the 'Notes' field to flag it for review by a culturally competent expert. Do not assume malicious intent based on grammar or dialect alone.",
            
        "accessibility_considerations":
            "Guideline: Communication from neurodivergent individuals may be more direct or literal than typical communication. Do not automatically flag bluntness as aggression. Focus on whether the content's purpose is to cause emotional harm. If a communication style seems unique but not malicious, classify as non-bullying and note the pattern.",
            
        "language_variations":
            "Guideline: Handle code-switching (mixing languages) and generational slang carefully. For slang, consider the overall context. 'Deadass' or 'I'm dead' are often intensifiers, not threats. For code-switching, if the non-English portion seems critical to the meaning and you are unsure, flag it for a bilingual annotator.",
    }
    return guidelines

if __name__ == "__main__":
    import json
    
    print("--- Part 3: Data Quality & Edge Cases ---")
    
    # 1. Show the Edge Case Guidelines
    print("\n[A] Guidelines for Difficult Annotation Cases:")
    guidelines = handle_edge_cases()
    # Pretty-print the dictionary
    print(json.dumps(guidelines, indent=2))
    
    print("\n----------------------------------")
    print("\n[B] Annotation Quality Issues Function:")
    print("The `detect_annotation_quality_issues` function is a utility designed to be imported and used with a real annotations dataframe.")
    print("It cannot be run standalone without an annotation dataset.")