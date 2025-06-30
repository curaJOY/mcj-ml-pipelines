from typing import List, Dict
import pandas as pd

"""
Handle real-world data challenges
"""

def detect_annotation_quality_issues(annotations: pd.DataFrame) -> List[str]:
    """
    TODO: Identify problematic annotations
    
    Red flags:
    - Annotator fatigue (declining quality over time)
    - Systematic disagreement on specific post types
    - Cultural/linguistic misunderstandings
    - Gaming the system (random clicking)
    """
    issues = []

    # Check for annotator fatigue
    annotators = annotations["annotator_id"].unique()
    for annotator in annotators:
        annots = annotations[annotations["annotator_id"] == annotator]
        if len(annots) >= 10:
            midpoint = len(annots) // 2
            early = annots.iloc[:midpoint]["label"].value_counts(normalize=True)
            late = annots.iloc[midpoint:]["label"].value_counts(normalize=True)
            if (early - late).abs().sum() > 0.7:
                issues.append(f"Annotator {annotator} shows potential fatigue or inconsistency over time.")

    # Check for low-quality annotations
    for annotator in annotators:
        labels = annotations[annotations["annotator_id"] == annotator]["label"]
        if labels.nunique() == 1:
            issues.append(f"Annotator {annotator} used the same label for all posts.")

    # Check for systematic bias
    grouped = annotations.groupby("post_id")["label"].nunique()
    conflicts = grouped[grouped > 1]
    for post_id in conflicts.index:
        issues.append(f"Post {post_id} has conflicting labels from different annotators.")
    
    return issues



def handle_edge_cases():
    """
    TODO: Create guidelines for difficult cases
    
    Edge cases to consider:
    - Sarcasm vs genuine threats
    - Cultural differences in communication
    - Neurodivergent communication styles
    - Code-switching between languages
    - Generational slang differences
    """
    
    guidelines = {
        "sarcasm_detection": "Annotators must evaluate tone, punctuation, emojis, and context. "+
                "Posts that appear humorous but involve mockery, threats, or demeaning language should be flagged as 'sarcasm-potential'. "+
                "Never assume positive intent without evidence — escalate unclear cases for senior review.",
        "cultural_sensitivity": "Annotators must refrain from labeling posts that reference "+
                "unfamiliar traditions, dialects, or humor styles without seeking context. If unsure, note that there is"+ 
                "a cultural ambiguity and flag for peer audit. ",
        "accessibility_considerations": "Code-switching, hybrid dialects, or non-English phrasing should be carefully studied."+
                "If the annotator cannot fully interpret the language or slang, the post should be noted as linguistically unclear "+
                "to avoid introducing linguistic bias.",
        "language_variations": "Code-switching, hybrid dialects, or non-English phrasing should be carefully studied."+
                "If the annotator cannot fully interpret the language or slang, the post should be noted as linguistically unclear "+
                "to avoid introducing linguistic bias."
    }


    
    return guidelines
