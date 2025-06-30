from sklearn.metrics import accuracy_score, precision_score, recall_score
from typing import List, Dict
import pandas as pd


"""
Design experiments to validate the AI safety system
"""

def design_validation_study():
    """
    TODO: Design A/B test framework
    
    Research Questions:
    1. Does the AI coach reduce harmful behavior?
    2. What's the optimal intervention timing?
    3. How do we measure "effectiveness" ethically?
    
    Consider:
    - IRB approval requirements
    - Teen privacy protection
    - Bias in intervention targeting
    - Long-term behavioral change measurement
    """
    
    study_design = {
        "hypothesis": " A teen who receive warning/intervention messages immediately will show reduced engagement " +
                        "in harmful online behavior compared to those  who do not",
        "methodology": "This will be a randomized A/B/C test with teen users. \nParticipants will be randomly assigned to one of three groups:" + 
                        "Group A receives intervention messages immediately after a flagged post,"+
                        " Group B receives no intervention messages,"+ 
                        " and Group C receives intervention messages for a flagged post after 24 hours." + 
                        "The intervention messages will be compassionate, age-appropriate, and focused on"+
                        " empathy. Over 72 hours, changes will be tracked regarding the tone, frequency, and severity of posts"+
                        " across all groups to assess the impact of timing on behavior change.",
        "ethics_considerations": [
            "Obtain informed consent from participants and their guardians.",
            "Ensure all data is anonymous.",
            "Avoid contemptous or shaming language in intervention messages.",
            "Allow participants to opt-out at any time without penalty.",
            "Ensure interventions do not unintentionally escalate harm or distress."
        ],
        "success_metrics": [
            "Reduction in the number of harmful or bullying posts after messages.",
            "Decrease in severity of flagged content.",
            "Fewer repeat offenses within the same user session or account."
        ],
        "sample_size_calculation": "For the starting period, Each group will have 30 members. totalling 90 participants",
        "bias_mitigation": [
            "Make sure there is diversity in participant groups based on gender, race, and age.",
            "Randomize the group assignment.",
            "Regularly review flagged posts for bias."
        ]
    }
    return study_design


def analyze_algorithm_bias(predictions: List[Dict], demographics: List[Dict]) -> Dict:
    """
    TODO: Detect if algorithm has demographic bias
    
    Check for:
    - Gender bias in bullying detection
    - Racial bias in language interpretation  
    - Age bias in severity assessment
    - Platform bias (Discord vs Instagram)
    """
    
    df_pred = pd.DataFrame(predictions)
    df_demo = pd.DataFrame(demographics)
    df = pd.merge(df_pred, df_demo, on="id")

    results = {}
    for attr in ["gender", "race", "age", "platform"]:
        results[attr] = {}
        for group in df[attr].unique():
            subgroup = df[df[attr] == group]
            if len(subgroup) > 0:
                results[attr][group] = {
                    "accuracy": accuracy_score(subgroup["actual"], subgroup["predicted"]),
                    "precision": precision_score(subgroup["actual"], subgroup["predicted"], zero_division=0),
                    "recall": recall_score(subgroup["actual"], subgroup["predicted"], zero_division=0)
                }

    return results
