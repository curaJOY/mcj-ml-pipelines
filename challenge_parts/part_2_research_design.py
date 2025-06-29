import pandas as pd
from typing import Dict, List

def design_validation_study() -> Dict:
    """
    Designs an A/B test framework to validate the AI coach safety system.
    """
    study_design = {
        "title": "A/B Test for CuraJOY AI Coach Intervention Efficacy",
        "hypothesis": "Adolescents exposed to the AI coach intervention (Treatment Group) after posting potentially harmful content will exhibit a statistically significant reduction in subsequent harmful posts and an increase in positive bystander interventions compared to a control group receiving no intervention.",
        "methodology": {
            "type": "Randomized Controlled Trial (RCT) / A/B Test",
            "control_group": "Users whose potentially harmful posts are detected but receive no AI intervention (status quo). Data is still collected for comparison.",
            "treatment_group": "Users whose potentially harmful posts are detected and who receive an immediate, private AI coach intervention message.",
            "duration": "12 weeks to observe short-term and medium-term behavioral changes.",
            "randomization": "User-level randomization upon their first detected offense to ensure no user is in both groups."
        },
        "ethics_considerations": [
            "IRB (Institutional Review Board) Approval: Submit study design for formal ethical review before commencement.",
            "Informed Consent: Obtain consent from parents/guardians and assent from teen participants, clearly explaining the study, potential for receiving AI messages, and data usage.",
            "Teen Privacy Protection: All data will be fully anonymized. The AI coach interacts privately and does not expose the user's behavior to others.",
            "Bias in Intervention Targeting: The underlying detection model must be rigorously audited for demographic bias (see `analyze_algorithm_bias`) before deployment to ensure fair targeting.",
            "Distress Protocol: A clear protocol for escalating high-risk cases (e.g., credible self-harm threats) to human moderators and emergency services, bypassing the A/B test logic."
        ],
        "success_metrics": {
            "primary": "Reduction in the rate of posts flagged as cyberbullying per user in the treatment group vs. control group.",
            "secondary": [
                "Increase in 'positive' interactions (e.g., supportive comments) from treatment group users.",
                "Recidivism Rate: Percentage of users who post harmful content again after a first intervention.",
                "User Engagement with Coach: Click-through rate on resources provided by the AI coach.",
                "Qualitative Feedback: Opt-in surveys to gauge user perception of the AI coach (helpful, annoying, creepy)."
            ]
        },
        "sample_size_calculation": "Perform a power analysis based on a desired minimum detectable effect size (e.g., a 5% reduction in harmful posts), a significance level (alpha=0.05), and power (beta=0.8). This will determine the number of participants needed per group.",
        "bias_mitigation": [
            "Stratified Sampling: If randomization shows imbalance, use stratified sampling to ensure groups are balanced on key demographics (age, gender, platform).",
            "Regular Bias Audits: Continuously run the `analyze_algorithm_bias` function on live data to monitor for performance drift or emergent biases."
        ]
    }
    return study_design

def analyze_algorithm_bias(predictions: pd.DataFrame, demographics: pd.DataFrame) -> Dict:
    """
    Detects if the algorithm has demographic bias using model performance metrics.
    `predictions` DataFrame needs 'post_id', 'user_id', 'prediction', 'ground_truth'.
    `demographics` DataFrame needs 'user_id', 'gender', 'race', 'age_group'.
    """
    data = pd.merge(predictions, demographics, on='user_id')
    bias_report = {}
    
    demographic_groups = ['gender', 'race', 'age_group']
    
    for group in demographic_groups:
        group_report = {}
        for value in data[group].unique():
            subset = data[data[group] == value]
            if len(subset) == 0: continue
            
            # Calculate False Positive Rate (FPR) for each subgroup
            # FPR = FP / (FP + TN)
            fp = len(subset[(subset['prediction'] == 1) & (subset['ground_truth'] == 0)])
            tn = len(subset[(subset['prediction'] == 0) & (subset['ground_truth'] == 0)])
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Calculate False Negative Rate (FNR)
            # FNR = FN / (FN + TP)
            fn = len(subset[(subset['prediction'] == 0) & (subset['ground_truth'] == 1)])
            tp = len(subset[(subset['prediction'] == 1) & (subset['ground_truth'] == 1)])
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            group_report[value] = {'False_Positive_Rate': fpr, 'False_Negative_Rate': fnr}
            
        bias_report[group] = group_report
        
    return bias_report


if __name__ == "__main__":
    import json 
    
    print("--- Part 2: Research Design ---")
    
    # 1. Validation Study Design
    print("\n[A] A/B Test Validation Study Design:")
    study_plan = design_validation_study()
    # Pretty-print the dictionary
    print(json.dumps(study_plan, indent=2))
    
    print("\n----------------------------------")
    print("\n[B] Algorithm Bias Analysis Function:")
    print("The `analyze_algorithm_bias` function is a utility designed to be imported and used with real prediction data.")
    print("It cannot be run standalone without predictions and demographics dataframes.")