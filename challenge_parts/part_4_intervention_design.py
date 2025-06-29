from typing import Dict, List

def design_intervention_messages() -> Dict:
    """
    Creates compassionate, age-appropriate intervention message templates.
    """
    interventions = {
        "mild_bullying": {
            "message_template": [
                "Hey, we noticed your recent comment. Sometimes things we say online can come across differently than we mean them to. It's always a good idea to pause and reread before you post. Let's keep this a supportive space for everyone.",
                "Just a heads up, your message could be seen as hurtful. We're all about positive vibes here. Maybe try phrasing it another way? Learn more about respectful communication here: [Link to resource]"
            ],
            "tone": "Supportive & Educational",
            "call_to_action": "Encourage reflection and self-correction."
        },
        "moderate_bullying": {
            "message_template": [
                "This is a warning: Your recent comment violates our community guidelines against harassment. This kind of behavior is not okay and can seriously hurt people. Please review our rules here: [Link]. Further violations will result in account restrictions.",
            ],
            "tone": "Firm but Fair",
            "call_to_action": "State consequences and direct to rules."
        },
        "self_harm_risk": {
            "message_template": [
                "We're concerned about you. Your recent post suggests you might be going through a tough time. Please know you are not alone and help is available. You can connect with someone who can support you right now, for free and confidentially. Text or call 988 in the US & Canada, or visit [Link to Global Helpline Directory].",
                "It sounds like you are in a lot of pain. It’s important to talk to someone. If you are in immediate danger, please call 911. For support, you can reach out to trained counselors 24/7 at the Crisis Text Line by texting HOME to 741741."
            ],
            "tone": "Urgent & Caring",
            "call_to_action": "Provide immediate, direct links to professional help.",
            "resources": ["988 Suicide & Crisis Lifeline", "Crisis Text Line", "The Trevor Project"]
        }
    }
    return interventions

def measure_intervention_effectiveness() -> Dict[str, str]:
    """
    Designs metrics for measuring intervention success while respecting user privacy.
    """
    metrics_design = {
        "privacy_first_quantitative_metrics": {
            "recidivism_rate": "Aggregate, anonymized percentage of users who post another flagged comment within 7, 30, and 90 days of an intervention.",
            "de_escalation_rate": "Percentage of conversations where, after an intervention, the subsequent reply from the user is classified as 'neutral' or 'positive' by our sentiment model.",
            "community_health_score": "Overall percentage of flagged content in a community over time. A decrease suggests the interventions are improving the ecosystem's health."
        },
        "opt_in_qualitative_metrics": {
            "follow_up_micro_survey": "24 hours after an intervention, send a private, optional, one-question survey: 'Was our recent message helpful? (Yes/No/Unsure)'.",
            "user_perception_study": "Recruit a separate panel of users for a paid study to review intervention messages and provide detailed qualitative feedback on tone, helpfulness, and potential for improvement."
        },
        "attribution_challenges_and_solutions": {
            "challenge": "How do we know the AI coach caused the change, not other factors (e.g., peer influence, human moderation)?",
            "solution": "The A/B test (designed in Part 2) is the gold standard for attribution. By comparing the treatment group (with AI coach) to the control group (without), we can isolate the causal effect of the intervention."
        },
        "ethical_measurement": {
            "principle": "We will NEVER measure 'success' by tracking individual user journeys or analyzing their private mental health outcomes without explicit, informed consent for a formal research study.",
            "focus": "Our focus is on measuring the health of the platform and the effectiveness of the intervention content, not on surveilling individuals."
        }
    }
    return metrics_design

if __name__ == "__main__":
    import json
    
    print("--- Part 4: Intervention Design ---")
    
    # 1. Show the Intervention Message Templates
    print("\n[A] Compassionate Intervention Message Templates:")
    messages = design_intervention_messages()
    print(json.dumps(messages, indent=2))

    # 2. Show the Effectiveness Measurement Plan
    print("\n[B] Plan for Measuring Intervention Effectiveness:")
    metrics = measure_intervention_effectiveness()
    print(json.dumps(metrics, indent=2))