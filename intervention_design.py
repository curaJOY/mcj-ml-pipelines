

"""
Design the AI coach intervention system
"""

def design_intervention_messages():
    """
    TODO: Create compassionate intervention templates
    
    Requirements:
    - Age-appropriate language for teens
    - Non-judgmental tone
    - Actionable guidance
    - Cultural sensitivity
    - De-escalation focus
    """
    
    interventions = {
        "mild_bullying": {
            "message_template": (
                "Hey there — just a quick reminder: words matter. "
                "Please keep things respectful, even when joking. "
                "This might hurt someones, it’s okay to pause and reword."
            ),
            "tone": "supportive",
            "call_to_action": "Reflect and consider editing or deleting the comment."
            },
        
        "severe_bullying": {
            "message_template": (
                "This comment might be harmful. Before it goes further, please take a moment. "
                "We’re here to keep everyone safe and respected. "
                "Your hurtful language can have lasting effects."
            ),
            "tone": "firm but non-judgmental",
            "call_to_action": "Reconsider posting. Seek support if you’re feeling overwhelmed."
        },
        
        "self_harm_risk": {
            "message_template": (
                "You're not alone. If you’re struggling, there’s help available. "
                "Please consider talking to someone you trust or reaching out to a support line."
            ),
            "tone": "caring",
            "resources": [
                "Crisis Text Line: Text HOME to 741741",
                "Teen Lifeline: 800-248-8336",
                "988 Suicide & Crisis Lifeline"
            ],
        }
    }
    return interventions

def measure_intervention_effectiveness():
    """
    TODO: Design metrics for measuring intervention success
    
    Challenges:
    - Privacy: Can't track individuals long-term
    - Attribution: Was improvement due to AI or other factors?
    - Timing: Short-term vs long-term effects
    - Ethics: Measuring sensitive mental health outcomes
    """

    metrics = {
        "short_term": {
            "post_change_rate": "Percentage of flagged posts edited or deleted after receiving an intervention message.",
            "tone_shift": "Sentiment analysis comparing tone before and after interventions.",
            "repeat_offense_reduction": "Change in number of flagged posts per user over 3 days."
        },
        "long_term": {
            "reoffend_rate": "User IDs are hashed and not linked to personal data."+ 
                            "Reoffense metrics are monitored over a short period (max 2 weeks) to"+
                            " assess immediate intervention impact, after which data is deleted.",
            "positive_engagement": "Increase in constructive, respectful posts post-intervention.",
            "self-help_engagement": "Clicks or usage of linked mental health resources.",
            "control_comparison": "Compare outcomes between non-intervention group and group with intervention "+
                                  "to access the impact of the AI"
        },
        "ethical_safeguards": [
            "Make all logs and activity data anonymous before analysis.",
            "No login or personal information required to access help resources.",
            "Consult mental health professionals when designing evaluation thresholds."
        ]
    }



    
    pass
