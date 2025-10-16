"""
Override explanation prompts for the Hermes Consensus Mediator Agent.
"""

from typing import Dict, Any

def get_override_explanation_prompt(context_summary: str) -> str:
    """
    Generate a prompt for explaining agent overrides.
    
    Args:
        context_summary: A string containing details about the override context.
        
    Returns:
        The override explanation prompt as a string.
    """
    return f"""Explain why one agent overrode another in a decision process:

{context_summary}

Provide a clear, professional explanation highlighting:
1. Why the override happened (trust score, confidence, expertise)
2. The implications of this override
3. A concise analysis of the trade-offs involved
"""

def get_override_context_summary(
    overriding_agent: str, 
    overridden_agent: str, 
    context: Dict[str, Any]
) -> str:
    """
    Generate a summary of the override context.
    
    Args:
        overriding_agent: The name of the agent that overrode the decision.
        overridden_agent: The name of the agent that was overridden.
        context: A dictionary containing additional context about the override.
        
    Returns:
        A formatted string summarizing the override context.
    """
    return f"""
Overriding agent: {overriding_agent}
Overridden agent: {overridden_agent}
Overriding agent trust score: {context.get('override_trust', 0.0):.2f}
Overridden agent trust score: {context.get('overridden_trust', 0.0):.2f}
Confidence difference: {context.get('confidence_delta', 0.0):.2%}
Context reason: {context.get('reason', 'Higher trust weight and confidence')}
"""

def get_fallback_override_explanation(
    overriding_agent: str, 
    overridden_agent: str, 
    context: Dict[str, Any]
) -> str:
    """
    Generate a fallback template-based override explanation.
    
    Args:
        overriding_agent: The name of the agent that overrode the decision.
        overridden_agent: The name of the agent that was overridden.
        context: A dictionary containing additional context about the override.
        
    Returns:
        A formatted string explaining the override.
    """
    return f"""Agent Override Analysis:

{overriding_agent.upper()} overrode {overridden_agent.upper()}

Reason: {context.get('reason', 'Higher trust weight and confidence')}
Trust Score: {overriding_agent} ({context.get('override_trust', 0.0):.2f}) vs {overridden_agent} ({context.get('overridden_trust', 0.0):.2f})
Confidence Delta: {context.get('confidence_delta', 0.0):.2%}

The system prioritized {overriding_agent}'s assessment based on:
1. Superior trust score from historical performance
2. Higher confidence level in current assessment
3. Specialization alignment with current context"""