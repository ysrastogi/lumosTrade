"""
Debate summary prompts for the Hermes Consensus Mediator Agent.
"""

def get_debate_summary_prompt(structured_summary: str) -> str:
    """
    Generate a prompt for summarizing agent debates.
    
    Args:
        structured_summary: A string containing the basic structured summary of agent votes.
        
    Returns:
        The debate summary prompt as a string.
    """
    return f"""Summarize the following agent debate in 3 clear, insightful bullets:

{structured_summary}

Focus on the key points of disagreement and the relative confidence levels.
"""