"""
Base module for Daedalus prompts
Contains utility functions for prompt management and formatting
"""

def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with provided variables
    
    Args:
        template: The prompt template string with placeholders
        **kwargs: Variables to insert into the template
        
    Returns:
        Formatted prompt string
    """
    return template.format(**kwargs)
    
# Common styling elements for prompts
PROMPT_STYLING = {
    "heading": "\n\n---\n{}\n---\n\n",
    "section": "\n## {}\n",
    "data_section": "\n{}: {}\n",
}

# Temperature presets for different prompt types
TEMPERATURE_PRESETS = {
    "classification": 0.1,   # Low temperature for deterministic results
    "extraction": 0.2,       # Slightly higher for parameter extraction
    "explanation": 0.3,      # Balanced for coherent explanations
    "creativity": 0.7,       # Higher for strategy ideas generation
}

# Token limits for different operations
TOKEN_LIMITS = {
    "classification": 20,
    "extraction": 1000,
    "explanation": 200,
    "ideas": 2000,
}