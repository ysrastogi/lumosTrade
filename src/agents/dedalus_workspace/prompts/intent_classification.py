"""
Intent classification prompts for Daedalus agent
Used to determine the user's intent from their query
"""

# Define the intents and their associated keywords
INTENT_KEYWORDS = {
    "create_strategy": ["create", "design", "build", "new strategy", "configure"],
    "simulate": ["simulate", "backtest", "test", "run", "execute"],
    "optimize": ["optimize", "tune", "find best", "parameter search", "improve"],
    "analyze": ["analyze", "walk forward", "stress test", "validate", "robust"],
    "forecast": ["forecast", "predict", "monte carlo", "future", "project"],
    "compare": ["compare", "versus", "vs", "which is better", "rank"],
    "status": ["status", "report", "state", "summary", "what have", "show me"],
    "generate_ideas": ["ideas", "suggest", "recommend", "brainstorm", "generate strategy"]
}

# Prompt for advanced intent classification using LLM
INTENT_CLASSIFICATION_PROMPT = """
Analyze the following query and classify the intent into one of these categories:
- create_strategy: Creating or designing a new trading strategy
- simulate: Running a backtest or simulation
- optimize: Finding optimal parameters for a strategy
- analyze: Performing analysis like walk-forward or stress testing
- forecast: Generating Monte Carlo forecasts or predictions
- compare: Comparing different strategies or approaches
- status: Requesting status reports or summaries
- generate_ideas: Creating innovative trading strategy ideas
- unknown: If none of the above apply

Query: "{query}"

Return only the category name, nothing else.
"""

# List of all valid intents including unknown
VALID_INTENTS = list(INTENT_KEYWORDS.keys()) + ["unknown"]