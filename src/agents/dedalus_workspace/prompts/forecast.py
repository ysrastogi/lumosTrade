"""
Monte Carlo forecast parameter extraction prompts
Used for extracting forecast parameters from user queries
"""

# Prompt for extracting Monte Carlo forecast parameters
FORECAST_PARAMS_PROMPT = """
Extract Monte Carlo simulation parameters from the query.
Return a JSON object with these fields:
{
  "n_days": (int, number of days to forecast, default 252),
  "n_paths": (int, number of simulation paths, default 10000)
}

Query: "{query}"

Extract what you can, use defaults for missing values.
"""

# Default forecast parameters when extraction fails
DEFAULT_FORECAST_PARAMS = {
    "n_days": 252,
    "n_paths": 10000
}