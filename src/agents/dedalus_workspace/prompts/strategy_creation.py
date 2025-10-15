"""
Strategy creation and parameter extraction prompts
Used to extract trading strategy specifications from user queries
"""

# Main prompt for extracting strategy configuration from user query
STRATEGY_EXTRACTION_PROMPT = """
Extract trading strategy parameters from the following query. Return a JSON object with these fields:

{
  "name": "Strategy name (use a descriptive name)",
  "strategy_type": "One of: momentum, mean_reversion, trend_following, breakout, statistical_arbitrage, options_strategy",
  "parameters": {Extracted parameters as key-value pairs},
  "entry_rules": ["List of entry rules as strings"],
  "exit_rules": ["List of exit rules as strings"],
  "risk_params": {
    "position_size": float (0-1),
    "stop_loss": float (0-1), 
    "take_profit": float (0-1)
  }
}

Query: "{query}"

Parse as much as you can from the query. For any missing values, use sensible defaults.
For parameters, extract any numerical values mentioned (like periods, thresholds).
"""

# Default strategy configuration to use when extraction fails
DEFAULT_STRATEGY = {
    "name": "Custom_Strategy",
    "strategy_type": "momentum",
    "parameters": {"period": 20},
    "entry_rules": ["condition_1"],
    "exit_rules": ["condition_2"],
    "risk_params": {
        "position_size": 0.1,
        "stop_loss": 0.02,
        "take_profit": 0.05
    }
}

# Strategy types and their associated keywords for rule-based extraction
STRATEGY_TYPE_KEYWORDS = {
    "momentum": ["momentum", "trend", "moving average", "ma"],
    "mean_reversion": ["mean reversion", "reversion", "rsi", "overbought", "oversold"],
    "trend_following": ["trend following", "trend", "following", "adx"],
    "breakout": ["breakout", "break out", "range break", "channel"],
    "statistical_arbitrage": ["statistical arbitrage", "stat arb", "pair", "correlation"],
    "options_strategy": ["option", "call", "put", "strike", "expiry"]
}