"""
Strategy idea generation prompts
Used to generate innovative trading strategy ideas
"""

# Prompt for market parameter extraction from user query
MARKET_PARAMS_EXTRACTION_PROMPT = """
Extract market analysis parameters from this query about trading strategies.
Return a JSON object with these fields:
{
  "market_conditions": "Brief description of market conditions mentioned",
  "asset_class": "Asset class mentioned (equities, forex, crypto, commodities, etc.)",
  "risk_appetite": "Risk level mentioned (low, medium, high)"
}

Query: "{query}"

Use reasonable defaults for any missing fields.
"""

# Default market parameters when extraction fails
DEFAULT_MARKET_PARAMS = {
    "market_conditions": "current volatile market",
    "asset_class": "equities", 
    "risk_appetite": "medium"
}

# Main prompt for generating trading strategy ideas
STRATEGY_IDEAS_PROMPT = """
Generate {count} trading strategy ideas based on the following:

Market Conditions: {market_conditions}
Asset Class: {asset_class}
Risk Appetite: {risk_appetite}

For each strategy, provide:
1. Name
2. Strategy Type (momentum, mean_reversion, trend_following, breakout, etc.)
3. Key Parameters
4. Entry Rules
5. Exit Rules
6. Risk Management Parameters
7. Expected Performance

Format as a JSON array of strategy objects, like this:
[
  {{
    "name": "Strategy Name",
    "type": "Strategy Type",
    "description": "Brief explanation",
    "parameters": {{"param1": value1, "param2": value2}},
    "entry_rules": ["rule1", "rule2"],
    "exit_rules": ["rule1", "rule2"],
    "risk_params": {{"position_size": 0.1, "stop_loss": 0.02, "take_profit": 0.05}},
    "expected_performance": {{"sharpe": 1.5, "annual_return": 0.18, "max_drawdown": -0.15}}
  }},
  ...
]
"""