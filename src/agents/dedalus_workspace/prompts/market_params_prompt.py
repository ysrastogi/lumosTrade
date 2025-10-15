"""Market Parameters Prompt.

This module contains the prompt for extracting market analysis parameters.
"""

MARKET_PARAMS_PROMPT = """
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