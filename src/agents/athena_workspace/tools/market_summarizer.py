"""
Market summarization module for generating market commentary.

This module generates human-readable summaries and insights about
market conditions using both rule-based and LLM-based approaches.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from config.settings import settings
from src.agents.athena_workspace.tools.regime_detection import RegimeDetector
from src.llm.client import gemini

logger = logging.getLogger(__name__)

class MarketSummarizer:

    def __init__(self, use_llm: bool = True, api_key: Optional[str] = None):

        self.use_llm = use_llm
        self.llm_client = gemini
        logger.info("LLM client initialized successfully")


    def generate_llm_summary(self, symbol: str, features: Dict, regime: str, patterns: List[Dict],
                             time_horizon: str = "short-term") -> str:
            
        regime_description = RegimeDetector.get_regime_description(regime)
        
        # Extract the most relevant features
        relevant_features = {
            'price': features.get('close_price'),
            'rsi': features.get('rsi'),
            'volatility': features.get('volatility'),
            'macd': features.get('macd'),
            'bb_width': features.get('bb_width'),
            'price_change_pct': features.get('price_change_pct')
        }
        
        # Simplify patterns for the prompt
        simple_patterns = []
        for p in patterns[:5]:  # Top 5 patterns
            simple_patterns.append({
                "type": p["type"],
                "description": p["description"],
                "bias": p["bias"],
                "confidence": p["confidence"]
            })
            
        # Build the prompt for the LLM
        prompt = f"""
        You are Athena, a market intelligence AI analyzing {symbol}.

        Current Market State:
        - Price: {features.get('close_price', 'N/A')}
        - RSI: {features.get('rsi', 'N/A')}
        - Volatility: {features.get('volatility', 'N/A')}
        - Market Regime: {regime_description}

        Detected Patterns:
        {json.dumps(simple_patterns, indent=2)}

        Based on the technical analysis above, provide a concise 3-4 sentence summary for a {time_horizon} trader.
        Be specific about:
        1. Current market conditions
        2. Key technical signals and their implications
        3. Potential trade setup or recommendation
        4. Risk factors to watch for

        Use emoji where appropriate. Format the output as markdown.
        """

        try:
            print("Generating LLM summary with prompt:")
            print(prompt)
            # Use our new client with additional generation parameters
            llm_summary = self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=2000
            )
            
            # Log and return the summary
            print(llm_summary)
            return llm_summary
                
        except Exception as e:
            logger.error(f"Error generating LLM summary: {e}", exc_info=True)
            return None

        
    def generate_trade_ideas(self, symbol: str, features: Dict, patterns: List[Dict], 
                             time_horizon: str = "intraday") -> List[Dict]:
        trade_ideas = []
        
        # Skip if no patterns or errors
        if not patterns or 'error' in features:
            return trade_ideas
        
        current_price = features.get('close_price', 0)
        volatility = features.get('volatility', 0.01)
        atr = features.get('atr', current_price * 0.01)  # Default to 1% of price if not available
        
        # Process each high-confidence pattern (>65% confidence)
        for pattern in [p for p in patterns if p['confidence'] > 65]:
            # Default values
            stop_distance = atr * 1.5
            target_distance = atr * 3
            
            trade_idea = {
                "symbol": symbol,
                "pattern": pattern['type'],
                "direction": "buy" if pattern['bias'] == 'bullish' else "sell",
                "confidence": pattern['confidence'],
                "description": pattern['description'],
                "time_horizon": time_horizon,
                "entry": current_price,
                "stop_loss": current_price - stop_distance if pattern['bias'] == 'bullish' else current_price + stop_distance,
                "target": current_price + target_distance if pattern['bias'] == 'bullish' else current_price - target_distance,
                "risk_reward": target_distance / stop_distance,
                "timestamp": datetime.now().isoformat()
            }
            
            # Adjust for specific patterns
            if pattern['type'] == 'volatility_squeeze':
                # Breakout setups need confirmation - use "wait for breakout"
                trade_idea['entry'] = "Wait for breakout direction"
                trade_idea['direction'] = "pending"
                
            elif pattern['type'] in ['bb_lower_bounce', 'oversold_rebound']:
                # For bullish reversals, set tighter stops
                trade_idea['stop_loss'] = current_price - (atr * 1.0)
                
            elif pattern['type'] in ['bb_upper_rejection', 'overbought_reversal']:
                # For bearish reversals, set tighter stops
                trade_idea['stop_loss'] = current_price + (atr * 1.0)
                
            trade_ideas.append(trade_idea)
            
        return trade_ideas