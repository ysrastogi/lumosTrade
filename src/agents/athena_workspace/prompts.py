"""
Prompt templates for Athena Market Intelligence Agent.

This module contains all the prompt templates used by Athena for generating
market summaries, trade ideas, and other LLM-driven analyses.
"""

import json
from typing import Dict, List, Optional
from src.agents.athena_workspace.tools.regime_detection import RegimeDetector

class AthenaPrompts:
    """Collection of prompt templates for the Athena agent."""
    
    @staticmethod
    def market_summary_prompt(symbol: str, 
                             features: Dict, 
                             regime: str, 
                             patterns: List[Dict],
                             time_horizon: str = "short-term") -> str:
        """
        Generate a prompt for market summary.
        
        Args:
            symbol: Market symbol being analyzed
            features: Extracted technical features
            regime: Detected market regime
            patterns: List of detected patterns
            time_horizon: Time horizon for analysis
            
        Returns:
            Formatted prompt for LLM
        """
        
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
        
        return prompt
    
    @staticmethod
    def trade_idea_prompt(symbol: str, 
                         features: Dict, 
                         patterns: List[Dict],
                         time_horizon: str = "intraday") -> str:
        """
        Generate a prompt for trade idea generation.
        
        Args:
            symbol: Market symbol being analyzed
            features: Extracted technical features
            patterns: List of detected patterns
            time_horizon: Time horizon for trade
            
        Returns:
            Formatted prompt for LLM
        """
        # Extract key features for trade idea generation
        current_price = features.get('close_price', 0)
        volatility = features.get('volatility', 0.01)
        atr = features.get('atr', current_price * 0.01)
        rsi = features.get('rsi', 50)
        trend_direction = features.get('trend_direction', 'neutral')
        
        # Simplify patterns for the prompt
        simple_patterns = []
        for p in patterns[:3]:  # Top 3 patterns
            simple_patterns.append({
                "type": p["type"],
                "description": p["description"],
                "bias": p["bias"],
                "confidence": p["confidence"]
            })
            
        # Build the prompt for trade idea generation
        prompt = f"""
        You are Athena, a market intelligence AI generating {time_horizon} trade ideas for {symbol}.

        Current Market Data:
        - Price: {current_price}
        - Volatility: {volatility}
        - ATR: {atr}
        - RSI: {rsi}
        - Trend Direction: {trend_direction}

        Top Detected Patterns:
        {json.dumps(simple_patterns, indent=2)}

        Based on this analysis, generate a detailed trade idea with:
        1. Entry price or condition
        2. Stop loss level with reasoning
        3. Target price with reasoning
        4. Risk-reward ratio
        5. Key confirmation signals to look for
        6. Key invalidation signals

        Format as JSON with the following structure:
        {{
          "direction": "buy/sell",
          "entry": "price or condition",
          "stop_loss": "price with reasoning",
          "target": "price with reasoning",
          "risk_reward": "ratio",
          "confirmations": ["signal 1", "signal 2"],
          "invalidations": ["signal 1", "signal 2"]
        }}
        """
        
        return prompt
    
    @staticmethod
    def pattern_analysis_prompt(symbol: str, 
                               pattern_data: Dict,
                               price_history: List,
                               time_horizon: str = "short-term") -> str:
        """
        Generate a prompt for detailed pattern analysis.
        
        Args:
            symbol: Market symbol being analyzed
            pattern_data: Data about the detected pattern
            price_history: Recent price history data
            time_horizon: Time horizon for analysis
            
        Returns:
            Formatted prompt for LLM
        """
        # Create a condensed version of price history
        condensed_history = price_history[-10:]  # Last 10 price points
        
        # Build the prompt for pattern analysis
        prompt = f"""
        You are Athena, a market intelligence AI performing detailed analysis on a {pattern_data['type']} pattern detected in {symbol}.

        Pattern Details:
        {json.dumps(pattern_data, indent=2)}

        Recent Price Action:
        {json.dumps(condensed_history, indent=2)}

        Provide a detailed analysis of this {pattern_data['type']} pattern, including:
        1. Historical reliability of this pattern
        2. Key levels to watch (support, resistance, breakout levels)
        3. Typical duration and price targets for this pattern
        4. Common failure scenarios and warning signs
        5. Specific recommendations for a {time_horizon} trader

        Format your response as markdown with clear sections.
        """
        
        return prompt
    
    @staticmethod
    def market_context_prompt(symbol: str, 
                             recent_observations: List[Dict],
                             current_features: Dict,
                             memory_context: Optional[Dict] = None) -> str:
        """
        Generate a prompt for synthesizing market context from multiple observations.
        
        Args:
            symbol: Market symbol being analyzed
            recent_observations: List of recent market observations
            current_features: Current market features
            memory_context: Optional memory context data
            
        Returns:
            Formatted prompt for LLM
        """
        # Create a summary of recent observations
        observation_summary = []
        for i, obs in enumerate(recent_observations[:5]):  # Last 5 observations
            observation_summary.append({
                "timestamp": obs.get("timestamp", "unknown"),
                "regime": obs.get("regime", "unknown"),
                "trading_bias": obs.get("trading_bias", "neutral"),
                "key_patterns": [p["type"] for p in obs.get("patterns", [])[:2]]
            })
        
        # Include memory context if available
        memory_prompt = ""
        if memory_context:
            memory_prompt = f"""
            Historical Context from Memory System:
            - Prior Regimes: {memory_context.get('prior_regimes', [])}
            - Successful Patterns: {memory_context.get('successful_patterns', [])}
            - Important Support/Resistance: {memory_context.get('key_levels', {})}
            """
        
        # Build the prompt
        prompt = f"""
        You are Athena, a market intelligence AI synthesizing observations for {symbol}.

        Recent Market Observations:
        {json.dumps(observation_summary, indent=2)}

        Current Market Features:
        {json.dumps({k: v for k, v in current_features.items() if k in [
            'close_price', 'rsi', 'volatility', 'trend_direction', 'regime'
        ]}, indent=2)}
        {memory_prompt}

        Based on this series of observations, provide a comprehensive market context analysis:
        1. How the market regime has evolved
        2. Consistent patterns that have appeared repeatedly
        3. Key levels that have shown importance
        4. Current market positioning relative to historical context
        5. Most probable forward scenarios based on this context

        Format your response as markdown with clear sections.
        """
        
        return prompt