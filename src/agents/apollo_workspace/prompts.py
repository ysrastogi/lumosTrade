"""
Prompt templates for Apollo Signal Analysis Agent.

This module contains all the prompt templates used by Apollo for generating
signal explanations, reasoning, trade validations, and other LLM-driven analyses.
"""

from typing import Dict, List, Optional
import numpy as np
from src.agents.apollo_workspace.models import Signal

class ApolloPrompts:
    """Collection of prompt templates for the Apollo agent."""
    
    @staticmethod
    def reasoning_prompt(signal: Signal, 
                        supporting_factors: List[str],
                        risk_factors: List[str],
                        probabilities: Dict[str, float],
                        market_data: Dict) -> str:
        """
        Generate a prompt for signal reasoning.
        
        Args:
            signal: Trading signal being analyzed
            supporting_factors: List of factors supporting the signal
            risk_factors: List of risk factors for the signal
            probabilities: Dictionary of probability metrics
            market_data: Market context data
            
        Returns:
            Formatted prompt for LLM
        """
        return f"""
You are Apollo, an expert trading analyst. Explain this trade signal in 3-4 sentences.

SIGNAL DETAILS:
- Symbol: {signal.symbol}
- Pattern: {signal.pattern}
- Direction: {signal.direction.upper()}
- Entry: {signal.entry}
- Stop Loss: {signal.stop_loss}
- Target: {signal.target}
- Risk/Reward: {signal.risk_reward}:1
- Confidence: {signal.confidence:.1f}%

MARKET CONTEXT:
- Regime: {market_data['regime']}
- Regime Confidence: {market_data['regime_confidence']*100:.0f}%
- Summary: {market_data['summary']}

SUPPORTING FACTORS:
{chr(10).join(f'- {f}' for f in supporting_factors)}

RISK FACTORS:
{chr(10).join(f'- {f}' for f in risk_factors)}

PROBABILITY ANALYSIS:
- Historical Win Rate: {probabilities['base_win_rate']*100:.1f}%
- Adjusted Win Rate: {probabilities['adjusted_win_rate']*100:.1f}%
- Expected Value: {probabilities['expected_value']:.2f}R
- Sample Size: {probabilities['sample_size']} similar signals

INSTRUCTIONS:
1. Start with the core thesis (WHY this setup works)
2. Mention 2-3 strongest supporting factors
3. Acknowledge the primary risk
4. End with a probability-informed recommendation

Tone: Confident but measured. Use trader terminology.
Length: 3-4 sentences maximum.
"""

    @staticmethod
    def invalidation_criteria_prompt(signal: Signal, market_data: Dict) -> str:
        """
        Generate a prompt for invalidation criteria.
        
        Args:
            signal: Trading signal being analyzed
            market_data: Market context data
            
        Returns:
            Formatted prompt for LLM
        """
        return f"""
You are Apollo, an expert trading analyst. Identify the 5 most important conditions that would invalidate the following trade setup.

SIGNAL DETAILS:
- Symbol: {signal.symbol}
- Pattern: {signal.pattern}
- Direction: {signal.direction.upper()}
- Entry: {signal.entry}
- Stop Loss: {signal.stop_loss}
- Target: {signal.target}
- Risk/Reward: {signal.risk_reward}:1

MARKET CONTEXT:
- Regime: {market_data['regime']}
- Regime Confidence: {market_data['regime_confidence']*100:.0f}%

List ONLY the 5 most critical invalidation conditions in bullet points. Each should be a specific, observable market event that would invalidate the trade thesis BEFORE the stop loss is hit.

Examples:
- "Price breaks below the 50-period moving average"
- "Volume declines by more than 25% during the initial move"
- "A bearish engulfing pattern forms at resistance level"

Be specific to this setup, not generic invalidation criteria.
"""

    @staticmethod
    def historical_comparison_prompt(signal: Signal, similar_signals: List[Signal]) -> str:
        """
        Generate a prompt for historical comparison.
        
        Args:
            signal: Trading signal being analyzed
            similar_signals: List of similar historical signals
            
        Returns:
            Formatted prompt for LLM
        """
        # Get top 3 most similar (by outcome)
        wins = [s for s in similar_signals if s.outcome == 'win'][:3]
        losses = [s for s in similar_signals if s.outcome == 'loss'][:3]
        
        # Calculate average profits/losses if data exists
        avg_win = np.mean([s.pnl for s in wins if s.pnl]) if wins else 0
        avg_loss = np.mean([abs(s.pnl) for s in losses if s.pnl]) if losses else 0
        
        return f"""
Analyze this current trade setup:
{signal.pattern} on {signal.symbol} - {signal.direction} at {signal.entry}

Historical similar setups:
Winning examples: {len(wins)} signals (avg R: {avg_win:.2f})
Losing examples: {len(losses)} signals (avg R: {avg_loss:.2f})

Provide a 2-3 sentence comparison highlighting:
1. What made the winning setups successful
2. What caused the losing setups to fail
3. How the current setup compares

Be specific and actionable.
"""

    @staticmethod
    def confluence_analysis_prompt(signal: Signal, market_data: Dict) -> str:
        """
        Generate a prompt for confluence analysis.
        
        Args:
            signal: Trading signal being analyzed
            market_data: Market context data
            
        Returns:
            Formatted prompt for LLM
        """
        return f"""
You are Apollo, an expert trading analyst. Analyze the confluence factors for this trade setup.

SIGNAL DETAILS:
- Symbol: {signal.symbol}
- Pattern: {signal.pattern}
- Direction: {signal.direction.upper()}
- Entry: {signal.entry}
- Stop Loss: {signal.stop_loss}
- Target: {signal.target}
- Risk/Reward: {signal.risk_reward}:1

MARKET CONTEXT:
- Regime: {market_data['regime']}
- Regime Confidence: {market_data['regime_confidence']*100:.0f}%
- Market Summary: {market_data.get('summary', 'Not available')}

Based on the above, provide:
1. Four specific supporting factors that increase the probability of success for this setup
2. Three specific risk factors that could lead to failure

FORMAT YOUR RESPONSE AS JSON:
{{
  "supporting_factors": [
    "factor 1",
    "factor 2",
    "factor 3",
    "factor 4"
  ],
  "risk_factors": [
    "risk 1",
    "risk 2",
    "risk 3"
  ]
}}

Be specific, market-focused, and use trader terminology.
"""

    @staticmethod
    def probability_calculation_prompt(signal: Signal, 
                                     historical_data: Dict, 
                                     market_data: Dict) -> str:
        """
        Generate a prompt for probability calculation.
        
        Args:
            signal: Trading signal being analyzed
            historical_data: Historical data for similar setups
            market_data: Market context data
            
        Returns:
            Formatted prompt for LLM
        """
        base_win_rate = historical_data.get('win_rate', 0.5)
        sample_size = historical_data.get('sample_size', 0)
        
        return f"""
You are Apollo, an expert trading probabilistic analyst. Calculate the adjusted win rate for this trade setup.

SIGNAL DETAILS:
- Symbol: {signal.symbol}
- Pattern: {signal.pattern}
- Direction: {signal.direction.upper()}
- Risk/Reward: {signal.risk_reward}:1

HISTORICAL DATA:
- Base Win Rate: {base_win_rate*100:.1f}% (based on {sample_size} similar signals)

MARKET CONTEXT:
- Regime: {market_data['regime']}
- Regime Confidence: {market_data['regime_confidence']*100:.0f}%

Based on the base win rate, adjust the probability considering:
1. Current market regime alignment with the signal direction
2. Signal confidence level
3. Risk/reward ratio
4. Sample size reliability (smaller samples are less reliable)

CALCULATE:
1. Adjusted Win Rate (%)
2. Expected Value (in R - multiple of risk)
3. Kelly Criterion position size (%)

FORMAT YOUR RESPONSE AS JSON:
{{
  "base_win_rate": {base_win_rate},
  "adjusted_win_rate": [calculated value between 0-1],
  "expected_value": [calculated EV in R],
  "kelly_position_size": [calculated Kelly % between 0-100],
  "reasoning": "Brief explanation of adjustments"
}}

Use proper probabilistic reasoning.
"""

    @staticmethod
    def signal_validation_prompt(signal: Signal, 
                               backtest_data: Dict,
                               market_data: Dict) -> str:
        """
        Generate a prompt for signal validation.
        
        Args:
            signal: Trading signal being analyzed
            backtest_data: Backtest data for validation
            market_data: Market context data
            
        Returns:
            Formatted prompt for LLM
        """
        return f"""
You are Apollo, an expert trading validator. Assess whether this trade setup meets validation criteria.

SIGNAL DETAILS:
- Symbol: {signal.symbol}
- Pattern: {signal.pattern}
- Direction: {signal.direction.upper()}
- Risk/Reward: {signal.risk_reward}:1
- Confidence: {signal.confidence:.1f}%

BACKTEST DATA:
- Win Rate: {backtest_data.get('win_rate', 0)*100:.1f}%
- Profit Factor: {backtest_data.get('profit_factor', 1):.2f}
- Sample Size: {backtest_data.get('sample_size', 0)} trades

MARKET CONTEXT:
- Regime: {market_data['regime']}
- Regime Alignment: {'Aligned' if (market_data['regime'] == 'bullish' and signal.direction == 'buy') or (market_data['regime'] == 'bearish' and signal.direction == 'sell') else 'Misaligned'}

VALIDATION CRITERIA:
1. Win rate > 40%
2. Profit factor > 1.2
3. Sample size > 20 trades
4. Risk/reward > 1.5
5. Market regime alignment with direction

FORMAT YOUR RESPONSE AS JSON:
{{
  "valid": true/false,
  "reason": "Explanation of validation decision",
  "strength_score": [value between 1-10],
  "recommendations": ["recommendation 1", "recommendation 2"]
}}

Be thorough but concise.
"""

    @staticmethod
    def signal_generator_prompt(market_data: Dict) -> str:
        """
        Generate a prompt for signal generation.
        
        Args:
            market_data: Market context data
            
        Returns:
            Formatted prompt for LLM
        """
        symbol = market_data.get('symbol', 'UNKNOWN')
        patterns = market_data.get('patterns', [])
        regime = market_data.get('regime', 'unknown')
        features = market_data.get('features', {})
        
        # Format patterns for the prompt
        patterns_text = ""
        for p in patterns[:5]:
            patterns_text += f"- {p.get('type', 'unknown')}: {p.get('description', 'No description')} ({p.get('bias', 'neutral')}, {p.get('confidence', 0)}%)\n"
        
        if not patterns_text:
            patterns_text = "- No significant patterns detected\n"
        
        # Format key features
        price = features.get('close_price', 'unknown')
        rsi = features.get('rsi', 'unknown')
        volatility = features.get('volatility', 'unknown')
        
        return f"""
You are Apollo, an expert trading signal generator. Create actionable trade signals based on this market data.

MARKET ANALYSIS FOR {symbol}:
- Current Price: {price}
- Market Regime: {regime}
- RSI: {rsi}
- Volatility: {volatility}

DETECTED PATTERNS:
{patterns_text}

Based on this analysis, generate up to 2 high-probability trade setups.
For each setup, provide:

1. Pattern name
2. Direction (buy/sell)
3. Entry price or condition
4. Stop loss level
5. Target price
6. Risk-to-reward ratio
7. Confidence score (0-100%)
8. Time horizon (intraday, swing, position)

FORMAT YOUR RESPONSE AS JSON:
{{
  "signals": [
    {{
      "pattern": "pattern name",
      "direction": "buy/sell",
      "entry": "specific price or condition",
      "stop_loss": "specific price",
      "target": "specific price",
      "risk_reward": number,
      "confidence": number,
      "time_horizon": "intraday/swing/position"
    }}
  ]
}}

Only generate signals if there are clear setups with edge. Otherwise return an empty signals array.
"""