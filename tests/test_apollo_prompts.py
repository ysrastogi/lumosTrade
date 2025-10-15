"""
Tests for the Apollo prompts module.

This module contains tests to verify that the Apollo prompts module is working correctly
and generating appropriate prompts for different scenarios.
"""

import unittest
from datetime import datetime
from src.agents.apollo_workspace.prompts import ApolloPrompts
from src.agents.apollo_workspace.models import Signal, SignalStatus

class TestApolloPrompts(unittest.TestCase):
    """Test cases for Apollo prompts module."""
    
    def setUp(self):
        """Set up test data"""
        # Sample signal for testing
        self.test_signal = Signal(
            id="test-signal-1",
            symbol="BTC/USD",
            pattern="double_bottom",
            direction="buy",
            entry=50000,
            stop_loss=49000,
            target=53000,
            risk_reward=3.0,
            confidence=75.0,
            timestamp=datetime.fromisoformat("2025-10-10T12:00:00Z"),
            time_horizon="intraday",
            description="Double bottom pattern with strong support",
            reasoning="",
            invalidation_criteria=[],
            supporting_factors=[],
            similar_historical_count=0,
            historical_win_rate=0.0
        )
        
        # Sample supporting factors
        self.supporting_factors = [
            "Strong support level at 49000",
            "Bullish RSI divergence",
            "Increasing volume on bounces"
        ]
        
        # Sample risk factors
        self.risk_factors = [
            "Overhead resistance at 51000",
            "Decreasing overall market liquidity",
            "Potential bearish macro news pending"
        ]
        
        # Sample probabilities
        self.probabilities = {
            "base_win_rate": 0.65,
            "adjusted_win_rate": 0.7,
            "expected_value": 1.5,
            "sample_size": 42
        }
        
        # Sample market data
        self.market_data = {
            "regime": "bullish",
            "regime_confidence": 0.8,
            "summary": "Market showing strong upward momentum with decreasing volatility"
        }
        
        # Sample similar signals
        self.similar_signals = [
            Signal(
                id="historical-signal-1",
                symbol="BTC/USD",
                pattern="double_bottom",
                direction="buy",
                entry=48000,
                stop_loss=47000,
                target=51000,
                risk_reward=3.0,
                confidence=70.0,
                outcome="win",
                pnl=2.5,
                timestamp=datetime.fromisoformat("2025-09-10T12:00:00Z"),
                time_horizon="intraday",
                description="Double bottom pattern with strong support",
                reasoning="Strong signal with increasing volume",
                invalidation_criteria=["Price breaks below support"],
                supporting_factors=["Increasing volume", "RSI divergence"],
                similar_historical_count=15,
                historical_win_rate=0.7,
                status=SignalStatus.TARGET_HIT
            ),
            Signal(
                id="historical-signal-2",
                symbol="BTC/USD",
                pattern="double_bottom",
                direction="buy",
                entry=46000,
                stop_loss=45000,
                target=49000,
                risk_reward=3.0,
                confidence=65.0,
                outcome="loss",
                pnl=-1.0,
                timestamp=datetime.fromisoformat("2025-08-15T12:00:00Z"),
                time_horizon="intraday",
                description="Double bottom pattern with weak support",
                reasoning="Moderate signal with flat volume",
                invalidation_criteria=["Price breaks below support"],
                supporting_factors=["RSI divergence"],
                similar_historical_count=10,
                historical_win_rate=0.6,
                status=SignalStatus.STOPPED
            )
        ]
        
    def test_reasoning_prompt(self):
        """Test reasoning prompt generation"""
        prompt = ApolloPrompts.reasoning_prompt(
            signal=self.test_signal,
            supporting_factors=self.supporting_factors,
            risk_factors=self.risk_factors,
            probabilities=self.probabilities,
            market_data=self.market_data
        )
        
        # Check that the prompt includes key elements
        self.assertIn(self.test_signal.symbol, prompt)
        self.assertIn(self.test_signal.pattern, prompt)
        self.assertIn(str(self.test_signal.entry), prompt)
        self.assertIn(self.supporting_factors[0], prompt)
        self.assertIn(self.risk_factors[0], prompt)
        self.assertIn("PROBABILITY ANALYSIS", prompt)
        
    def test_invalidation_criteria_prompt(self):
        """Test invalidation criteria prompt generation"""
        prompt = ApolloPrompts.invalidation_criteria_prompt(
            signal=self.test_signal,
            market_data=self.market_data
        )
        
        # Check that the prompt includes key elements
        self.assertIn(self.test_signal.symbol, prompt)
        self.assertIn(self.test_signal.pattern, prompt)
        self.assertIn("invalidate", prompt.lower())
        self.assertIn("observable market event", prompt.lower())
        
    def test_historical_comparison_prompt(self):
        """Test historical comparison prompt generation"""
        prompt = ApolloPrompts.historical_comparison_prompt(
            signal=self.test_signal,
            similar_signals=self.similar_signals
        )
        
        # Check that the prompt includes key elements
        self.assertIn(self.test_signal.symbol, prompt)
        self.assertIn(self.test_signal.pattern, prompt)
        self.assertIn("Winning examples", prompt)
        self.assertIn("Losing examples", prompt)
        
    def test_confluence_analysis_prompt(self):
        """Test confluence analysis prompt generation"""
        prompt = ApolloPrompts.confluence_analysis_prompt(
            signal=self.test_signal,
            market_data=self.market_data
        )
        
        # Check that the prompt includes key elements
        self.assertIn(self.test_signal.symbol, prompt)
        self.assertIn(self.market_data["regime"], prompt)
        self.assertIn("supporting factors", prompt.lower())
        self.assertIn("risk factors", prompt.lower())
        self.assertIn("JSON", prompt)

if __name__ == "__main__":
    unittest.main()