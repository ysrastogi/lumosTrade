"""
Tests for the Athena prompts module.

This module contains tests to verify that the Athena prompts module is working correctly
and generating appropriate prompts for different scenarios.
"""

import unittest
import json
from src.agents.athena_workspace.prompts import AthenaPrompts

class TestAthenaPrompts(unittest.TestCase):
    """Test cases for Athena prompts module."""
    
    def setUp(self):
        """Set up test data"""
        self.test_symbol = "BTC/USD"
        
        # Sample features for testing
        self.test_features = {
            "close_price": 50000,
            "rsi": 65,
            "volatility": 0.02,
            "macd": 100,
            "bb_width": 0.05,
            "price_change_pct": 0.03,
            "atr": 500,
            "trend_direction": "bullish"
        }
        
        # Sample patterns for testing
        self.test_patterns = [
            {
                "type": "bullish_engulfing",
                "description": "Bullish engulfing pattern",
                "bias": "bullish",
                "confidence": 80
            },
            {
                "type": "rsi_divergence",
                "description": "Bullish RSI divergence",
                "bias": "bullish",
                "confidence": 75
            }
        ]
        
        # Sample price history for testing
        self.test_price_history = [
            {"timestamp": "2025-10-10T12:00:00Z", "price": 49000},
            {"timestamp": "2025-10-10T13:00:00Z", "price": 49500},
            {"timestamp": "2025-10-10T14:00:00Z", "price": 50000}
        ]
        
    def test_market_summary_prompt(self):
        """Test market summary prompt generation"""
        prompt = AthenaPrompts.market_summary_prompt(
            symbol=self.test_symbol,
            features=self.test_features,
            regime="bullish",
            patterns=self.test_patterns
        )
        
        # Check that the prompt includes key elements
        self.assertIn(self.test_symbol, prompt)
        self.assertIn(str(self.test_features["close_price"]), prompt)
        self.assertIn(self.test_patterns[0]["type"], prompt)
        
    def test_trade_idea_prompt(self):
        """Test trade idea prompt generation"""
        prompt = AthenaPrompts.trade_idea_prompt(
            symbol=self.test_symbol,
            features=self.test_features,
            patterns=self.test_patterns
        )
        
        # Check that the prompt includes key elements
        self.assertIn(self.test_symbol, prompt)
        self.assertIn(str(self.test_features["close_price"]), prompt)
        self.assertIn("JSON", prompt)
        
    def test_pattern_analysis_prompt(self):
        """Test pattern analysis prompt generation"""
        prompt = AthenaPrompts.pattern_analysis_prompt(
            symbol=self.test_symbol,
            pattern_data=self.test_patterns[0],
            price_history=self.test_price_history
        )
        
        # Check that the prompt includes key elements
        self.assertIn(self.test_symbol, prompt)
        self.assertIn(self.test_patterns[0]["type"], prompt)
        
    def test_market_context_prompt(self):
        """Test market context prompt generation"""
        # Create recent observations for testing
        recent_obs = [
            {
                "timestamp": "2025-10-10T12:00:00Z",
                "regime": "bullish",
                "trading_bias": "bullish",
                "patterns": self.test_patterns
            }
        ]
        
        prompt = AthenaPrompts.market_context_prompt(
            symbol=self.test_symbol,
            recent_observations=recent_obs,
            current_features=self.test_features
        )
        
        # Check that the prompt includes key elements
        self.assertIn(self.test_symbol, prompt)
        self.assertIn("regime has evolved", prompt.lower())

if __name__ == "__main__":
    unittest.main()