"""
Regime detection module for market analysis.

This module identifies the current market regime (trend, volatility state)
based on technical indicators and price patterns.
"""

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Detects market regime based on technical features.
    
    The RegimeDetector analyzes technical indicators to classify the
    current state of the market into different regimes like trending,
    ranging, volatile, etc. This helps in selecting appropriate
    trading strategies for the current conditions.
    """

    @staticmethod
    def detect_regime(features: Dict) -> str:
        """
        Classify market regime based on volatility and trend indicators.
        
        Args:
            features: Dictionary of technical indicators from FeatureExtractor
            
        Returns:
            String representing the detected market regime
        """
        if 'error' in features:
            logger.warning(f"Unable to detect regime: {features['error']}")
            return "unknown"

        # Extract key metrics
        volatility = features.get('volatility', 0)
        rsi = features.get('rsi', 50)
        bb_width = features.get('bb_width', 0)
        price_change = features.get('price_change_pct', 0)
        adx = features.get('adx', 20)  # ADX measures trend strength

        # Volatility classification
        if volatility < 0.01:
            vol_regime = "low_volatility"
        elif volatility < 0.025:
            vol_regime = "moderate_volatility"
        else:
            vol_regime = "high_volatility"

        # Trend classification
        if adx > 25:  # Strong trend according to ADX
            if rsi > 70 and price_change > 0:
                trend_regime = "strong_uptrend"
            elif rsi > 55 and price_change > 0:
                trend_regime = "uptrend"
            elif rsi < 30 and price_change < 0:
                trend_regime = "strong_downtrend"
            elif rsi < 45 and price_change < 0:
                trend_regime = "downtrend"
            else:
                trend_regime = "mixed_trend"
        else:  # Weak trend, likely ranging
            if bb_width < 0.02:  # Tight range
                trend_regime = "tight_range"
            else:
                trend_regime = "ranging"

        # Pattern detection based on Bollinger Band relationships
        if bb_width < 0.015:
            pattern = "consolidation_squeeze"
        elif abs(price_change) < 0.5 and volatility < 0.015:
            pattern = "tight_range"
        elif bb_width > 0.04 and abs(price_change) > 1.0:
            pattern = "expansion_move"
        else:
            pattern = "normal"

        # Combine into regime description
        regime = f"{vol_regime}_{trend_regime}"
        if pattern != "normal":
            regime += f"_{pattern}"

        return regime

    @staticmethod
    def get_regime_description(regime: str) -> str:
        """
        Convert regime code to natural language description.
        
        Args:
            regime: The regime code string from detect_regime()
            
        Returns:
            Human-readable description of the market regime
        """
        descriptions = {
            # Volatility descriptors
            "low_volatility": "calm",
            "moderate_volatility": "active",
            "high_volatility": "volatile",
            
            # Trend descriptors
            "strong_uptrend": "strong bullish momentum",
            "uptrend": "bullish trend",
            "downtrend": "bearish trend",
            "strong_downtrend": "strong bearish momentum",
            "mixed_trend": "mixed directional signals",
            "ranging": "sideways movement",
            "tight_range": "in narrow range",
            
            # Pattern descriptors
            "consolidation_squeeze": "with tight consolidation (potential breakout setup)",
            "expansion_move": "with expanding volatility (trend continuation likely)",
            "normal": "",
            
            # Unknown
            "unknown": "with insufficient data for classification"
        }

        parts = regime.split('_')
        desc_parts = [descriptions.get(part, part) for part in parts if part in descriptions]

        return " ".join(desc_parts)
        
    @staticmethod
    def calculate_confidence(regime: str, features: Dict) -> float:
        """
        Calculate confidence score for the regime classification.
        
        Args:
            regime: Detected regime string
            features: Dictionary of technical features
            
        Returns:
            Float between 0-1 representing confidence in the classification
        """
        if 'error' in features or regime == "unknown":
            return 0.0
            
        # Base confidence
        confidence = 0.6
        
        # Adjust based on ADX (trend strength indicator)
        if 'adx' in features:
            adx = features['adx']
            if adx > 30:  # Strong trend
                confidence += 0.15
            elif adx < 15:  # Very weak trend
                confidence -= 0.1
                
        # Adjust based on clarity of patterns
        if "strong_uptrend" in regime or "strong_downtrend" in regime:
            confidence += 0.15
        elif "mixed" in regime:
            confidence -= 0.1
            
        # Adjust based on data quality/quantity
        if features.get('sma_200') is not None:  # We have enough historical data
            confidence += 0.05
            
        return min(max(confidence, 0.0), 1.0)