"""
Data models for the Athena market intelligence agent.
These models define the structure for market analysis and insights.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field


class PatternType(str, Enum):
    """Common market pattern types"""
    OVERSOLD_REBOUND = "oversold_rebound"
    OVERBOUGHT_REVERSAL = "overbought_reversal"
    VOLATILITY_SQUEEZE = "volatility_squeeze"
    BB_LOWER_BOUNCE = "bb_lower_bounce"
    BB_UPPER_REJECTION = "bb_upper_rejection"
    MACD_BULLISH_MOMENTUM = "macd_bullish_momentum"
    MACD_BEARISH_MOMENTUM = "macd_bearish_momentum"
    MEAN_REVERSION = "mean_reversion"
    BEARISH_DIVERGENCE = "bearish_divergence"
    BULLISH_DIVERGENCE = "bullish_divergence"


class TradeBias(str, Enum):
    """Trading bias directions"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Pattern(BaseModel):
    """Detected market pattern"""
    type: str = Field(..., description="Type of pattern detected")
    confidence: float = Field(..., description="Confidence score (0-100)")
    description: str = Field(..., description="Human-readable pattern description")
    bias: TradeBias = Field(..., description="Directional bias of the pattern")
    key_metrics: Optional[Dict[str, Any]] = Field(None, description="Key metrics supporting the pattern")


class TradeBiasAssessment(BaseModel):
    """Overall trading bias assessment"""
    bias: TradeBias = Field(..., description="Overall directional bias")
    confidence: float = Field(..., description="Confidence level (0-1)")
    description: str = Field(..., description="Explanation of the bias")


class TradeIdea(BaseModel):
    """Concrete trading opportunity"""
    symbol: str = Field(..., description="Trading symbol")
    pattern: str = Field(..., description="Pattern type triggering the idea")
    direction: str = Field(..., description="Trade direction (buy, sell)")
    confidence: float = Field(..., description="Confidence level (0-100)")
    description: str = Field(..., description="Trade idea description")
    time_horizon: str = Field(..., description="Trading timeframe (intraday, swing, position)")
    entry: Union[float, str] = Field(..., description="Entry price or condition")
    stop_loss: Union[float, str] = Field(..., description="Stop loss level or condition")
    target: Union[float, str] = Field(..., description="Price target or condition")
    risk_reward: float = Field(..., description="Risk/reward ratio")
    timestamp: str = Field(..., description="Generation timestamp")


class MarketContext(BaseModel):
    """Complete market context from Athena analysis"""
    timestamp: str = Field(..., description="Analysis timestamp")
    symbol: str = Field(..., description="Symbol analyzed")
    interval: int = Field(..., description="Timeframe in seconds")
    features: Dict[str, Any] = Field(..., description="Technical features and indicators")
    regime: str = Field(..., description="Market regime classification")
    regime_confidence: float = Field(..., description="Confidence in regime assessment")
    patterns: List[Pattern] = Field(..., description="Detected patterns")
    trading_bias: TradeBiasAssessment = Field(..., description="Overall trading bias")
    summary: str = Field(..., description="Market summary text")
    trade_ideas: List[TradeIdea] = Field(..., description="Potential trade ideas")
    confidence: float = Field(..., description="Overall context confidence score")


class MarketInsightsResponse(BaseModel):
    """Response model for market insights API"""
    status: str = Field(..., description="API response status")
    timestamp: str = Field(..., description="Response timestamp") 
    top_opportunities: List[MarketContext] = Field(..., description="Top market opportunities")
    symbols_analyzed: int = Field(..., description="Number of symbols analyzed")


class SymbolRequest(BaseModel):
    """Request to analyze a specific symbol"""
    symbol: str = Field(..., description="Symbol to analyze")
    interval: Optional[int] = Field(60, description="Timeframe in seconds")
    use_llm: Optional[bool] = Field(True, description="Whether to use LLM for insights")


class MultiSymbolRequest(BaseModel):
    """Request to analyze multiple symbols"""
    symbols: List[str] = Field(..., description="List of symbols to analyze")
    interval: Optional[int] = Field(60, description="Timeframe in seconds")
    use_llm: Optional[bool] = Field(True, description="Whether to use LLM for insights")