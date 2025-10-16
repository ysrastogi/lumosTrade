"""
Data models for Market Stream Aggregator.
These models define the structure for normalized and aggregated market data.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field



class DirectionalBias(str, Enum):
    """Market directional bias classification"""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


class NormalizedSymbol(BaseModel):
    """Normalized format for a trading symbol"""
    base: str = Field(..., description="Base currency or asset (e.g., 'BTC')")
    quote: str = Field(..., description="Quote currency (e.g., 'USD')")
    original: str = Field(..., description="Original symbol from data source (e.g., 'BTCUSD', 'BTC-USD')")
    display: str = Field(..., description="Normalized display format (e.g., 'BTC/USD')")
    asset_name: str = Field(..., description="Full asset name (e.g., 'Bitcoin')")


class RawMarketTick(BaseModel):
    """Raw market tick data from stream"""
    symbol: str = Field(..., description="Original trading symbol")
    price: float = Field(..., description="Current price")
    timestamp: int = Field(..., description="Unix timestamp")
    ask: Optional[float] = Field(None, description="Ask price if available")
    bid: Optional[float] = Field(None, description="Bid price if available")
    volume: Optional[float] = Field(None, description="Trading volume if available")
    pip_size: Optional[float] = Field(None, description="Minimum price movement size")


class NormalizedMarketTick(BaseModel):
    """Normalized market tick data"""
    symbol: NormalizedSymbol = Field(..., description="Normalized symbol information")
    price: float = Field(..., description="Current price")
    timestamp: int = Field(..., description="Unix timestamp")
    timestamp_dt: datetime = Field(..., description="Python datetime object")
    ask: Optional[float] = Field(None, description="Ask price if available")
    bid: Optional[float] = Field(None, description="Bid price if available")
    volume: Optional[float] = Field(None, description="Trading volume if available")
    pip_size: Optional[float] = Field(None, description="Minimum price movement size")


class SymbolMetrics(BaseModel):
    """Metrics for a single market symbol at a specific timeframe"""
    symbol: NormalizedSymbol = Field(..., description="Normalized symbol information")
    last_price: float = Field(..., description="Most recent price")
    last_updated: datetime = Field(..., description="Last update time")
    
    # Price changes
    price_change_1m: float = Field(0.0, description="1-minute price change percentage")
    price_change_5m: float = Field(0.0, description="5-minute price change percentage")
    price_change_15m: float = Field(0.0, description="15-minute price change percentage")
    price_change_1h: float = Field(0.0, description="1-hour price change percentage")
    
    # Volume metrics
    volume_1m: float = Field(0.0, description="1-minute trading volume")
    volume_5m: float = Field(0.0, description="5-minute trading volume")
    volume_15m: float = Field(0.0, description="15-minute trading volume")
    
    # Calculated metrics
    volatility: float = Field(0.0, description="Recent volatility measurement")
    directional_bias: DirectionalBias = Field(DirectionalBias.NEUTRAL, description="Current market bias")
    sentiment_score: float = Field(0.0, description="Computed market sentiment (-1.0 to 1.0)")
    
    # For UI tracking
    status: str = Field("neutral", description="UI status: 'up', 'down', or 'neutral'")


class MarketSnapshot(BaseModel):
    """Aggregated snapshot of market data at a specific time"""
    timestamp: datetime = Field(..., description="Snapshot timestamp")
    symbols: Dict[str, SymbolMetrics] = Field(..., description="Market data by symbol")
    top_gainers: List[str] = Field([], description="Top gaining symbols")
    top_losers: List[str] = Field([], description="Top losing symbols")
    top_volume: List[str] = Field([], description="Top symbols by volume")
    high_volatility: List[str] = Field([], description="Symbols with high volatility")


class AICommentaryData(BaseModel):
    """Generated AI commentary for the dashboard"""
    macro: str = Field(..., description="Macro market commentary")
    flows: str = Field(..., description="Market flow commentary")
    risk: str = Field(..., description="Risk metrics commentary")
    signal: str = Field(..., description="Technical signal commentary")
    generated_at: datetime = Field(..., description="When the commentary was generated")


class SetupType(str, Enum):
    """Type of trading setup"""
    VOL_COMPRESSION = "Volatility Compression Breakout"
    MEAN_REVERSION = "Mean Reversion + RSI Divergence"
    BREAKOUT = "Breakout"
    BREAKDOWN = "Breakdown Continuation" 
    MOMENTUM = "Momentum"
    RANGE_BOUND = "Range Bound"


class TradingSetup(BaseModel):
    """Detected trading setup with score"""
    pair: str = Field(..., description="Trading pair in normalized format")
    setup: SetupType = Field(..., description="Type of trading setup")
    score: int = Field(..., ge=0, le=100, description="Setup confidence score (0-100)")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional setup details")


# API Response Models

class MarketMapEntry(BaseModel):
    """Entry for the market map API response"""
    symbol: str = Field(..., description="Trading symbol in normalized format (e.g. 'BTC/USD')")
    status: str = Field(..., description="Price direction: 'up', 'down', or 'neutral'")
    volume: int = Field(..., description="Trading volume (normalized value for UI)")
    asset: str = Field(..., description="Asset name (e.g. 'Bitcoin')")


class MarketMapResponse(BaseModel):
    """Response for the /api/market-map endpoint"""
    symbols: List[MarketMapEntry] = Field(..., description="List of market symbols with status")
    last_updated: datetime = Field(..., description="Timestamp of the data")
    debug: Optional[Dict[str, Any]] = Field(None, description="Debug information")


class AICommentaryResponse(BaseModel):
    """Response for the /api/ai-commentary endpoint"""
    macro: str = Field(..., description="Macro market commentary")
    flows: str = Field(..., description="Market flow commentary") 
    risk: str = Field(..., description="Risk metrics commentary")
    signal: str = Field(..., description="Technical signal commentary")
    last_updated: datetime = Field(..., description="Timestamp of the data")


class TradeSetupResponse(BaseModel):
    """Single trade setup in the API response"""
    pair: str = Field(..., description="Trading pair (e.g. 'BTC/USD')")
    setup: str = Field(..., description="Setup type description")
    score: int = Field(..., ge=0, le=100, description="Setup score (0-100)")


class TopSetupsResponse(BaseModel):
    """Response for the /api/top-setups endpoint"""
    setups: List[TradeSetupResponse] = Field(..., description="List of top trading setups")
    last_updated: datetime = Field(..., description="Timestamp of the data")


class ExportFormat(str, Enum):
    """Export format options"""
    CSV = "csv"
    GOOGLE_SHEETS = "sheets"
    EXCEL = "excel"


class ExportRequest(BaseModel):
    """Request for the /api/export endpoint"""
    format: ExportFormat = Field(..., description="Export format")
    include_market_map: bool = Field(True, description="Include market map data")
    include_commentary: bool = Field(True, description="Include AI commentary")
    include_setups: bool = Field(True, description="Include top setups")


class ExportResponse(BaseModel):
    """Response for the /api/export endpoint"""
    export_url: str = Field(..., description="URL to download the exported data")
    expires_at: datetime = Field(..., description="Expiration time for the export URL")