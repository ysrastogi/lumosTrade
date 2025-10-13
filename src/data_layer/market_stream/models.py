"""
Data models and type definitions for the market stream components
"""

from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MarketConfig:
    """Configuration for market data streams"""
    websocket_url: str
    app_id: str
    reconnect_attempts: int
    reconnect_delay: int
    heartbeat_interval: int
    symbols: List[str]
    stream_types: List[str]
    candle_intervals: List[str]


@dataclass
class TickData:
    """Data structure for tick data"""
    symbol: str
    quote: float
    epoch: int
    timestamp: datetime
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TickData':
        """Create a TickData object from a dict"""
        tick = data.get('tick', {})
        timestamp = datetime.fromtimestamp(tick.get('epoch', 0))
        return cls(
            symbol=tick.get('symbol', ''),
            quote=tick.get('quote', 0.0),
            epoch=tick.get('epoch', 0),
            timestamp=timestamp
        )


@dataclass
class CandleData:
    """Data structure for candle data"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    epoch: int
    timestamp: datetime
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], symbol: str = '') -> List['CandleData']:
        """Create CandleData objects from a dict"""
        candles = []
        for candle in data.get('candles', []):
            timestamp = datetime.fromtimestamp(candle.get('epoch', 0))
            candles.append(cls(
                symbol=symbol,
                open=candle.get('open', 0.0),
                high=candle.get('high', 0.0),
                low=candle.get('low', 0.0),
                close=candle.get('close', 0.0),
                epoch=candle.get('epoch', 0),
                timestamp=timestamp
            ))
        return candles


@dataclass
class OHLCData:
    """Data structure for OHLC data"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    epoch: int
    timestamp: datetime
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OHLCData':
        """Create an OHLCData object from a dict"""
        ohlc = data.get('ohlc', {})
        timestamp = datetime.fromtimestamp(ohlc.get('epoch', 0))
        return cls(
            symbol=ohlc.get('symbol', ''),
            open=ohlc.get('open', 0.0),
            high=ohlc.get('high', 0.0),
            low=ohlc.get('low', 0.0),
            close=ohlc.get('close', 0.0),
            epoch=ohlc.get('epoch', 0),
            timestamp=timestamp
        )


@dataclass
class ContractData:
    """Data structure for contract data"""
    contract_id: str
    current_spot: float
    profit: float
    is_sold: bool
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContractData':
        """Create a ContractData object from a dict"""
        contract = data.get('proposal_open_contract', {})
        return cls(
            contract_id=str(contract.get('contract_id', '')),
            current_spot=contract.get('current_spot', 0.0),
            profit=contract.get('profit', 0.0),
            is_sold=contract.get('is_sold', False)
        )


# Type definitions for improved type hinting
MessageHandlerFunc = Callable[[Dict[str, Any]], None]
SubscriptionCallback = Callable[[Dict[str, Any]], None]
RequestID = int
"""A request ID for API calls"""

# Interval mapping for time-based data
INTERVAL_MAP = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400
}

GRANULARITY_MAP = {v: k for k, v in INTERVAL_MAP.items()}