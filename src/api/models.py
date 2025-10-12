"""
WebSocket API models for market data and trading
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class WSMessageType(str, Enum):
    """WebSocket message types for client-server communication"""
    # System messages
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    INFO = "info"
    
    # Authentication
    AUTH = "auth"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    
    # Market data
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    MARKET_DATA = "market_data"
    TICK = "tick"
    OHLC = "ohlc"
    
    # Trading operations
    BALANCE = "balance"
    SYMBOLS = "symbols"
    CONTRACTS = "contracts"
    PROPOSAL = "proposal"
    BUY = "buy"
    SELL = "sell"
    PORTFOLIO = "portfolio"
    PROFIT_TABLE = "profit_table"
    STATEMENT = "statement"


class BaseWSMessage(BaseModel):
    """Base WebSocket message structure"""
    type: WSMessageType = Field(..., description="Message type")
    req_id: Optional[str] = Field(None, description="Request ID for matching requests with responses")


class ErrorWSMessage(BaseWSMessage):
    """Error message"""
    type: WSMessageType = WSMessageType.ERROR
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional error data")


class PingMessage(BaseWSMessage):
    """Ping message for connection heartbeat"""
    type: WSMessageType = WSMessageType.PING
    timestamp: int = Field(..., description="Current timestamp in milliseconds")


class PongMessage(BaseWSMessage):
    """Pong response to ping message"""
    type: WSMessageType = WSMessageType.PONG
    timestamp: int = Field(..., description="Echo of the ping timestamp")


class AuthRequestMessage(BaseWSMessage):
    """Authentication request"""
    type: WSMessageType = WSMessageType.AUTH
    token: str = Field(..., description="Deriv API token")


class AuthResponseMessage(BaseWSMessage):
    """Authentication response"""
    type: WSMessageType = Union[WSMessageType.AUTH_SUCCESS, WSMessageType.AUTH_FAILURE]
    message: str = Field(..., description="Authentication status message")
    data: Optional[Dict[str, Any]] = Field(None, description="User account information if successful")


class SubscribeMessage(BaseWSMessage):
    """Subscribe to market data stream"""
    type: WSMessageType = WSMessageType.SUBSCRIBE
    symbols: List[str] = Field(..., description="List of symbols to subscribe to")
    stream_type: str = Field("tick", description="Stream type: tick, ohlc, candles")
    interval: Optional[str] = Field(None, description="Interval for OHLC data: 1m, 5m, 15m, 1h, 4h, 1d")


class UnsubscribeMessage(BaseWSMessage):
    """Unsubscribe from market data stream"""
    type: WSMessageType = WSMessageType.UNSUBSCRIBE
    subscription_id: Optional[str] = Field(None, description="Specific subscription ID to unsubscribe")
    symbols: Optional[List[str]] = Field(None, description="List of symbols to unsubscribe")
    stream_type: Optional[str] = Field(None, description="Stream type to unsubscribe")


class MarketDataMessage(BaseWSMessage):
    """Market data message"""
    type: WSMessageType = WSMessageType.MARKET_DATA
    symbol: str = Field(..., description="Trading symbol")
    stream_type: str = Field(..., description="Type of data: tick, ohlc, candles")
    subscription_id: str = Field(..., description="Subscription ID")
    data: Dict[str, Any] = Field(..., description="Market data payload")


class TickData(BaseModel):
    """Tick data structure"""
    symbol: str
    price: float
    timestamp: int
    pip_size: float


class OHLCData(BaseModel):
    """OHLC data structure"""
    symbol: str
    interval: str
    open: float
    high: float
    low: float
    close: float
    timestamp: int


class TickMessage(BaseWSMessage):
    """Tick data message"""
    type: WSMessageType = WSMessageType.TICK
    data: TickData


class OHLCMessage(BaseWSMessage):
    """OHLC data message"""
    type: WSMessageType = WSMessageType.OHLC
    data: OHLCData


class BalanceRequestMessage(BaseWSMessage):
    """Balance request"""
    type: WSMessageType = WSMessageType.BALANCE


class BalanceResponseMessage(BaseWSMessage):
    """Balance response"""
    type: WSMessageType = WSMessageType.BALANCE
    balance: float
    currency: str
    loginid: str
    is_virtual: bool


class SymbolsRequestMessage(BaseWSMessage):
    """Active symbols request"""
    type: WSMessageType = WSMessageType.SYMBOLS


class SymbolsResponseMessage(BaseWSMessage):
    """Active symbols response"""
    type: WSMessageType = WSMessageType.SYMBOLS
    symbols: List[Dict[str, Any]]


class ContractsRequestMessage(BaseWSMessage):
    """Contracts for symbol request"""
    type: WSMessageType = WSMessageType.CONTRACTS
    symbol: str


class ContractsResponseMessage(BaseWSMessage):
    """Contracts for symbol response"""
    type: WSMessageType = WSMessageType.CONTRACTS
    symbol: str
    contracts: List[Dict[str, Any]]


class ProposalRequestMessage(BaseWSMessage):
    """Proposal request"""
    type: WSMessageType = WSMessageType.PROPOSAL
    symbol: str
    contract_type: str
    amount: float
    duration: int
    duration_unit: str = "m"
    barrier: Optional[float] = None


class ProposalResponseMessage(BaseWSMessage):
    """Proposal response"""
    type: WSMessageType = WSMessageType.PROPOSAL
    symbol: str
    proposal_id: str
    ask_price: float
    payout: float
    spot: float
    display_value: str
    date_start: int
    date_expiry: int


class BuyRequestMessage(BaseWSMessage):
    """Buy contract request"""
    type: WSMessageType = WSMessageType.BUY
    proposal_id: str
    price: float


class BuyResponseMessage(BaseWSMessage):
    """Buy contract response"""
    type: WSMessageType = WSMessageType.BUY
    contract_id: str
    transaction_id: str
    balance_after: float
    buy_price: float
    purchase_time: int
    longcode: str


class SellRequestMessage(BaseWSMessage):
    """Sell contract request"""
    type: WSMessageType = WSMessageType.SELL
    contract_id: str


class SellResponseMessage(BaseWSMessage):
    """Sell contract response"""
    type: WSMessageType = WSMessageType.SELL
    contract_id: str
    transaction_id: str
    sold_for: float
    remaining_contracts: int


class PortfolioRequestMessage(BaseWSMessage):
    """Portfolio request"""
    type: WSMessageType = WSMessageType.PORTFOLIO


class PortfolioResponseMessage(BaseWSMessage):
    """Portfolio response"""
    type: WSMessageType = WSMessageType.PORTFOLIO
    contracts: List[Dict[str, Any]]


class ProfitTableRequestMessage(BaseWSMessage):
    """Profit table request"""
    type: WSMessageType = WSMessageType.PROFIT_TABLE
    limit: Optional[int] = 50
    offset: Optional[int] = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ProfitTableResponseMessage(BaseWSMessage):
    """Profit table response"""
    type: WSMessageType = WSMessageType.PROFIT_TABLE
    transactions: List[Dict[str, Any]]
    count: int


class StatementRequestMessage(BaseWSMessage):
    """Statement request"""
    type: WSMessageType = WSMessageType.STATEMENT
    limit: Optional[int] = 50
    offset: Optional[int] = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class StatementResponseMessage(BaseWSMessage):
    """Statement response"""
    type: WSMessageType = WSMessageType.STATEMENT
    transactions: List[Dict[str, Any]]
    count: int