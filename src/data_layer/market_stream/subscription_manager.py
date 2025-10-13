"""
Subscription manager for market data streams
"""

import logging
from typing import Dict, List, Callable, Optional, Any

from src.data_layer.market_stream.models import RequestID, INTERVAL_MAP

logger = logging.getLogger(__name__)


class SubscriptionManager:
    """Manages market data subscriptions and callbacks"""
    
    def __init__(self, send_message_func: Callable[[Dict], None], get_request_id_func: Callable[[], RequestID]):
        """Initialize the subscription manager
        
        Args:
            send_message_func: Function to send messages to the WebSocket
            get_request_id_func: Function to get the next request ID
        """
        self.logger = logger.getChild("SubscriptionManager")
        self.send_message = send_message_func
        self.get_next_request_id = get_request_id_func
        
        # Dictionaries to store subscriptions and callbacks
        self.subscriptions: Dict[str, Dict] = {}
        self.callbacks: Dict[RequestID, Callable] = {}
    
    def subscribe_ticks(self, symbol: str, callback: Optional[Callable] = None) -> bool:
        """Subscribe to tick data for a symbol
        
        Args:
            symbol: Trading symbol (e.g. "R_100")
            callback: Optional callback function for tick data
            
        Returns:
            bool: Success status
        """
        req_id = self.get_next_request_id()
        request = {
            "ticks": symbol,
            "subscribe": 1,
            "req_id": req_id
        }
        
        if callback:
            self.callbacks[req_id] = callback
        
        self.subscriptions[f"tick_{symbol}"] = request
        self.send_message(request)
        self.logger.info(f"Subscribed to tick data for {symbol}")
        return True
    
    def unsubscribe_ticks(self, symbol: str) -> bool:
        """Unsubscribe from tick data for a symbol
        
        Args:
            symbol: Trading symbol (e.g. "R_100")
            
        Returns:
            bool: Success status
        """
        subscription_key = f"tick_{symbol}"
        if subscription_key not in self.subscriptions:
            self.logger.warning(f"Not subscribed to tick data for {symbol}")
            return False
        
        request = self.subscriptions[subscription_key]
        unsub_request = request.copy()
        unsub_request["subscribe"] = 0
        
        self.send_message(unsub_request)
        del self.subscriptions[subscription_key]
        
        self.logger.info(f"Unsubscribed from tick data for {symbol}")
        return True
    
    def subscribe_candles(self, symbol: str, interval: str = "1m", callback: Optional[Callable] = None) -> bool:
        """Subscribe to candle data for a symbol
        
        Args:
            symbol: Trading symbol (e.g. "R_100")
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            callback: Optional callback function for candle data
            
        Returns:
            bool: Success status
        """
        granularity = INTERVAL_MAP.get(interval, 60)
        req_id = self.get_next_request_id()
        
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": 1000,
            "end": "latest",
            "granularity": granularity,
            "subscribe": 1,
            "req_id": req_id
        }
        
        if callback:
            self.callbacks[req_id] = callback
        
        self.subscriptions[f"candle_{symbol}_{interval}"] = request
        self.send_message(request)
        self.logger.info(f"Subscribed to {interval} candle data for {symbol}")
        return True
    
    def subscribe_ohlc(self, symbol: str, interval: str = "1m", callback: Optional[Callable] = None) -> bool:
        """Subscribe to OHLC data for a symbol
        
        Args:
            symbol: Trading symbol (e.g. "R_100")
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            callback: Optional callback function
            
        Returns:
            bool: Success status
        """
        granularity = INTERVAL_MAP.get(interval, 60)
        req_id = self.get_next_request_id()
        
        request = {
            "ohlc": symbol,
            "granularity": granularity,
            "subscribe": 1,
            "req_id": req_id
        }
        
        if callback:
            self.callbacks[req_id] = callback
        
        self.subscriptions[f"ohlc_{symbol}_{interval}"] = request
        self.send_message(request)
        self.logger.info(f"Subscribed to {interval} OHLC data for {symbol}")
        return True
    
    def unsubscribe_ohlc(self, symbol: str, interval: str = "1m") -> bool:
        """Unsubscribe from OHLC data for a symbol
        
        Args:
            symbol: Trading symbol (e.g. "R_100")
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            bool: Success status
        """
        subscription_key = f"ohlc_{symbol}_{interval}"
        if subscription_key not in self.subscriptions:
            self.logger.warning(f"Not subscribed to OHLC data for {symbol} with interval {interval}")
            return False
            
        request = self.subscriptions[subscription_key]
        unsub_request = request.copy()
        unsub_request["subscribe"] = 0
        
        self.send_message(unsub_request)
        del self.subscriptions[subscription_key]
        
        self.logger.info(f"Unsubscribed from {interval} OHLC data for {symbol}")
        return True
    
    def get_callback(self, req_id: RequestID) -> Optional[Callable]:
        """Get a callback for a request ID
        
        Args:
            req_id: Request ID
            
        Returns:
            Optional[Callable]: Callback function if found, None otherwise
        """
        return self.callbacks.get(req_id)
    
    def remove_callback(self, req_id: RequestID) -> bool:
        """Remove a callback for a request ID
        
        Args:
            req_id: Request ID
            
        Returns:
            bool: True if callback was removed, False otherwise
        """
        if req_id in self.callbacks:
            del self.callbacks[req_id]
            return True
        return False
    
    def register_callback(self, req_id: RequestID, callback: Callable) -> None:
        """Register a callback for a request ID
        
        Args:
            req_id: Request ID
            callback: Callback function
        """
        self.callbacks[req_id] = callback
    
    def resubscribe_all(self) -> None:
        """Re-establish all subscriptions"""
        for key, request in self.subscriptions.items():
            self.send_message(request)
            self.logger.info(f"Re-subscribed: {key}")
    
    def get_active_subscriptions(self) -> List[str]:
        """Get list of active subscriptions
        
        Returns:
            List[str]: List of subscription keys
        """
        return list(self.subscriptions.keys())