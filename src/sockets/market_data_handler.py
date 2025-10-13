"""
Market data handler for WebSocket server
"""
import asyncio
import logging
from typing import Dict, Any, Optional

from .models import WSMessageType

logger = logging.getLogger(__name__)

class MarketDataHandler:
    """
    Handles market data processing and distribution
    """
    def __init__(self, market_stream, connection_manager):
        """
        Initialize the market data handler
        
        Args:
            market_stream: The market stream to get data from
            connection_manager: The connection manager for client connections
        """
        self.market_stream = market_stream
        self.connection_manager = connection_manager
    
    def setup_callbacks(self):
        """
        Set up callbacks for market data
        """
        self.market_stream.add_callback("tick", self._on_tick_data)
        self.market_stream.add_callback("ohlc", self._on_ohlc_data)
    
    def subscribe_ticks(self, symbol: str):
        """
        Subscribe to tick data for a symbol
        
        Args:
            symbol: The symbol to subscribe to
        """
        logger.info(f"Subscribing to tick data for {symbol}")
        self.market_stream.subscribe_ticks(symbol)
    
    def unsubscribe_ticks(self, symbol: str):
        """
        Unsubscribe from tick data for a symbol
        
        Args:
            symbol: The symbol to unsubscribe from
        """
        logger.info(f"Unsubscribing from tick data for {symbol}")
        self.market_stream.forget_ticks(symbol)
    
    def subscribe_ohlc(self, symbol: str, interval: str):
        """
        Subscribe to OHLC data for a symbol
        
        Args:
            symbol: The symbol to subscribe to
            interval: The candle interval
        """
        logger.info(f"Subscribing to OHLC data for {symbol} with interval {interval}")
        self.market_stream.subscribe_ohlc(symbol, interval)
    
    def unsubscribe_ohlc(self, symbol: str, interval: str):
        """
        Unsubscribe from OHLC data for a symbol
        
        Args:
            symbol: The symbol to unsubscribe from
            interval: The candle interval
        """
        logger.info(f"Unsubscribing from OHLC data for {symbol} with interval {interval}")
        self.market_stream.forget_ohlc(symbol, interval)
    
    async def _on_tick_data(self, tick_data: Dict):
        """
        Handle tick data from market stream
        
        Args:
            tick_data: The tick data
        """
        try:
            symbol = tick_data.get("tick", {}).get("symbol")
            if not symbol:
                return
            
            subscription_id = f"{symbol}_tick"
            
            client_ids = self.connection_manager.get_subscription_clients(subscription_id)
            if not client_ids:
                return
                
            # Create tick message
            tick_msg = {
                "type": WSMessageType.TICK,
                "data": {
                    "symbol": symbol,
                    "price": tick_data.get("tick", {}).get("quote"),
                    "timestamp": tick_data.get("tick", {}).get("epoch"),
                    "pip_size": tick_data.get("tick", {}).get("pip_size", 0.01)
                }
            }
            
            # Send to all subscribed clients
            for client_id in client_ids:
                await self.connection_manager.send_message(client_id, tick_msg)
        
        except Exception as e:
            logger.error(f"Error handling tick data: {e}")
    
    async def _on_ohlc_data(self, ohlc_data: Dict):
        """
        Handle OHLC data from market stream
        
        Args:
            ohlc_data: The OHLC data
        """
        try:
            symbol = ohlc_data.get("ohlc", {}).get("symbol")
            interval = ohlc_data.get("ohlc", {}).get("granularity")
            
            if not symbol or not interval:
                return
            
            subscription_id = f"{symbol}_ohlc_{interval}"
            
            client_ids = self.connection_manager.get_subscription_clients(subscription_id)
            if not client_ids:
                return
                
            # Create OHLC message
            ohlc_msg = {
                "type": WSMessageType.OHLC,
                "data": {
                    "symbol": symbol,
                    "interval": interval,
                    "open": ohlc_data.get("ohlc", {}).get("open"),
                    "high": ohlc_data.get("ohlc", {}).get("high"),
                    "low": ohlc_data.get("ohlc", {}).get("low"),
                    "close": ohlc_data.get("ohlc", {}).get("close"),
                    "timestamp": ohlc_data.get("ohlc", {}).get("open_time")
                }
            }
            
            # Send to all subscribed clients
            for client_id in client_ids:
                await self.connection_manager.send_message(client_id, ohlc_msg)
        
        except Exception as e:
            logger.error(f"Error handling OHLC data: {e}")
    
    def handle_portfolio_update(self, portfolio_data: Dict):
        """
        Handle portfolio updates from trading client
        
        Args:
            portfolio_data: The portfolio data
        """
        try:
            # Create task to broadcast portfolio update to authenticated clients
            asyncio.create_task(self.connection_manager.broadcast_to_authenticated({
                "type": WSMessageType.PORTFOLIO,
                "contracts": portfolio_data.get("portfolio", {}).get("contracts", [])
            }))
        except Exception as e:
            logger.error(f"Error handling portfolio update: {e}")
    
    def connect(self):
        """
        Connect to the market stream if not already connected
        """
        if not self.market_stream.is_connected:
            logger.info("Connecting to market stream")
            self.market_stream.connect()
    
    def disconnect(self):
        """
        Disconnect from the market stream if connected
        """
        if self.market_stream.is_connected:
            logger.info("Disconnecting from market stream")
            self.market_stream.disconnect()