import logging
import yaml
from typing import Dict, List, Optional, Callable, Any
from dotenv import load_dotenv

from config.settings import settings
from src.utils.callback_manager import CallbackManager
from src.data_layer.market_stream.models import MarketConfig
from src.data_layer.market_stream.connection_manager import ConnectionManager
from src.data_layer.market_stream.subscription_manager import SubscriptionManager
from src.data_layer.market_stream.message_handler import MessageHandler

load_dotenv()
logger = logging.getLogger(__name__)


class MarketStream:
    
    def __init__(self, config_path: str = "config/tradding_config.yaml", auth_token: str = None):

        self.logger = logger.getChild("MarketStream")
        self.config = self._load_config(config_path)
        self.auth_token = auth_token or settings.deriv_auth_token
        self.logger.info(f"Using Deriv auth token: {self.auth_token[:5]}...")
        
        self.ws_url = f"{self.config['websocket']['url']}?app_id={self.config['websocket']['app_id']}"
        
        self.callback_manager = CallbackManager()
    
        self.market_config = self._create_market_config()
    
        self._initialize_components()
    
    def _load_config(self, config_path: str) -> Dict:
        
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def _create_market_config(self) -> MarketConfig:
    
        websocket_config = self.config['websocket']
        market_data_config = self.config.get('market_data', {})
        
        # Get symbols from configuration or use defaults
        symbols = market_data_config.get('symbols', [])
        if not symbols:
            self.logger.warning("No symbols configured in config file")
            # Default symbols if none configured
            symbols = ["R_10", "R_50", "R_100", "EURUSD", "GBPUSD", "USDJPY"]
        
        return MarketConfig(
            websocket_url=websocket_config['url'],
            app_id=websocket_config['app_id'],
            reconnect_attempts=websocket_config['reconnect_attempts'],
            reconnect_delay=websocket_config['reconnect_delay'],
            heartbeat_interval=websocket_config['heartbeat_interval'],
            symbols=symbols,
            stream_types=market_data_config.get('stream_types', ['tick']),
            candle_intervals=market_data_config.get('candle_intervals', ['1m'])
        )
    
    def _initialize_components(self) -> None:
       
        self.connection_manager = ConnectionManager(
            ws_url=self.ws_url,
            auth_token=self.auth_token,
            reconnect_attempts=self.market_config.reconnect_attempts,
            reconnect_delay=self.market_config.reconnect_delay,
            heartbeat_interval=self.market_config.heartbeat_interval,
            message_handler=self._handle_message
        )
        
        self.subscription_manager = SubscriptionManager(
            send_message_func=self.connection_manager.send_message,
            get_request_id_func=self.connection_manager.get_next_request_id
        )
        
        self.message_handler = MessageHandler(
            auth_token=self.auth_token,
            callback_manager=self.callback_manager,
            subscription_manager=self.subscription_manager,
            connection_manager=self.connection_manager,
            subscribe_configured_symbols_func=self._subscribe_to_configured_symbols
        )
    
    def _handle_message(self, data: Dict[str, Any]) -> None:
        self.message_handler.handle_message(data)
    
    def _subscribe_to_configured_symbols(self) -> None:
        try:
            self.logger.info("Subscribing to configured symbols")
            
            symbols = self.market_config.symbols
            stream_types = self.market_config.stream_types
            candle_intervals = self.market_config.candle_intervals
            
            self.logger.info(f"Subscribing to {len(symbols)} symbols: {symbols}")
            
            # Subscribe to all symbols with all configured stream types
            for symbol in symbols:
                if 'tick' in stream_types:
                    self.subscribe_ticks(symbol)
                    
                if 'ohlc' in stream_types:
                    for interval in candle_intervals:
                        self.subscribe_ohlc(symbol, interval)
                        
                if 'candles' in stream_types:
                    for interval in candle_intervals:
                        self.subscribe_candles(symbol, interval)
            
            self.logger.info(f"Successfully subscribed to {len(symbols)} symbols with {len(stream_types)} stream types")
        except Exception as e:
            self.logger.error(f"Error subscribing to symbols: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    
    def connect(self) -> bool:
        return self.connection_manager.connect()
    
    def disconnect(self) -> None:
        self.connection_manager.disconnect()
    
    def subscribe_ticks(self, symbol: str, callback: Optional[Callable] = None) -> bool:
        if not self.connection_manager.is_ready():
            self.logger.error("Not connected to WebSocket")
            return False
            
        return self.subscription_manager.subscribe_ticks(symbol, callback)
    
    def unsubscribe_ticks(self, symbol: str) -> bool:
        if not self.connection_manager.is_ready():
            self.logger.error("Not connected to WebSocket")
            return False
            
        return self.subscription_manager.unsubscribe_ticks(symbol)
    
    def subscribe_candles(self, symbol: str, interval: str = "1m", callback: Optional[Callable] = None) -> bool:
        if not self.connection_manager.is_ready():
            self.logger.error("Not connected to WebSocket")
            return False
            
        return self.subscription_manager.subscribe_candles(symbol, interval, callback)
    
    def subscribe_ohlc(self, symbol: str, interval: str = "1m", callback: Optional[Callable] = None) -> bool:
        if not self.connection_manager.is_ready():
            self.logger.error("Not connected to WebSocket")
            return False
            
        return self.subscription_manager.subscribe_ohlc(symbol, interval, callback)
    
    def unsubscribe_ohlc(self, symbol: str, interval: str = "1m") -> bool:
        if not self.connection_manager.is_ready():
            self.logger.error("Not connected to WebSocket")
            return False
            
        return self.subscription_manager.unsubscribe_ohlc(symbol, interval)
    
    def add_callback(self, event_type: str, callback: Callable) -> None:
        self.callback_manager.add_callback(event_type, callback)
        self.logger.info(f"Added callback for event type: {event_type}")
    
    def remove_callback(self, event_type: str, callback: Callable) -> bool:
        result = self.callback_manager.remove_callback(event_type, callback)
        if result:
            self.logger.info(f"Removed callback for event type: {event_type}")
        else:
            self.logger.warning(f"Failed to remove callback for event type: {event_type}")
        return result
    
    def get_active_subscriptions(self) -> List[str]:
        return self.subscription_manager.get_active_subscriptions()
    
    def is_ready(self) -> bool:
        return self.connection_manager.is_ready()
    

    

        
    def _subscribe_to_configured_symbols(self):
        try:
            self.logger.info("Subscribing to configured symbols")
            
            market_data_config = self.config.get('market_data', {})
            symbols = market_data_config.get('symbols', [])
            stream_types = market_data_config.get('stream_types', ['tick'])
            candle_intervals = market_data_config.get('candle_intervals', ['1m'])
            
            if not symbols:
                self.logger.warning("No symbols configured in config file")
                symbols = ["R_10", "R_50", "R_100", "EURUSD", "GBPUSD", "USDJPY"]
                
            self.logger.info(f"Subscribing to {len(symbols)} symbols: {symbols}")
            
            for symbol in symbols:
                if 'tick' in stream_types:
                    self.subscribe_ticks(symbol)
                    
                if 'ohlc' in stream_types:
                    for interval in candle_intervals:
                        self.subscribe_ohlc(symbol, interval)
                        
                if 'candles' in stream_types:
                    for interval in candle_intervals:
                        self.subscribe_candles(symbol, interval)
            
            self.logger.info(f"Successfully subscribed to {len(symbols)} symbols with {len(stream_types)} stream types")
        except Exception as e:
            self.logger.error(f"Error subscribing to symbols: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
