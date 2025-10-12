import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any
import websocket
import threading
from datetime import datetime
import yaml
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MarketStream:
    """
    WebSocket-based market stream for Deriv API
    Handles real-time market data streaming and subscriptions
    """
    
    def __init__(self, config_path: str = "config/tradding_config.yaml"):
        self.config = self._load_config(config_path)
        self.ws_url = f"{self.config['websocket']['url']}?app_id={self.config['websocket']['app_id']}"
        self.auth_token = os.getenv('DERIV_AUTH_TOKEN')
        
        # WebSocket connection
        self.ws = None
        self.is_connected = False
        self.is_authenticated = False
        
        # Connection management
        self.reconnect_attempts = self.config['websocket']['reconnect_attempts']
        self.reconnect_delay = self.config['websocket']['reconnect_delay']
        self.heartbeat_interval = self.config['websocket']['heartbeat_interval']
        
        # Subscriptions and callbacks
        self.subscriptions = {}
        self.callbacks = {}
        self.request_id = 1
        
        # Threading
        self.connection_thread = None
        self.heartbeat_thread = None
        self.running = False
        
        # Logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def _get_next_request_id(self) -> int:
        """Get next request ID for WebSocket messages"""
        self.request_id += 1
        return self.request_id
    
    def connect(self) -> bool:
        """Establish WebSocket connection"""
        try:
            self.logger.info(f"Connecting to Deriv WebSocket: {self.ws_url}")
            
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            self.running = True
            self.connection_thread = threading.Thread(target=self.ws.run_forever)
            self.connection_thread.daemon = True
            self.connection_thread.start()
            
            # Wait for connection to establish
            timeout = 10
            start_time = time.time()
            while not self.is_connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.is_connected:
                self.logger.info("Successfully connected to Deriv WebSocket")
                if self.auth_token:
                    self._authenticate()
                return True
            else:
                self.logger.error("Failed to connect to Deriv WebSocket")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to WebSocket: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from WebSocket"""
        self.logger.info("Disconnecting from Deriv WebSocket")
        self.running = False
        self.is_connected = False
        self.is_authenticated = False
        
        if self.ws:
            self.ws.close()
        
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=1)
        
        if self.connection_thread and self.connection_thread.is_alive():
            self.connection_thread.join(timeout=1)
    
    def _on_open(self, ws):
        """WebSocket on_open callback"""
        self.logger.info("WebSocket connection opened")
        self.is_connected = True
        self._start_heartbeat()
    
    def _on_message(self, ws, message):
        """WebSocket on_message callback"""
        try:
            data = json.loads(message)
            self._handle_message(data)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing WebSocket message: {e}")
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        """WebSocket on_error callback"""
        self.logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket on_close callback"""
        self.logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_connected = False
        self.is_authenticated = False
        
        if self.running:
            self._reconnect()
    
    def _handle_message(self, data: Dict):
        """Handle incoming WebSocket messages"""
        msg_type = data.get('msg_type')
        
        # Handle all Deriv API message types
        if msg_type == 'authorize':
            self._handle_auth_response(data)
        elif msg_type == 'balance':
            self._handle_balance_response(data)
        elif msg_type == 'active_symbols':
            self._handle_active_symbols_response(data)
        elif msg_type == 'contracts_for':
            self._handle_contracts_for_response(data)
        elif msg_type == 'proposal':
            self._handle_proposal_response(data)
        elif msg_type == 'buy':
            self._handle_buy_response(data)
        elif msg_type == 'sell':
            self._handle_sell_response(data)
        elif msg_type == 'portfolio':
            self._handle_portfolio_response(data)
        elif msg_type == 'profit_table':
            self._handle_profit_table_response(data)
        elif msg_type == 'statement':
            self._handle_statement_response(data)
        elif msg_type == 'proposal_open_contract':
            self._handle_contract_update(data)
        elif msg_type == 'tick':
            self._handle_tick_data(data)
        elif msg_type == 'candles':
            self._handle_candle_data(data)
        elif msg_type == 'ohlc':
            self._handle_ohlc_data(data)
        elif msg_type == 'ping':
            self._send_pong()
        elif msg_type == 'forget':
            self._handle_forget_response(data)
        elif msg_type == 'forget_all':
            self._handle_forget_all_response(data)
        else:
            # Check if there's a specific callback for this message type
            req_id = data.get('req_id')
            if req_id and req_id in self.callbacks:
                callback = self.callbacks[req_id]
                callback(data)
                del self.callbacks[req_id]
            else:
                self.logger.debug(f"Unhandled message type: {msg_type}")
        
        # Always check for errors
        if data.get('error'):
            self._handle_error_response(data)
    
    def _authenticate(self):
        """Authenticate with Deriv API"""
        if not self.auth_token:
            self.logger.warning("No auth token provided, skipping authentication")
            return
        
        req_id = self._get_next_request_id()
        auth_request = {
            "authorize": self.auth_token,
            "req_id": req_id
        }
        
        self._send_message(auth_request)
        self.logger.info("Authentication request sent")
    
    def _handle_auth_response(self, data: Dict):
        """Handle authentication response"""
        if data.get('authorize'):
            self.is_authenticated = True
            self.logger.info("Successfully authenticated with Deriv API")
        else:
            error = data.get('error', {})
            self.logger.error(f"Authentication failed: {error.get('message', 'Unknown error')}")
    
    def _start_heartbeat(self):
        """Start heartbeat thread to keep connection alive"""
        def heartbeat():
            while self.running and self.is_connected:
                try:
                    ping_request = {"ping": 1}
                    self._send_message(ping_request)
                    time.sleep(self.heartbeat_interval)
                except Exception as e:
                    self.logger.error(f"Heartbeat error: {e}")
                    break
        
        self.heartbeat_thread = threading.Thread(target=heartbeat)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
    
    def _send_pong(self):
        """Send pong response"""
        pong_request = {"pong": 1}
        self._send_message(pong_request)
    
    def _send_message(self, message: Dict):
        """Send message through WebSocket"""
        if self.ws and self.is_connected:
            try:
                self.ws.send(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Error sending message: {e}")
        else:
            self.logger.error("WebSocket not connected, cannot send message")
    
    def _reconnect(self):
        """Attempt to reconnect to WebSocket"""
        for attempt in range(self.reconnect_attempts):
            self.logger.info(f"Reconnection attempt {attempt + 1}/{self.reconnect_attempts}")
            time.sleep(self.reconnect_delay)
            
            if self.connect():
                # Re-establish subscriptions
                self._resubscribe()
                return True
        
        self.logger.error("Failed to reconnect after maximum attempts")
        return False
    
    def subscribe_ticks(self, symbol: str, callback: Optional[Callable] = None) -> bool:
        """Subscribe to tick data for a symbol"""
        if not self.is_connected:
            self.logger.error("Not connected to WebSocket")
            return False
        
        req_id = self._get_next_request_id()
        request = {
            "ticks": symbol,
            "subscribe": 1,
            "req_id": req_id
        }
        
        if callback:
            self.callbacks[req_id] = callback
        
        self.subscriptions[f"tick_{symbol}"] = request
        self._send_message(request)
        self.logger.info(f"Subscribed to tick data for {symbol}")
        return True
    
    def subscribe_candles(self, symbol: str, interval: str = "1m", callback: Optional[Callable] = None) -> bool:
        """Subscribe to candle data for a symbol"""
        if not self.is_connected:
            self.logger.error("Not connected to WebSocket")
            return False
        
        # Convert interval to Deriv API format
        interval_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400
        }
        
        granularity = interval_map.get(interval, 60)
        req_id = self._get_next_request_id()
        
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
        self._send_message(request)
        self.logger.info(f"Subscribed to {interval} candle data for {symbol}")
        return True
    
    def _handle_tick_data(self, data: Dict):
        """Handle incoming tick data"""
        tick = data.get('tick', {})
        if tick:
            symbol = tick.get('symbol')
            price = tick.get('quote')
            timestamp = tick.get('epoch')
            
            self.logger.info(f"Tick - {symbol}: {price} at {datetime.fromtimestamp(timestamp)}")
            
            # Call any registered callbacks for this symbol
            callback_key = f"tick_{symbol}"
            if callback_key in self.callbacks:
                self.callbacks[callback_key](data)
    
    def _handle_candle_data(self, data: Dict):
        """Handle incoming candle data"""
        candles = data.get('candles', [])
        if candles:
            self.logger.info(f"Received {len(candles)} candles")
            
            # Process each candle
            for candle in candles:
                open_price = candle.get('open')
                high_price = candle.get('high')
                low_price = candle.get('low')
                close_price = candle.get('close')
                timestamp = candle.get('epoch')
                
                self.logger.debug(f"Candle - O:{open_price} H:{high_price} L:{low_price} C:{close_price} T:{datetime.fromtimestamp(timestamp)}")
    
    def _handle_ohlc_data(self, data: Dict):
        """Handle incoming OHLC data"""
        ohlc = data.get('ohlc', {})
        if ohlc:
            symbol = ohlc.get('symbol')
            open_price = ohlc.get('open')
            high_price = ohlc.get('high')
            low_price = ohlc.get('low')
            close_price = ohlc.get('close')
            timestamp = ohlc.get('epoch')
            
            self.logger.info(f"OHLC - {symbol}: O:{open_price} H:{high_price} L:{low_price} C:{close_price} at {datetime.fromtimestamp(timestamp)}")
    
    def _resubscribe(self):
        """Re-establish all subscriptions after reconnection"""
        for subscription_key, request in self.subscriptions.items():
            self._send_message(request)
            self.logger.info(f"Re-subscribed: {subscription_key}")
    
    def unsubscribe(self, subscription_key: str):
        """Remove a subscription"""
        if subscription_key in self.subscriptions:
            del self.subscriptions[subscription_key]
            self.logger.info(f"Unsubscribed: {subscription_key}")
    
    def get_active_subscriptions(self) -> List[str]:
        """Get list of active subscriptions"""
        return list(self.subscriptions.keys())
    
    def is_ready(self) -> bool:
        """Check if the stream is ready for subscriptions"""
        return self.is_connected and (not self.auth_token or self.is_authenticated)
    
    # Additional response handlers for all API message types
    def _handle_balance_response(self, data: Dict):
        """Handle balance response"""
        if data.get('balance'):
            balance_info = data['balance']
            self.logger.info(f"Balance: ${balance_info.get('balance', 0):.2f} {balance_info.get('currency', 'USD')}")
    
    def _handle_active_symbols_response(self, data: Dict):
        """Handle active symbols response"""
        if data.get('active_symbols'):
            symbols_count = len(data['active_symbols'])
            self.logger.info(f"Loaded {symbols_count} active symbols")
    
    def _handle_contracts_for_response(self, data: Dict):
        """Handle contracts for response"""
        if data.get('contracts_for'):
            symbol = data['contracts_for'].get('symbol')
            contracts_count = len(data['contracts_for'].get('available', []))
            self.logger.info(f"Loaded {contracts_count} contracts for {symbol}")
    
    def _handle_proposal_response(self, data: Dict):
        """Handle proposal response"""
        if data.get('proposal'):
            proposal = data['proposal']
            ask_price = proposal.get('ask_price', 0)
            payout = proposal.get('payout', 0)
            self.logger.info(f"Proposal: Price ${ask_price}, Payout ${payout:.2f}")
    
    def _handle_buy_response(self, data: Dict):
        """Handle buy response"""
        if data.get('buy'):
            buy_info = data['buy']
            contract_id = buy_info.get('contract_id')
            buy_price = buy_info.get('buy_price', 0)
            self.logger.info(f"Contract purchased: ID {contract_id}, Price ${buy_price}")
    
    def _handle_sell_response(self, data: Dict):
        """Handle sell response"""
        if data.get('sell'):
            sell_info = data['sell']
            contract_id = sell_info.get('contract_id')
            sell_price = sell_info.get('sell_price', 0)
            self.logger.info(f"Contract sold: ID {contract_id}, Price ${sell_price}")
    
    def _handle_portfolio_response(self, data: Dict):
        """Handle portfolio response"""
        if data.get('portfolio'):
            contracts = data['portfolio'].get('contracts', [])
            self.logger.info(f"Portfolio: {len(contracts)} open contracts")
    
    def _handle_profit_table_response(self, data: Dict):
        """Handle profit table response"""
        if data.get('profit_table'):
            transactions = data['profit_table'].get('transactions', [])
            self.logger.info(f"Profit table: {len(transactions)} transactions")
    
    def _handle_statement_response(self, data: Dict):
        """Handle statement response"""
        if data.get('statement'):
            transactions = data['statement'].get('transactions', [])
            self.logger.info(f"Statement: {len(transactions)} transactions")
    
    def _handle_contract_update(self, data: Dict):
        """Handle contract update from subscription"""
        if data.get('proposal_open_contract'):
            contract = data['proposal_open_contract']
            contract_id = contract.get('contract_id')
            current_spot = contract.get('current_spot')
            profit = contract.get('profit', 0)
            is_sold = contract.get('is_sold', False)
            
            status = "CLOSED" if is_sold else "OPEN"
            self.logger.info(f"Contract {contract_id} ({status}): Spot {current_spot}, P&L ${profit:.2f}")
    
    def _handle_forget_response(self, data: Dict):
        """Handle forget subscription response"""
        if data.get('forget'):
            subscription_id = data['forget']
            self.logger.info(f"Subscription {subscription_id} cancelled")
    
    def _handle_forget_all_response(self, data: Dict):
        """Handle forget all subscriptions response"""
        if data.get('forget_all'):
            self.logger.info("All subscriptions cancelled")
    
    def _handle_error_response(self, data: Dict):
        """Handle API error responses"""
        error = data.get('error', {})
        error_code = error.get('code', 'Unknown')
        error_message = error.get('message', 'Unknown error')
        req_id = data.get('req_id', 'Unknown')
        self.logger.error(f"API Error [{error_code}] (req_id: {req_id}): {error_message}")
