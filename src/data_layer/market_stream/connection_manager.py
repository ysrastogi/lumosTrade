import json
import logging
import time
import threading
import websocket
from typing import Dict, Any, Optional, Callable

from src.data_layer.market_stream.models import RequestID

logger = logging.getLogger(__name__)


class ConnectionManager:
    
    def __init__(
        self, 
        ws_url: str, 
        auth_token: Optional[str],
        reconnect_attempts: int,
        reconnect_delay: int,
        heartbeat_interval: int,
        message_handler: Callable[[Dict[str, Any]], None]
    ):
        self.logger = logger.getChild("ConnectionManager")
        self.ws_url = ws_url
        self.auth_token = auth_token
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.heartbeat_interval = heartbeat_interval
        self.message_handler = message_handler
        
        self.ws = None
        self.is_connected = False
        self.is_authenticated = False
        self.running = False
        self.connection_thread = None
        self.heartbeat_thread = None
        self._request_id = 1
    
    def get_next_request_id(self) -> RequestID:
        self._request_id += 1
        return self._request_id
    
    def connect(self) -> bool:
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
        self.logger.info("WebSocket connection opened")
        self.is_connected = True
    
        if self.auth_token:
            self._authenticate()
        else:
            ping_request = {
                "ping": 1
            }
            self.send_message(ping_request)
        
        self._start_heartbeat()
    
    def _on_message(self, ws, message):
        try:
            # Log the raw message for debugging
            self.logger.debug(f"Raw message received: {message}")
            
            data = json.loads(message)
            
            # Check for error responses
            if "error" in data:
                self.logger.error(f"API Error [{data['error'].get('code', 'Unknown')}]: {data['error'].get('message', 'Unknown error')}")
            
            # Handle ping-pong for keepalive
            if data.get("msg_type") == "ping":
                self.logger.debug("Received ping response - connection is alive")
                # No need to respond as this is a response to our ping
                
            # Check for authorization response
            if data.get("msg_type") == "authorize":
                self.logger.info("Authorization successful")
                self.is_authenticated = True
            
            # Process all messages through the handler
            self.message_handler(data)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing WebSocket message: {e}")
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        self.logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        self.logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_connected = False
        self.is_authenticated = False
        
        if self.running:
            self._reconnect()
    
    def _start_heartbeat(self):
        def heartbeat():
            while self.running and self.is_connected:
                try:
                    # Deriv API ping format
                    ping_request = {
                        "ping": 1
                    }
                    self.logger.debug("Sending heartbeat ping")
                    self.send_message(ping_request)
                    time.sleep(self.heartbeat_interval)
                except Exception as e:
                    self.logger.error(f"Heartbeat error: {e}")
                    break
        
        self.heartbeat_thread = threading.Thread(target=heartbeat)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
    
    def send_message(self, message: Dict):
        if self.ws and self.is_connected:
            try:
                self.ws.send(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Error sending message: {e}")
        else:
            self.logger.error("WebSocket not connected, cannot send message")
    
    def _reconnect(self) -> bool:
        for attempt in range(self.reconnect_attempts):
            self.logger.info(f"Reconnection attempt {attempt + 1}/{self.reconnect_attempts}")
            time.sleep(self.reconnect_delay)
            
            if self.connect():
                return True
        
        self.logger.error("Failed to reconnect after maximum attempts")
        return False
    
    def _authenticate(self):
        if not self.auth_token:
            self.logger.warning("No auth token provided, skipping authentication")
            return
        self.logger.info(f"Authenticating with Deriv API using token: {self.auth_token[:5]}...")
        
        auth_request = {
            "authorize": self.auth_token
        }
        
        self.send_message(auth_request)
        self.logger.info("Authentication request sent")
    
    def set_authenticated(self, is_authenticated: bool):
        self.is_authenticated = is_authenticated
    
    def is_ready(self) -> bool:
        return self.is_connected and (not self.auth_token or self.is_authenticated)