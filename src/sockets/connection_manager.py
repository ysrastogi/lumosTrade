"""
Connection manager for WebSocket connections
"""
import asyncio
import json
import logging
import time
import websockets
from typing import Dict, Set, Optional, Any, List

from .client_connection import ClientConnection
from .models import WSMessageType, ErrorWSMessage

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manages WebSocket client connections and subscriptions
    """
    def __init__(self, market_data_handler=None):
        """
        Initialize the connection manager
        
        Args:
            market_data_handler: Optional handler for market data
        """
        self.connections: Dict[str, ClientConnection] = {}
        self.subscription_to_clients: Dict[str, Set[str]] = {}  # subscription_id -> set of client_ids
        self.market_data_handler = market_data_handler
        
    def add_client(self, websocket, client_id: str) -> ClientConnection:
        """
        Add a new client connection
        
        Args:
            websocket: The WebSocket connection
            client_id: Unique ID for the client
            
        Returns:
            The newly created client connection
        """
        client = ClientConnection(websocket, client_id)
        self.connections[client_id] = client
        logger.info(f"New connection: {client}")
        return client
    
    def remove_client(self, client_id: str):
        """
        Remove a client and clean up subscriptions
        
        Args:
            client_id: The ID of the client to remove
        """
        if client_id not in self.connections:
            return
        
        client = self.connections[client_id]
        
        # Clean up subscriptions
        for symbol, subscription_ids in client.subscriptions.items():
            for sub_id in subscription_ids:
                if sub_id in self.subscription_to_clients:
                    self.subscription_to_clients[sub_id].discard(client_id)
                    
                    # If no more clients for this subscription, unsubscribe from market stream
                    if not self.subscription_to_clients[sub_id] and self.market_data_handler:
                        stream_type = sub_id.split('_')[1]  # Format is {symbol}_{stream_type}_{interval}
                        symbol = sub_id.split('_')[0]
                        
                        if stream_type == "tick":
                            self.market_data_handler.unsubscribe_ticks(symbol)
                        elif stream_type == "ohlc":
                            interval = sub_id.split('_')[2] if len(sub_id.split('_')) > 2 else None
                            if interval:
                                self.market_data_handler.unsubscribe_ohlc(symbol, interval)
        
        # Remove client
        del self.connections[client_id]
        logger.info(f"Removed client: {client}")
    
    def get_client(self, client_id: str) -> Optional[ClientConnection]:
        """
        Get a client by ID
        
        Args:
            client_id: The ID of the client
            
        Returns:
            The client connection or None if not found
        """
        return self.connections.get(client_id)
    
    def set_auth_token(self, client_id: str, token: str):
        """
        Set the authentication token for a client
        
        Args:
            client_id: The client ID
            token: The authentication token
        """
        client = self.get_client(client_id)
        if client:
            client.auth_token = token
    
    def set_authenticated(self, client_id: str, authenticated: bool):
        """
        Set the authentication status for a client
        
        Args:
            client_id: The client ID
            authenticated: The authentication status
        """
        client = self.get_client(client_id)
        if client:
            client.authenticated = authenticated
    
    def is_authenticated(self, client_id: str) -> bool:
        """
        Check if a client is authenticated
        
        Args:
            client_id: The client ID
            
        Returns:
            True if the client is authenticated, False otherwise
        """
        client = self.get_client(client_id)
        return client and client.authenticated
    
    def set_login_id(self, client_id: str, login_id: str):
        """
        Set the login ID for a client
        
        Args:
            client_id: The client ID
            login_id: The login ID
        """
        client = self.get_client(client_id)
        if client:
            client.login_id = login_id
    
    def update_activity(self, client_id: str):
        """
        Update the last activity timestamp for a client
        
        Args:
            client_id: The client ID
        """
        client = self.get_client(client_id)
        if client:
            client.last_activity = time.time()
    
    def add_subscription(self, client_id: str, symbol: str, subscription_id: str) -> bool:
        """
        Add a subscription for a client
        
        Args:
            client_id: The client ID
            symbol: The trading symbol
            subscription_id: The subscription ID
            
        Returns:
            True if this is the first client for this subscription, False otherwise
        """
        client = self.get_client(client_id)
        if not client:
            return False
        
        # Initialize subscription for symbol if needed
        if symbol not in client.subscriptions:
            client.subscriptions[symbol] = set()
        
        # Add subscription ID to client
        client.subscriptions[symbol].add(subscription_id)
        
        # Add client to subscription tracking
        if subscription_id not in self.subscription_to_clients:
            self.subscription_to_clients[subscription_id] = set()
        
        self.subscription_to_clients[subscription_id].add(client_id)
        
        # Return True if this is the first client for this subscription
        return len(self.subscription_to_clients[subscription_id]) == 1
    
    def remove_subscription(self, client_id: str, symbol: str, subscription_id: str) -> bool:
        """
        Remove a subscription for a client
        
        Args:
            client_id: The client ID
            symbol: The trading symbol
            subscription_id: The subscription ID
            
        Returns:
            True if there are no more clients for this subscription, False otherwise
        """
        client = self.get_client(client_id)
        if not client or symbol not in client.subscriptions or subscription_id not in client.subscriptions[symbol]:
            return False
        
        # Remove subscription from client
        client.subscriptions[symbol].remove(subscription_id)
        if not client.subscriptions[symbol]:
            del client.subscriptions[symbol]
        
        # Remove client from subscription
        if subscription_id in self.subscription_to_clients:
            self.subscription_to_clients[subscription_id].discard(client_id)
            
            # Return True if there are no more clients for this subscription
            return len(self.subscription_to_clients[subscription_id]) == 0
            
        return False
    
    def remove_all_symbol_subscriptions(self, client_id: str, symbol: str):
        """
        Remove all subscriptions for a symbol for a client
        
        Args:
            client_id: The client ID
            symbol: The trading symbol
        """
        client = self.get_client(client_id)
        if not client or symbol not in client.subscriptions:
            return
        
        # Get all subscription IDs for this symbol
        subscription_ids = list(client.subscriptions[symbol])
        
        # Remove each subscription
        for sub_id in subscription_ids:
            self.remove_subscription(client_id, symbol, sub_id)
    
    def remove_all_subscriptions(self, client_id: str):
        """
        Remove all subscriptions for a client
        
        Args:
            client_id: The client ID
        """
        client = self.get_client(client_id)
        if not client:
            return
        
        # Get all symbols
        symbols = list(client.subscriptions.keys())
        
        # Remove subscriptions for each symbol
        for symbol in symbols:
            self.remove_all_symbol_subscriptions(client_id, symbol)
    
    def get_client_subscriptions(self, client_id: str) -> Dict[str, Set[str]]:
        """
        Get all subscriptions for a client
        
        Args:
            client_id: The client ID
            
        Returns:
            A dictionary of symbols to subscription IDs
        """
        client = self.get_client(client_id)
        return client.subscriptions if client else {}
    
    def get_subscription_clients(self, subscription_id: str) -> List[str]:
        """
        Get all clients for a subscription
        
        Args:
            subscription_id: The subscription ID
            
        Returns:
            A list of client IDs
        """
        return list(self.subscription_to_clients.get(subscription_id, set()))
    
    async def send_message(self, client_id: str, message: Any):
        """
        Send a WebSocket message to a client
        
        Args:
            client_id: The client ID
            message: The message to send
        """
        client = self.get_client(client_id)
        if not client:
            logger.warning(f"Attempted to send message to unknown client: {client_id}")
            return
            
        try:
            if not isinstance(message, str):
                message = json.dumps(message)
            
            await client.websocket.send(message)
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection already closed for {client}")
            self.remove_client(client_id)
        except Exception as e:
            logger.error(f"Error sending message to {client}: {e}")
    
    async def send_error(self, client_id: str, code: str, message: str, data: Dict = None):
        """
        Send an error message to a client
        
        Args:
            client_id: The client ID
            code: The error code
            message: The error message
            data: Optional additional error data
        """
        error_msg = ErrorWSMessage(
            type=WSMessageType.ERROR,
            code=code,
            message=message,
            data=data
        )
        await self.send_message(client_id, error_msg.dict())
    
    async def send_info_message(self, client_id: str, message: str):
        """
        Send an info message to a client
        
        Args:
            client_id: The client ID
            message: The info message
        """
        await self.send_message(client_id, {
            "type": WSMessageType.INFO,
            "message": message
        })
    
    async def broadcast_to_authenticated(self, message: Any):
        """
        Broadcast a message to all authenticated clients
        
        Args:
            message: The message to broadcast
        """
        for client_id, client in list(self.connections.items()):
            if client.authenticated:
                await self.send_message(client_id, message)
    
    async def send_heartbeats(self):
        """Send heartbeat messages to all clients"""
        timestamp = int(time.time() * 1000)
        
        # Send heartbeat to all clients
        for client_id, client in list(self.connections.items()):
            try:
                await self.send_message(client_id, {
                    "type": WSMessageType.PING,
                    "timestamp": timestamp
                })
            except Exception as e:
                logger.error(f"Error sending heartbeat to {client}: {e}")
    
    async def cleanup_inactive(self, timeout_seconds: int = 300):
        """
        Clean up inactive connections
        
        Args:
            timeout_seconds: The timeout in seconds after which to remove inactive connections
        """
        current_time = time.time()
        
        # Check all connections
        for client_id, client in list(self.connections.items()):
            if current_time - client.last_activity > timeout_seconds:
                logger.info(f"Removing inactive connection: {client}")
                
                try:
                    await client.websocket.close()
                except Exception as e:
                    logger.error(f"Error closing connection for {client}: {e}")
                
                self.remove_client(client_id)
    
    def get_all_clients(self) -> List[ClientConnection]:
        """
        Get all client connections
        
        Returns:
            A list of all client connections
        """
        return list(self.connections.values())