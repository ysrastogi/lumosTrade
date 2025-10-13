"""
Client connection class for WebSocket server
"""
import time
from typing import Dict, Set, Optional


class ClientConnection:
    """
    Represents a connected WebSocket client
    """
    def __init__(self, websocket, client_id: str):
        """
        Initialize a new client connection
        
        Args:
            websocket: The WebSocket connection object
            client_id: Unique ID for the client
        """
        self.websocket = websocket
        self.client_id = client_id
        self.authenticated = False
        self.auth_token = None
        self.login_id = None
        self.subscriptions: Dict[str, Set[str]] = {}  # symbol -> set of subscription_ids
        self.last_activity = time.time()
    
    def __str__(self):
        return f"Client({self.client_id}{'✓' if self.authenticated else ''})"