"""
Adapter module to provide backward compatibility for the WebSocket server API
"""
import logging
import warnings
from typing import Dict, Any

from src.sockets.ws_server import WebSocketServer

logger = logging.getLogger(__name__)

class WebSocketServer(WebSocketServer):
    def __init__(self, connect_market_stream=True):
        warnings.warn(
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(connect_market_stream=connect_market_stream)

    async def _send_message(self, client, message: Dict):
        if hasattr(client, 'client_id'):
            await self.connection_manager.send_message(client.client_id, message)
        else:

            await self.connection_manager.send_message(client, message)
    
    async def _send_error(self, client, code: str, message: str, data: Dict = None):
        if hasattr(client, 'client_id'):
            await self.connection_manager.send_error(client.client_id, code, message, data)
        else:

            await self.connection_manager.send_error(client, code, message, data)
    
    async def _send_info_message(self, client, message: str):
        if hasattr(client, 'client_id'):
            await self.connection_manager.send_info_message(client.client_id, message)
        else:

            await self.connection_manager.send_info_message(client, message)
    
    def _remove_client(self, client_id: str):
        self.connection_manager.remove_client(client_id)