import asyncio
import json
import logging
import time
import uuid
import websockets
from typing import Dict, List, Any, Set, Optional, Union

from src.data_layer.market_stream import MarketStream
from src.data_layer.trading_client import TradingClient

from src.sockets.client_connection import ClientConnection
from src.sockets.connection_manager import ConnectionManager
from src.sockets.market_data_handler import MarketDataHandler
from src.sockets.message_handler import MessageHandler
from src.sockets.models import WSMessageType

logger = logging.getLogger(__name__)


class WebSocketServer:
    
    def __init__(self, connect_market_stream=True):

        self.market_stream = MarketStream()
        self.trading_client = TradingClient(market_stream=self.market_stream)
        self.connection_manager = ConnectionManager()
        self.market_data_handler = MarketDataHandler(self.market_stream, self.connection_manager)
        self.message_handler = MessageHandler(
            trading_client=self.trading_client,
            connection_manager=self.connection_manager,
            market_data_handler=self.market_data_handler
        )
        
        self.connection_manager.market_data_handler = self.market_data_handler
        
        self.running = False
        self.server = None
        self.heartbeat_task = None
        self.cleanup_task = None
        self.connect_market_stream = connect_market_stream

        self.trading_client.add_portfolio_callback(self.market_data_handler.handle_portfolio_update)

        if connect_market_stream:
            self.market_data_handler.setup_callbacks()
    
    async def start(self, host: str = "0.0.0.0", port: int = 8765, connect_market=None):
        if self.running:
            return
        
        self.running = True

        should_connect = connect_market if connect_market is not None else self.connect_market_stream
        if should_connect:
            self.market_data_handler.connect()
        else:
            logger.info("Skipping market stream connection as configured")
        
        logger.info(f"Starting WebSocket server on {host}:{port}")
        self.server = await websockets.serve(self._handle_client, host, port)
        
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"WebSocket server is running on {host}:{port}")
    
    async def stop(self):
        """
        Stop the WebSocket server
        """
        if not self.running:
            return
        
        self.running = False
        
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            
        if self.cleanup_task:
            self.cleanup_task.cancel()
            
        if self.server:
            self.server.close()
            await self.server.wait_closed()
    
        for client in self.connection_manager.get_all_clients():
            try:
                await client.websocket.close()
            except Exception as e:
                logger.error(f"Error closing connection for {client}: {e}")
            
            self.connection_manager.remove_client(client.client_id)
        
        self.market_data_handler.disconnect()
        
        logger.info("WebSocket server stopped")
    
    async def _handle_client(self, websocket, path):
        client_id = str(uuid.uuid4())
        client = self.connection_manager.add_client(websocket, client_id)
        
        logger.info(f"New connection: {client}")
        
        try:
            await self.connection_manager.send_info_message(
                client_id, 
                f"Welcome to LumosTrade WebSocket API! Client ID: {client_id}"
            )
            
            async for message in websocket:
                self.connection_manager.update_activity(client_id)
                
                try:
                    data = json.loads(message)
                    await self.message_handler.handle_message(client_id, data)
                except json.JSONDecodeError:
                    await self.connection_manager.send_error(client_id, "invalid_format", "Invalid JSON format")
                except Exception as e:
                    logger.error(f"Error handling message from {client}: {e}")
                    await self.connection_manager.send_error(client_id, "internal_error", f"Internal error: {str(e)}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {client}")
        except Exception as e:
            logger.error(f"Error handling connection {client}: {e}")
        finally:
            self.connection_manager.remove_client(client_id)
    
    async def _heartbeat_loop(self):
        try:
            while self.running:
                await self.connection_manager.send_heartbeats()
                
                # Sleep until next heartbeat
                await asyncio.sleep(30)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in heartbeat loop: {e}")
    
    async def _cleanup_loop(self):
        try:
            while self.running:
                await self.connection_manager.cleanup_inactive(timeout_seconds=300)
                
                # Sleep until next check
                await asyncio.sleep(60)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")