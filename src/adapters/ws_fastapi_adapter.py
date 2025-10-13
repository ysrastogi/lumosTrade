import asyncio
import logging
import json
from typing import Dict, Any, Optional

from src.sockets.client_connection import ClientConnection

logger = logging.getLogger(__name__)



async def _handle_client_connection(self, websocket, client_id, message_queue=None):
    client = ClientConnection(websocket, client_id)
    self.connections[client_id] = client
    logger.info(f"New FastAPI WebSocket connection: {client}")
    
    await self._send_info_message(
        client, 
        f"Welcome to LumosTrade WebSocket API! Client ID: {client_id}"
    )
    if message_queue:
        processor_task = asyncio.create_task(self._process_message_queue(client, message_queue))
    
    try:
        while True:
            message = await websocket.receive_text()
            client.last_activity = asyncio.get_event_loop().time()
            
            try:
                data = json.loads(message)
                await self._handle_message(client, data)
            except json.JSONDecodeError:
                await self._send_error_message(
                    client,
                    "Invalid JSON message",
                    code="INVALID_JSON"
                )
            except Exception as e:
                logger.error(f"Error handling message from {client}: {e}")
                await self._send_error_message(
                    client,
                    f"Error processing message: {str(e)}",
                    code="MESSAGE_ERROR"
                )
    finally:
        if message_queue and 'processor_task' in locals():
            processor_task.cancel()
        self._remove_client(client_id)

async def _process_message_queue(self, client, message_queue):

    while True:
        try:
            message = await message_queue.get()
            await client.websocket.send_text(json.dumps(message))
            message_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error sending message to {client}: {e}")

def patch_websocket_server():
    """Patch the WebSocketServer class with FastAPI integration methods"""
    from src.api.ws_server import WebSocketServer
    
    WebSocketServer._handle_client_connection = _handle_client_connection
    WebSocketServer._process_message_queue = _process_message_queue