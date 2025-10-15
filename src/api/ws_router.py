import logging
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from starlette.websockets import WebSocketState
from typing import Dict, Any

from config.settings import settings
from src.sockets.ws_server import WebSocketServer
from src.adapters.ws_fastapi_adapter import patch_websocket_server

patch_websocket_server()

logger = logging.getLogger(__name__)

ws_router = APIRouter(tags=["websocket"])

ws_server = WebSocketServer(connect_market_stream=False)
ws_server_started = False

async def ensure_ws_server():
    global ws_server_started
    if not ws_server_started:
        await ws_server.start(host=settings.ws_host, port=settings.ws_port, connect_market=False)
        ws_server_started = True
    return ws_server

@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    server = await ensure_ws_server()
    await websocket.accept()
    client_id = f"fastapi_{id(websocket)}"
    message_queue = asyncio.Queue()
    
    try:
        await server._handle_client_connection(websocket, client_id, message_queue)
        
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
        server._remove_client(client_id)

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1011, reason=f"Server error: {str(e)}")
        # Clean up the connection
        server._remove_client(client_id)

@ws_router.on_event("startup")
async def startup_ws_server():

    await ensure_ws_server()

@ws_router.on_event("shutdown")
async def shutdown_ws_server():

    global ws_server_started
    if ws_server_started:
        await ws_server.stop()
        ws_server_started = False