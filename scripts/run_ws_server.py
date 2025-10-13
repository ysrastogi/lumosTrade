import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from sockets.ws_server_refactored import WebSocketServer

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()

async def main():
    auth_token = os.getenv('DERIV_AUTH_TOKEN')
    if auth_token:
        print(f" ✅ DERIV_AUTH_TOKEN loaded: {auth_token[:10]}...")
    else:
        print(" ❌ DERIV_AUTH_TOKEN not set! Please set it in the .env file")

    ws_port = int(os.getenv('WS_PORT', '8765'))
    server = WebSocketServer()
    
    try:
        await server.start(host="0.0.0.0", port=ws_port)
        await asyncio.Future()

    finally:
        await server.stop()
        print("✅ Server stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.exception("Unhandled exception in main")
        sys.exit(1)