#!/usr/bin/env python3
"""
Run script for the WebSocket API server
"""
import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
auth_token = os.getenv("DERIV_AUTH_TOKEN")

if auth_token:
    logger.info("DERIV_AUTH_TOKEN found in .env file")
else:
    logger.warning("DERIV_AUTH_TOKEN not found in .env file. Authentication may fail.")

# Load environment variables
load_dotenv()

async def main():
    # Import here to ensure sys.path is updated
    from src.api.ws_server import WebSocketServer
    
    # Print banner
    print("\n" + "=" * 60)
    print(" 🚀 LumosTrade WebSocket API Server")
    print("=" * 60)
    
    # Show config
    auth_token = os.getenv('DERIV_AUTH_TOKEN')
    if auth_token:
        print(f" ✅ DERIV_AUTH_TOKEN loaded: {auth_token[:10]}...")
    else:
        print(" ❌ DERIV_AUTH_TOKEN not set! Please set it in the .env file")
    
    # Create server
    ws_port = int(os.getenv('WS_PORT', '8765'))
    server = WebSocketServer()
    
    print(f"\n 📡 Starting WebSocket server on 0.0.0.0:{ws_port}")
    print(" 🔗 WebSocket URL: ws://localhost:{ws_port}")
    print("\n 📋 Available Operations:")
    print("   - Authentication (auth)")
    print("   - Market Data Streaming (subscribe/unsubscribe)")
    print("   - Account Balance (balance)")
    print("   - Active Symbols (symbols)")
    print("   - Contract Types (contracts)")
    print("   - Contract Pricing (proposal)")
    print("   - Trading (buy/sell)")
    print("   - Portfolio Management (portfolio)")
    print("   - Transaction History (profit-table/statement)")
    
    print("\n 🔌 Press CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    try:
        # Start WebSocket server
        await server.start(host="0.0.0.0", port=ws_port)
        
        # Keep running - wait for Ctrl+C
        # This creates a future that never completes, allowing the server to run
        # until the program is interrupted
        await asyncio.Future()
    
    except KeyboardInterrupt:
        print("\n\n⚡ Stopping server...")
    finally:
        # Stop server
        await server.stop()
        print("✅ Server stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Server shutdown by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.exception("Unhandled exception in main")
        sys.exit(1)