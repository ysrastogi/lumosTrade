"""
Test script for verifying the OHLC subscription functionality
"""

import logging
import time
import json
import sys
from typing import Dict, Any

from src.data_layer.market_stream.stream import MarketStream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TEST_OHLC")

# Optional: Reduce log noise from other modules
logging.getLogger("websocket").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def ohlc_callback(data: Dict[str, Any]) -> None:
    """Callback function for OHLC data"""
    logger.info("=" * 50)
    logger.info("OHLC DATA RECEIVED:")
    logger.info(json.dumps(data, indent=2))
    
    # Extract and display the most relevant information
    if data.get('candles'):
        symbol = data.get('echo_req', {}).get('ticks_history', 'Unknown')
        granularity = data.get('echo_req', {}).get('granularity', 0)
        candles = data.get('candles', [])
        
        if candles:
            latest = candles[-1]
            logger.info(f"Symbol: {symbol}, Granularity: {granularity}s")
            logger.info(f"Latest candle: Open={latest.get('open')}, High={latest.get('high')}, Low={latest.get('low')}, Close={latest.get('close')}")


def run_ohlc_test():
    """Run the OHLC subscription test"""
    logger.info("Starting OHLC Subscription Test...")
    
    # Create a market stream instance
    market_stream = MarketStream()
    
    # Connect to the WebSocket server
    if not market_stream.connect():
        logger.error("Failed to connect to the WebSocket server.")
        return
    
    logger.info("Connected to the WebSocket server.")
    
    # Wait for connection to be established
    time.sleep(2)
    
    # Check if the connection is ready
    if not market_stream.is_ready():
        logger.error("Connection is not ready.")
        return
    
    # Subscribe to OHLC data for a test symbol
    symbol = "R_50"
    interval = "1m"
    
    logger.info(f"Subscribing to OHLC data for {symbol} with interval {interval}...")
    success = market_stream.subscribe_ohlc(symbol, interval, ohlc_callback)
    
    if not success:
        logger.error("Failed to subscribe to OHLC data.")
        market_stream.disconnect()
        return
    
    logger.info(f"Successfully subscribed to OHLC data for {symbol}.")
    
    # Keep the connection open for a while to receive data
    try:
        logger.info("Waiting for OHLC data... (Press Ctrl+C to stop)")
        for _ in range(60):
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    finally:
        # Disconnect from the WebSocket server
        logger.info("Disconnecting from the WebSocket server...")
        market_stream.disconnect()
        logger.info("Disconnected.")


if __name__ == "__main__":
    run_ohlc_test()