import logging
import time
import signal
import sys
from typing import Dict, Any
import argparse

from src.data_layer.market_stream.stream import MarketStream
from src.data_layer.aggregator.worker import MarketAggregatorProcessor, get_market_data, InMemoryCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/market_aggregator.log')
    ]
)

logger = logging.getLogger(__name__)

def process_callback(data: Dict[str, Any]) -> None:
    """Optional callback for processing market data"""
    data_type = "tick" if "tick" in data else "ohlc" if "ohlc" in data else "unknown"
    symbol = data.get("tick", {}).get("symbol") or data.get("ohlc", {}).get("symbol")
    
    if data_type == "tick":
        price = data.get("tick", {}).get("quote")
        logger.debug(f"Processing {data_type} data for {symbol}: price={price}")
        
        # Manually update the cache for debugging purposes
        if price:
            cache = InMemoryCache.get_instance()
            tick_data = {
                "symbol": symbol,
                "price": price,
                "timestamp": data.get("tick", {}).get("epoch")
            }
            cache.update_tick(symbol, tick_data)
            
    elif data_type == "ohlc":
        interval = data.get("ohlc", {}).get("granularity", "1m")
        close = data.get("ohlc", {}).get("close")
        logger.debug(f"Processing {data_type} data for {symbol} ({interval}): close={close}")

def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.info("Received termination signal. Shutting down...")
    if MarketAggregatorProcessor._instance:
        MarketAggregatorProcessor.get_instance().stop()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Market Aggregator Worker")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--symbols", type=str, default="R_10,R_25,R_50,R_75,R_100",
                        help="Comma-separated list of symbols to track")
    parser.add_argument("--api-key", type=str, help="API key for market data provider")
    parser.add_argument("--api-secret", type=str, help="API secret for market data provider")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Set to DEBUG anyway to see more information
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Handle signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize market stream
    symbols = [s.strip() for s in args.symbols.split(",")]
    logger.info(f"Starting market aggregator worker for symbols: {', '.join(symbols)}")
    
    # Get a clean cache instance
    cache = InMemoryCache.get_instance()
    logger.info(f"Initial cache stats: {cache.get_stats()}")
    
    market_stream = MarketStream()
    
    # Configure market stream with API credentials if provided
    if args.api_key and args.api_secret:
        market_stream.configure(api_key=args.api_key, api_secret=args.api_secret)
    
    # Connect to market stream
    logger.info("Connecting to market stream...")
    connected = market_stream.connect()
    
    if not connected:
        logger.error("Failed to connect to market stream. Exiting.")
        return
    
    logger.info("Successfully connected to market stream")
    
    # Initialize and start market aggregator BEFORE subscribing to ticks
    logger.info("Initializing market aggregator processor...")
    processor = MarketAggregatorProcessor.initialize(
        market_stream=market_stream,
        process_callback=process_callback  # Always use the callback for debugging
    )
    
    if not processor.start():
        logger.error("Failed to start market aggregator processor. Exiting.")
        return
    
    logger.info("Market aggregator processor started successfully")
    
    # Now subscribe to market ticks for each symbol
    logger.info("Subscribing to market ticks...")
    for symbol in symbols:
        try:
            market_stream.subscribe_ticks(symbol)
            logger.info(f"Subscribed to ticks for {symbol}")
            
            # Add a small delay between subscriptions
            time.sleep(0.5)
            
            market_stream.subscribe_ohlc(symbol, interval="1m")
            logger.info(f"Subscribed to OHLC for {symbol}")
            
            # Add another small delay
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {e}")
    
    logger.info("Market aggregator worker running. Press Ctrl+C to exit.")
    
    # Initial delay to let some data accumulate
    logger.info("Waiting for initial data to accumulate...")
    time.sleep(5)
    
    try:
        # Keep running and periodically show status
        while True:
            status = processor.get_status()
            
            logger.info(f"Worker status: running={status['running']}, "
                       f"queue={status['queue_size']}/{status['queue_full_percent']:.1f}%, "
                       f"processed={status['processed_count']}, "
                       f"dropped={status['dropped_count']}")
            
            # Check the cache directly
            cache_stats = cache.get_stats()
            logger.info(f"Cache stats: {cache_stats}")
            
            # Log some current market data
            logger.info("Getting market data from cache...")
            market_data = get_market_data()
            
            if 'error' in market_data:
                logger.error(f"Error getting market data: {market_data['error']}")
            else:
                symbols_in_cache = list(market_data.get('symbols', {}).keys())
                logger.info(f"Symbols in cache: {symbols_in_cache}")
                
                if 'market_summary' in market_data and 'top_gainers' in market_data['market_summary']:
                    gainers = market_data['market_summary']['top_gainers']
                    losers = market_data['market_summary']['top_losers']
                    
                    if gainers:
                        gainer_str = ', '.join(gainers[:3])
                        logger.info(f"Top gainers: {gainer_str}")
                    
                    if losers:
                        loser_str = ', '.join(losers[:3])
                        logger.info(f"Top losers: {loser_str}")
            
            # Wait for next update
            time.sleep(10)
            
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Stopping market aggregator worker...")
        processor.stop()
        market_stream.disconnect()
        logger.info("Market aggregator worker stopped.")

if __name__ == "__main__":
    main()