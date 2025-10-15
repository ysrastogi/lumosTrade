"""
Test script to verify Athena's OHLC data handling and generation of synthetic candles when needed.
"""

import asyncio
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Import needed components
from agents.athena_workspace.athena import AthenaAgent
from src.data_layer.aggregator import InMemoryCache

async def test_athena_candles():
    """Test Athena's ability to handle limited candle data."""
    logger.info("Starting Athena candle handling test")
    
    # Create an instance of AthenaAgent
    agent = AthenaAgent(use_llm=False)  # Don't use LLM to keep test focused
    
    # Add a simple test symbol to the cache
    cache = InMemoryCache()
    
    # Create a simple OHLC dataset with just a few candles
    now = int(datetime.now().timestamp())
    
    # Add just 5 candles to test synthetic data generation
    for i in range(5):
        timestamp = now - i * 60  # One minute intervals
        ohlc_data = {
            'time': timestamp,
            'open': 100 + i,
            'high': 101 + i,
            'low': 99 + i,
            'close': 100.5 + i,
            'volume': 100
        }
        cache.update_ohlc("R_50", "60", ohlc_data)  # "60" = 60 second timeframe
    
    # Add tick data as well
    tick_data = {
        'symbol': 'R_50',
        'price': 100.75,
        'quote': 100.75,
        'volume': 100,
        'time': now
    }
    cache.update_tick("R_50", tick_data)
    
    # Use this cache for the agent
    agent.cache = cache
    
    # Test the _fetch_market_data method directly
    market_data = await agent._fetch_market_data("R_50", 60, 100)
    logger.info(f"Fetched market data: {market_data}")
    
    # Verify we have candles
    if 'candles' in market_data:
        candle_count = len(market_data['candles'])
        logger.info(f"Got {candle_count} candles")
        
        # Should have at least 20 candles (synthetic ones should be generated)
        assert candle_count >= 20, f"Expected at least 20 candles, got {candle_count}"
        logger.info("✅ Synthetic candles were generated successfully")
    else:
        logger.error("❌ Failed to get candles from market data")
    
    # Now test the full observe method
    context = await agent.observe("R_50", 60)
    logger.info(f"Observe context: {context}")
    
    # Check if we got valid data and not an error
    if 'error' not in context:
        logger.info("✅ Observe method completed successfully")
    else:
        logger.error(f"❌ Observe method failed with error: {context['error']}")
    
    logger.info("Test completed!")

if __name__ == "__main__":
    asyncio.run(test_athena_candles())