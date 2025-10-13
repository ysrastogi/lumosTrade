#!/usr/bin/env python3
"""
Test script for Athena agent with live Redis cache data.
This script tests the Athena agent's ability to access market data directly from the Redis cache.
"""

import asyncio
import logging
import sys
import json
import os
from datetime import datetime

# Make sure the script can find the project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Import the InMemoryCache class and Athena agent
try:
    from src.data_layer.aggregator.worker import InMemoryCache
    from src.agents.athena import AthenaAgent
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

async def test_athena_live():
    """Run a test of Athena agent with live cache data"""
    logger.info("Getting Redis cache instance...")
    
    # Get the cache instance directly first
    cache = InMemoryCache.get_instance()
    
    # Check Redis connection
    connected = cache.check_redis_connection()
    logger.info(f"Redis connection test: {'Successful' if connected else 'Failed'}")
    
    if not connected:
        logger.error("Failed to connect to Redis. Please make sure Redis is running.")
        return
    
    # Show Redis connection information
    info = cache.get_redis_info()
    logger.info(f"Redis connection: {info.get('redis_url', 'unknown')}")
    
    # Now create the Athena agent with our verified cache
    logger.info("Creating Athena agent with live Redis cache...")
    agent = AthenaAgent(use_llm=True)
    await agent.initialize()
    
    # Get cache statistics
    stats = cache.get_serializable_stats()
    logger.info(f"Cache statistics: {stats}")
    
    logger.info("Checking available symbols in cache...")
    # Get available symbols from the cache
    try:
        ticks = cache.get_all_ticks() or {}
        symbols = list(ticks.keys())
        if not symbols:
            symbols = ['R_10']  # Default symbol if none found
        logger.info(f"Available symbols: {symbols}")
    except Exception as e:
        logger.error(f"Error getting symbols: {e}")
        symbols = ['R_10']  # Default symbol on error
    
    # Process each available symbol
    for symbol in symbols:
        logger.info(f"Testing Athena observation for {symbol}...")
        
        try:
            # Get observation for this symbol
            result = await agent.observe(symbol, interval=60, count=100)
            
            # Check if we got an error
            if 'error' in result:
                logger.warning(f"Error in observation for {symbol}: {result['error']}")
                continue
                
            # Print key information from the result
            logger.info(f"Observation result for {symbol}:")
            
            if 'features' in result:
                logger.info(f"  Technical features: {list(result['features'].keys())[:5]}...")
            
            if 'patterns' in result:
                logger.info(f"  Found patterns: {len(result['patterns'])}")
                if result['patterns']:
                    for i, pattern in enumerate(result['patterns'][:3]):  # Show top 3
                        logger.info(f"    Pattern {i+1}: {pattern['name']} (confidence: {pattern['confidence']}%)")
            
            if 'regime' in result:
                logger.info(f"  Market regime: {result['regime']} (confidence: {result.get('regime_confidence', 0):.2f})")
            
            if 'trading_bias' in result:
                logger.info(f"  Trading bias: {result['trading_bias']}")
            
            if 'summary' in result:
                logger.info(f"  Summary: {result['summary']}")
            
            if 'trade_ideas' in result and result['trade_ideas']:
                logger.info(f"  Trade ideas: {result['trade_ideas']}")
            
            if 'confidence' in result:
                logger.info(f"  Overall confidence: {result['confidence']:.2f}")
            
            # Save the full result to a JSON file for detailed analysis
            with open(f"athena_result_{symbol.replace('-', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                json.dump(result, f, indent=2, default=str)
                logger.info(f"  Full result saved to {f.name}")
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
    
    logger.info("Test completed!")

if __name__ == "__main__":
    asyncio.run(test_athena_live())