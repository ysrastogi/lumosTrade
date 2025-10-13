#!/usr/bin/env python
"""
Test script for the InMemoryCache to verify cross-process persistence.
This script can be run in multiple terminal windows to see data sharing between processes.

Usage:
    python test_cache_persistence.py set --key <key> --value <value>
    python test_cache_persistence.py get --key <key>
    python test_cache_persistence.py list
    python test_cache_persistence.py info
    python test_cache_persistence.py flush
"""

import sys
import os
import time
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Make sure the script can find the project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the InMemoryCache class
try:
    from src.data_layer.aggregator.worker import InMemoryCache
    from src.data_layer.aggregator.models import SymbolMetrics
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Test InMemoryCache persistence across processes")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Set command
    set_parser = subparsers.add_parser('set', help='Set a value in the cache')
    set_parser.add_argument('--key', required=True, help='Key to set')
    set_parser.add_argument('--value', required=True, help='Value to set')
    set_parser.add_argument('--type', default='tick', choices=['tick', 'ohlc', 'metrics'], 
                          help='Type of data to store')
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Get a value from the cache')
    get_parser.add_argument('--key', required=True, help='Key to get')
    get_parser.add_argument('--type', default='tick', choices=['tick', 'ohlc', 'metrics'],
                          help='Type of data to retrieve')
    
    # List command
    subparsers.add_parser('list', help='List all values in the cache')
    
    # Info command
    subparsers.add_parser('info', help='Show Redis connection information')
    
    # Flush command
    subparsers.add_parser('flush', help='Flush the cache')
    
    args = parser.parse_args()
    
    # Get cache instance
    cache = InMemoryCache.get_instance()
    
    if args.command == 'set':
        # Create test data based on the specified type
        if args.type == 'tick':
            data = {
                'price': float(args.value),
                'timestamp': datetime.now().isoformat(),
                'process_id': os.getpid()
            }
            cache.update_tick(args.key, data)
            print(f"Set tick '{args.key}' with value: {json.dumps(data, indent=2)}")
        
        elif args.type == 'ohlc':
            data = {
                'open': float(args.value),
                'high': float(args.value) + 1.0,
                'low': float(args.value) - 1.0,
                'close': float(args.value) + 0.5,
                'timestamp': datetime.now().isoformat(),
                'process_id': os.getpid()
            }
            interval = '1m'
            cache.update_ohlc(args.key, interval, data)
            print(f"Set OHLC '{args.key}:{interval}' with value: {json.dumps(data, indent=2)}")
        
    elif args.command == 'get':
        if args.type == 'tick':
            data = cache.get_tick(args.key)
            if data:
                print(f"Got tick '{args.key}': {json.dumps(data, indent=2)}")
            else:
                print(f"No tick found for key '{args.key}'")
                
        elif args.type == 'ohlc':
            interval = '1m'
            data = cache.get_ohlc(args.key, interval)
            if data:
                print(f"Got OHLC '{args.key}:{interval}': {json.dumps(data, indent=2)}")
            else:
                print(f"No OHLC data found for key '{args.key}' and interval '{interval}'")
    
    elif args.command == 'list':
        # Get all ticks
        ticks = cache.get_all_ticks()
        print("\n--- Ticks in Cache ---")
        for symbol, data in ticks.items():
            print(f"  {symbol}: {data.get('price')} (from process: {data.get('process_id', 'unknown')})")
        
        # Get all OHLC
        ohlc_data = cache.get_all_ohlc()
        print("\n--- OHLC in Cache ---")
        for symbol, intervals in ohlc_data.items():
            for interval, data in intervals.items():
                print(f"  {symbol}:{interval}: O={data.get('open')} H={data.get('high')} L={data.get('low')} C={data.get('close')} (from process: {data.get('process_id', 'unknown')})")
        
        # Get statistics that are already JSON serializable
        stats = cache.get_serializable_stats()
        
        print("\n--- Cache Statistics ---")
        print(json.dumps(stats, indent=2))
    
    elif args.command == 'info':
        # Show Redis connection information
        info = cache.get_redis_info()
        print("\n--- Redis Connection Information ---")
        print(json.dumps(info, indent=2))
        
        # Test connection
        connected = cache.check_redis_connection()
        print(f"\nRedis connection test: {'Successful' if connected else 'Failed'}")
    
    elif args.command == 'flush':
        # Flush the cache
        result = cache.flush_cache()
        print(f"Cache flush {'successful' if result else 'failed'}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()