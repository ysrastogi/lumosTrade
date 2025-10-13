#!/usr/bin/env python
"""
Worker CLI

Command-line interface for managing workers in the lumosTrade system.
This script allows you to start, stop, and monitor workers from the command line.

Usage:
  python worker_cli.py list                      # List all available workers
  python worker_cli.py start <worker_name>       # Start a specific worker
  python worker_cli.py stop <worker_name>        # Stop a specific worker
  python worker_cli.py start-all                 # Start all registered workers
  python worker_cli.py stop-all                  # Stop all registered workers
  python worker_cli.py status [worker_name]      # Show status of a worker or all workers
  python worker_cli.py monitor                   # Monitor all workers continuously
  python worker_cli.py demo                      # Run a demo with market data worker
"""

import sys
import logging
import time
import argparse
import threading
from typing import Dict, Any, List, Optional
import json
import signal
import os
from src.data_layer.market_stream import MarketStream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
from src.data_layer.worker_manager import WorkerManager
from src.data_layer.aggregator.worker import MarketAggregatorProcessor, AggregatorWorker
from src.data_layer.market_stream import MarketStream
from src.data_layer.aggregator.market_aggregator import get_aggregator_instance


class WorkerCLI:
    """Command-line interface for worker management"""
    
    def __init__(self):
        """Initialize the CLI"""
        self.worker_manager = WorkerManager.get_instance()
        self.running = False
        self.demo_resources = {}
    
    def register_default_workers(self):
        """Register default workers for testing"""
        # For now we just have the market data worker
        if not self._is_worker_registered("market_data_worker"):
            logger.info("Registering market data worker...")
            
            # Create a demo market stream
            market_stream = MarketStream()
            market_stream.connect()
            
            # Create processor with callback to log data
            processor = MarketAggregatorProcessor(
                market_stream=market_stream,
                process_callback=self._process_market_data,
                worker_name="market_data_worker"
            )
            
            # Save resources for cleanup
            self.demo_resources["market_stream"] = market_stream
            self.demo_resources["processor"] = processor
            
            # Register with worker manager manually (processor.start would also do this, 
            # but we want to register without starting)
            self.worker_manager.register_worker("market_data_worker", processor.worker)
            
            # The MarketStream will automatically subscribe to symbols from config
            # when connected and authenticated - no need to manually subscribe
            
            logger.info("Market data worker registered successfully")
            return True
        return False
    
    def _is_worker_registered(self, name: str) -> bool:
        """Check if a worker is registered"""
        status = self.worker_manager.get_all_worker_status()
        return name in status
    
    def _process_market_data(self, data: Dict[str, Any]):
        """Process market data (demo callback)"""
        symbol = data.get("symbol", "unknown")
        price = data.get("price", 0.0)
        
        # Log only every 10th tick to avoid flooding
        if hash(f"{symbol}_{price}") % 10 == 0:
            logger.info(f"Market data: {symbol} @ {price:.2f}")
    
    def start_worker(self, name: str) -> bool:
        """Start a worker by name"""
        if not self._is_worker_registered(name):
            logger.error(f"Worker '{name}' is not registered")
            return False
        
        logger.info(f"Starting worker '{name}'...")
        result = self.worker_manager.start_worker(name)
        
        if result:
            logger.info(f"Worker '{name}' started successfully")
        else:
            logger.error(f"Failed to start worker '{name}'")
        
        return result
    
    def stop_worker(self, name: str) -> bool:
        """Stop a worker by name"""
        if not self._is_worker_registered(name):
            logger.error(f"Worker '{name}' is not registered")
            return False
        
        logger.info(f"Stopping worker '{name}'...")
        result = self.worker_manager.stop_worker(name)
        
        if result:
            logger.info(f"Worker '{name}' stopped successfully")
        else:
            logger.error(f"Failed to stop worker '{name}'")
        
        return result
    
    def start_all_workers(self) -> Dict[str, bool]:
        """Start all registered workers"""
        logger.info("Starting all workers...")
        results = self.worker_manager.start_all_workers()
        
        # Log results
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Started {success_count}/{len(results)} workers")
        
        return results
    
    def stop_all_workers(self) -> Dict[str, bool]:
        """Stop all registered workers"""
        logger.info("Stopping all workers...")
        results = self.worker_manager.stop_all_workers()
        
        # Log results
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Stopped {success_count}/{len(results)} workers")
        
        return results
    
    def get_worker_status(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of a worker or all workers"""
        if name:
            if not self._is_worker_registered(name):
                logger.error(f"Worker '{name}' is not registered")
                return {}
            
            status = self.worker_manager.get_worker_status(name)
            return {name: status}
        else:
            # Get all worker statuses
            return self.worker_manager.get_all_worker_status()
    
    def list_workers(self) -> List[str]:
        """List all registered workers"""
        status = self.worker_manager.get_all_worker_status()
        return list(status.keys())
    
    def monitor_workers(self, interval: float = 2.0):
        """Monitor all workers continuously"""
        self.running = True
        
        # Start monitoring in worker manager
        self.worker_manager.start_monitoring(interval=interval)
        
        try:
            # Display headers
            print("\n{:<20} {:<10} {:<15} {:<10} {:<15} {:<10}".format(
                "Worker Name", "Status", "Processed", "Dropped", "Queue Size", "Errors"
            ))
            print("-" * 80)
            
            # Monitor loop
            while self.running:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"Worker Monitor - Press Ctrl+C to exit - {time.strftime('%H:%M:%S')}")
                print("\n{:<20} {:<10} {:<15} {:<10} {:<15} {:<10}".format(
                    "Worker Name", "Status", "Processed", "Dropped", "Queue Size", "Errors"
                ))
                print("-" * 80)
                
                # Get all worker statuses
                statuses = self.worker_manager.get_all_worker_status()
                
                for name, status in statuses.items():
                    running = status.get("running", False)
                    running_status = "RUNNING" if running else "STOPPED"
                    processed = status.get("processed_count", 0)
                    dropped = status.get("dropped_count", 0)
                    queue_size = status.get("queue_size", 0)
                    errors = status.get("error_count", 0)
                    
                    print("{:<20} {:<10} {:<15} {:<10} {:<15} {:<10}".format(
                        name, running_status, processed, dropped, queue_size, errors
                    ))
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        
        finally:
            self.running = False
            self.worker_manager.stop_monitoring()
    
    def run_demo(self):
        """Run a demo with the market data worker"""
        logger.info("Starting market data worker demo")
        
        try:
            # Register default workers
            self.register_default_workers()
            
            # Get demo market stream and ensure it's connected
            market_stream = self.demo_resources.get("market_stream")
            if market_stream:
                if not market_stream.is_connected:
                    market_stream.connect()
                    
                # Market stream will automatically subscribe to symbols from config
                # when connected and authenticated via _subscribe_to_configured_symbols method
            
            # Start worker
            self.start_worker("market_data_worker")
            
            # Start monitoring
            self.worker_manager.start_monitoring(interval=5.0)
            
            # Monitor the worker
            self.monitor_workers()
            
        except KeyboardInterrupt:
            logger.info("Demo stopped by user")
            
        finally:
            # Clean up
            self.stop_all_workers()
            self.worker_manager.stop_monitoring()
            
            # Disconnect market stream
            market_stream = self.demo_resources.get("market_stream")
            if market_stream:
                market_stream.disconnect()
    
    def print_status_table(self, statuses: Dict[str, Dict[str, Any]]):
        """Print worker status as a formatted table"""
        if not statuses:
            print("No workers registered")
            return
            
        # Print headers
        print("\n{:<20} {:<10} {:<15} {:<10} {:<15}".format(
            "Worker Name", "Status", "Uptime", "Processed", "Errors"
        ))
        print("-" * 70)
        
        # Print each worker status
        for name, status in statuses.items():
            running = status.get("running", False)
            running_status = "RUNNING" if running else "STOPPED"
            
            uptime = status.get("uptime_seconds", 0)
            uptime_str = f"{int(uptime)}s" if uptime < 60 else f"{int(uptime/60)}m {int(uptime%60)}s"
            
            processed = status.get("processed_count", 0)
            errors = status.get("error_count", 0)
            
            print("{:<20} {:<10} {:<15} {:<10} {:<15}".format(
                name, running_status, uptime_str, processed, errors
            ))
        
        print()


def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="Worker Management CLI")
    
    # Define commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all registered workers")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start a worker")
    start_parser.add_argument("name", help="Name of the worker to start")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop a worker")
    stop_parser.add_argument("name", help="Name of the worker to stop")
    
    # Start all command
    start_all_parser = subparsers.add_parser("start-all", help="Start all registered workers")
    
    # Stop all command
    stop_all_parser = subparsers.add_parser("stop-all", help="Stop all registered workers")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show status of a worker or all workers")
    status_parser.add_argument("name", nargs="?", help="Name of the worker (optional)")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor all workers continuously")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demo with market data worker")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create CLI instance
    cli = WorkerCLI()
    
    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Stopping workers due to signal")
        cli.stop_all_workers()
        cli.running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register default workers for testing
    cli.register_default_workers()
    
    # Execute command
    if args.command == "list":
        workers = cli.list_workers()
        print(f"\nRegistered workers ({len(workers)}):")
        for worker in workers:
            print(f"  - {worker}")
        print()
    
    elif args.command == "start":
        cli.start_worker(args.name)
    
    elif args.command == "stop":
        cli.stop_worker(args.name)
    
    elif args.command == "start-all":
        cli.start_all_workers()
    
    elif args.command == "stop-all":
        cli.stop_all_workers()
    
    elif args.command == "status":
        statuses = cli.get_worker_status(args.name)
        cli.print_status_table(statuses)
    
    elif args.command == "monitor":
        try:
            cli.monitor_workers()
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
    
    elif args.command == "demo":
        cli.run_demo()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()