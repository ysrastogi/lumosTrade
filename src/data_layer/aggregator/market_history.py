"""
Historical storage for market data.

This module provides an in-memory time-series storage for historical
market data to be used for trend analysis and visualization.
"""

import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import os
from pathlib import Path

from src.data_layer.aggregator.models import (
    MarketSnapshot,
    SymbolMetrics,
    NormalizedSymbol,
    DirectionalBias
)

logger = logging.getLogger(__name__)


class TimeSeriesPoint:
    """A single data point in the time series"""
    def __init__(self, timestamp: datetime, value: Any):
        self.timestamp = timestamp
        self.value = value


class TimeSeriesData:
    """A time series of data points with a fixed capacity"""
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.points = deque(maxlen=capacity)
    
    def add(self, timestamp: datetime, value: Any):
        """Add a data point to the series"""
        self.points.append(TimeSeriesPoint(timestamp, value))
    
    def get_range(self, start_time: datetime, end_time: Optional[datetime] = None) -> List[TimeSeriesPoint]:
        """Get data points within a time range"""
        if end_time is None:
            end_time = datetime.now()
            
        return [p for p in self.points if start_time <= p.timestamp <= end_time]
    
    def get_latest(self) -> Optional[TimeSeriesPoint]:
        """Get the latest data point"""
        if not self.points:
            return None
        return self.points[-1]


class MarketHistoryStorage:
    """
    In-memory historical storage for market data.
    
    Stores time series data for different market metrics and provides
    query capabilities for retrieving historical trends.
    """
    
    def __init__(self, backup_dir: Optional[str] = None):
        """Initialize the storage"""
        self._lock = threading.RLock()
        
        # Storage structures
        self._price_series: Dict[str, TimeSeriesData] = defaultdict(lambda: TimeSeriesData(capacity=1440))  # 1-day of minute data
        self._volume_series: Dict[str, TimeSeriesData] = defaultdict(lambda: TimeSeriesData(capacity=1440))
        self._sentiment_series: Dict[str, TimeSeriesData] = defaultdict(lambda: TimeSeriesData(capacity=1440))
        
        # Snapshot history
        self._snapshots: Dict[str, MarketSnapshot] = {}  # timestamp -> snapshot
        
        # Backup directory for persistence
        self.backup_dir = backup_dir
        if backup_dir:
            os.makedirs(backup_dir, exist_ok=True)
    
    def add_price_point(self, symbol: str, timestamp: datetime, price: float):
        """Add a price data point"""
        with self._lock:
            self._price_series[symbol].add(timestamp, price)
    
    def add_volume_point(self, symbol: str, timestamp: datetime, volume: float):
        """Add a volume data point"""
        with self._lock:
            self._volume_series[symbol].add(timestamp, volume)
    
    def add_sentiment_point(self, symbol: str, timestamp: datetime, sentiment: float):
        """Add a sentiment data point"""
        with self._lock:
            self._sentiment_series[symbol].add(timestamp, sentiment)
    
    def add_snapshot(self, snapshot: MarketSnapshot):
        """Store a market snapshot"""
        with self._lock:
            # Store the snapshot
            timestamp_key = snapshot.timestamp.isoformat()
            self._snapshots[timestamp_key] = snapshot
            
            # Extract and store individual metrics
            for symbol, metrics in snapshot.symbols.items():
                self.add_price_point(symbol, metrics.last_updated, metrics.last_price)
                
                if hasattr(metrics, 'volume_1m'):
                    self.add_volume_point(symbol, metrics.last_updated, metrics.volume_1m)
                
                if hasattr(metrics, 'sentiment_score'):
                    self.add_sentiment_point(symbol, metrics.last_updated, metrics.sentiment_score)
            
            # Cleanup old snapshots (keep last 24 hours)
            self._cleanup_snapshots()
    
    def get_price_history(self, symbol: str, start_time: datetime, 
                          end_time: Optional[datetime] = None) -> List[Tuple[datetime, float]]:
        """Get price history for a symbol within a time range"""
        with self._lock:
            points = self._price_series[symbol].get_range(start_time, end_time)
            return [(p.timestamp, p.value) for p in points]
    
    def get_volume_history(self, symbol: str, start_time: datetime,
                           end_time: Optional[datetime] = None) -> List[Tuple[datetime, float]]:
        """Get volume history for a symbol within a time range"""
        with self._lock:
            points = self._volume_series[symbol].get_range(start_time, end_time)
            return [(p.timestamp, p.value) for p in points]
    
    def get_sentiment_history(self, symbol: str, start_time: datetime,
                              end_time: Optional[datetime] = None) -> List[Tuple[datetime, float]]:
        """Get sentiment history for a symbol within a time range"""
        with self._lock:
            points = self._sentiment_series[symbol].get_range(start_time, end_time)
            return [(p.timestamp, p.value) for p in points]
    
    def get_snapshot(self, timestamp_key: str) -> Optional[MarketSnapshot]:
        """Get a specific snapshot by timestamp"""
        with self._lock:
            return self._snapshots.get(timestamp_key)
    
    def get_snapshots_in_range(self, start_time: datetime, 
                               end_time: Optional[datetime] = None) -> List[MarketSnapshot]:
        """Get all snapshots within a time range"""
        if end_time is None:
            end_time = datetime.now()
        
        with self._lock:
            result = []
            for timestamp_key, snapshot in self._snapshots.items():
                snapshot_time = snapshot.timestamp
                if start_time <= snapshot_time <= end_time:
                    result.append(snapshot)
            
            return sorted(result, key=lambda s: s.timestamp)
    
    def get_latest_snapshot(self) -> Optional[MarketSnapshot]:
        """Get the most recent snapshot"""
        with self._lock:
            if not self._snapshots:
                return None
                
            # Find the most recent timestamp
            latest_key = max(self._snapshots.keys())
            return self._snapshots[latest_key]
    
    def get_price_change(self, symbol: str, lookback_period: timedelta) -> Optional[float]:
        """Calculate price change over a lookback period"""
        with self._lock:
            if symbol not in self._price_series:
                return None
                
            latest = self._price_series[symbol].get_latest()
            if not latest:
                return None
                
            # Get points within lookback period
            start_time = latest.timestamp - lookback_period
            points = self._price_series[symbol].get_range(start_time, latest.timestamp)
            
            if not points or len(points) < 2:
                return None
                
            # Calculate percentage change
            start_price = points[0].value
            end_price = latest.value
            
            return ((end_price - start_price) / start_price) * 100
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols tracked in the storage"""
        with self._lock:
            return list(self._price_series.keys())
    
    def backup_to_file(self):
        """Backup the current state to files"""
        if not self.backup_dir:
            logger.warning("No backup directory specified, skipping backup")
            return
            
        try:
            # Create timestamp for the backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Backup snapshots
            latest_snapshot = self.get_latest_snapshot()
            if latest_snapshot:
                backup_path = os.path.join(self.backup_dir, f"snapshot_{timestamp}.json")
                
                # Convert to JSON-serializable format
                snapshot_dict = {
                    "timestamp": latest_snapshot.timestamp.isoformat(),
                    "symbols": {},
                    "top_gainers": latest_snapshot.top_gainers,
                    "top_losers": latest_snapshot.top_losers,
                    "top_volume": latest_snapshot.top_volume
                }
                
                # Convert symbols dictionary
                for symbol, metrics in latest_snapshot.symbols.items():
                    # Convert to dictionary for JSON serialization
                    metrics_dict = {
                        "last_price": metrics.last_price,
                        "last_updated": metrics.last_updated.isoformat(),
                        "price_change_1m": metrics.price_change_1m,
                        "price_change_5m": metrics.price_change_5m,
                        "price_change_15m": metrics.price_change_15m,
                        "price_change_1h": metrics.price_change_1h,
                        "volume_1m": metrics.volume_1m,
                        "volume_5m": metrics.volume_5m,
                        "volume_15m": metrics.volume_15m,
                        "volatility": metrics.volatility,
                        "directional_bias": metrics.directional_bias.value,
                        "sentiment_score": metrics.sentiment_score,
                        "status": metrics.status
                    }
                    
                    snapshot_dict["symbols"][symbol] = metrics_dict
                
                # Write to file
                with open(backup_path, 'w') as f:
                    json.dump(snapshot_dict, f, indent=2)
                    
                logger.info(f"Backed up snapshot to {backup_path}")
                
        except Exception as e:
            logger.error(f"Error backing up data: {e}")
    
    def restore_from_backup(self, backup_file: str) -> bool:
        """Restore data from a backup file"""
        if not os.path.exists(backup_file):
            logger.error(f"Backup file not found: {backup_file}")
            return False
            
        try:
            with open(backup_file, 'r') as f:
                data = json.load(f)
                
            # Parse snapshot
            snapshot_time = datetime.fromisoformat(data["timestamp"])
            
            symbols = {}
            for symbol, metrics_dict in data["symbols"].items():
                # Reconstruct symbol metrics
                symbol_parts = symbol.split('/')
                
                if len(symbol_parts) == 2:
                    base, quote = symbol_parts
                else:
                    base = symbol_parts[0]
                    quote = "USD"  # Default
                
                norm_symbol = NormalizedSymbol(
                    base=base,
                    quote=quote,
                    original=symbol.replace('/', ''),
                    display=symbol,
                    asset_name=base
                )
                
                metrics = SymbolMetrics(
                    symbol=norm_symbol,
                    last_price=metrics_dict["last_price"],
                    last_updated=datetime.fromisoformat(metrics_dict["last_updated"]),
                    price_change_1m=metrics_dict["price_change_1m"],
                    price_change_5m=metrics_dict["price_change_5m"],
                    price_change_15m=metrics_dict["price_change_15m"],
                    price_change_1h=metrics_dict["price_change_1h"],
                    volume_1m=metrics_dict["volume_1m"],
                    volume_5m=metrics_dict["volume_5m"],
                    volume_15m=metrics_dict["volume_15m"],
                    volatility=metrics_dict["volatility"],
                    directional_bias=DirectionalBias(metrics_dict["directional_bias"]),
                    sentiment_score=metrics_dict["sentiment_score"],
                    status=metrics_dict["status"]
                )
                
                symbols[symbol] = metrics
                
                # Store price point
                self.add_price_point(symbol, metrics.last_updated, metrics.last_price)
                self.add_volume_point(symbol, metrics.last_updated, metrics.volume_1m)
                self.add_sentiment_point(symbol, metrics.last_updated, metrics.sentiment_score)
            
            # Create snapshot
            snapshot = MarketSnapshot(
                timestamp=snapshot_time,
                symbols=symbols,
                top_gainers=data["top_gainers"],
                top_losers=data["top_losers"],
                top_volume=data["top_volume"]
            )
            
            # Store the snapshot
            self._snapshots[snapshot_time.isoformat()] = snapshot
            
            logger.info(f"Restored data from {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return False
    
    def _cleanup_snapshots(self):
        """Remove old snapshots to conserve memory"""
        with self._lock:
            if len(self._snapshots) <= 288:  # Keep up to 24 hours of 5-min snapshots
                return
                
            # Sort snapshots by time
            sorted_keys = sorted(self._snapshots.keys())
            
            # Remove oldest snapshots
            keys_to_remove = sorted_keys[:-288]  # Keep the latest 288 snapshots
            for key in keys_to_remove:
                del self._snapshots[key]
            
            logger.debug(f"Cleaned up {len(keys_to_remove)} old snapshots")


# Singleton instance
_history_instance = None

def get_history_instance(backup_dir: Optional[str] = None) -> MarketHistoryStorage:
    """Get or create the global history storage instance"""
    global _history_instance
    if _history_instance is None:
        _history_instance = MarketHistoryStorage(backup_dir=backup_dir)
    return _history_instance