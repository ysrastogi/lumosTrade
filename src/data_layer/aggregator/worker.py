import logging
import threading
import queue
import time
import os
import pickle
from typing import Dict, Any, Optional, Callable, Union, List
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
import redis

from src.data_layer.aggregator.models import (
    RawMarketTick, NormalizedMarketTick, SymbolMetrics, 
    MarketSnapshot, DirectionalBias
)
from src.data_layer.market_stream.stream import MarketStream
from src.data_layer.worker_manager import WorkerManager

logger = logging.getLogger(__name__)

class InMemoryCache:
    """
    Multi-process cache for storing market data using Redis
    This cache serves as the single source of truth for all APIs and agents across different processes
    
    Configuration via environment variables:
    - REDIS_URL: The Redis connection URL (default: redis://localhost:6379/0)
    - USE_REDIS_CACHE: Whether to use Redis for cache persistence (default: true)
    - REDIS_KEY_PREFIX: Prefix for all Redis keys (default: lumos:cache:)
    - REDIS_EXPIRE_SECONDS: TTL for cache entries in seconds, 0 means no expiry (default: 0)
    - CACHE_FALLBACK_TO_LOCAL: Whether to fall back to local cache if Redis fails (default: true)
    """
    
    _instance = None
    _redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    _use_redis = os.environ.get("USE_REDIS_CACHE", "true").lower() == "true"
    _key_prefix = os.environ.get("REDIS_KEY_PREFIX", "lumos:cache:")
    _expire_seconds = int(os.environ.get("REDIS_EXPIRE_SECONDS", "0"))
    _fallback_to_local = os.environ.get("CACHE_FALLBACK_TO_LOCAL", "true").lower() == "true"
    _connection_timeout = 5  # Redis connection timeout in seconds
    
    @classmethod
    def get_instance(cls, force_new=False):
        """
        Get singleton instance of the cache
        
        Args:
            force_new (bool): If True, create a new instance even if one already exists
                             This is useful for testing or resetting the cache
        """
        if cls._instance is None or force_new:
            cls._instance = InMemoryCache()
        return cls._instance
    
    @classmethod
    def configure(cls, redis_url=None, use_redis=None, key_prefix=None, expire_seconds=None, fallback_to_local=None):
        """
        Configure the cache settings
        
        Args:
            redis_url (str, optional): Redis connection URL
            use_redis (bool, optional): Whether to use Redis
            key_prefix (str, optional): Prefix for all Redis keys
            expire_seconds (int, optional): TTL for cache entries
            fallback_to_local (bool, optional): Whether to use local cache as fallback
        """
        if redis_url is not None:
            cls._redis_url = redis_url
        if use_redis is not None:
            cls._use_redis = use_redis
        if key_prefix is not None:
            cls._key_prefix = key_prefix
        if expire_seconds is not None:
            cls._expire_seconds = expire_seconds
        if fallback_to_local is not None:
            cls._fallback_to_local = fallback_to_local
        
        # If we already have an instance, update its connection
        if cls._instance is not None:
            cls._instance._setup_redis_connection()
    
    def __init__(self):
        self._lock = threading.RLock()
        
        # Local cache as fallback
        self._local_ticks: Dict[str, Dict[str, Any]] = {}
        self._local_ohlc: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._local_metrics: Dict[str, SymbolMetrics] = {}
        self._local_snapshot: Optional[MarketSnapshot] = None
        self._local_last_update_time: Dict[str, datetime] = {}
        self._local_stats = {
            "tick_updates": 0,
            "ohlc_updates": 0,
            "metrics_updates": 0,
            "snapshot_updates": 0,
            "last_access": None
        }
        
        # Redis connection
        self._redis = None
        self._setup_redis_connection()
    
    def _setup_redis_connection(self):
        """Initialize Redis connection if enabled"""
        if self._use_redis:
            try:
                self._redis = redis.from_url(
                    self._redis_url, 
                    socket_connect_timeout=self._connection_timeout,
                    socket_timeout=self._connection_timeout
                )
                # Test connection
                self._redis.ping()
                logger.info(f"Connected to Redis at {self._redis_url}")
                
                # Initialize cache stats in Redis if they don't exist
                if not self._redis.exists(f"{self._key_prefix}stats"):
                    self._redis.hset(f"{self._key_prefix}stats", "tick_updates", 0)
                    self._redis.hset(f"{self._key_prefix}stats", "ohlc_updates", 0)
                    self._redis.hset(f"{self._key_prefix}stats", "metrics_updates", 0)
                    self._redis.hset(f"{self._key_prefix}stats", "snapshot_updates", 0)
                    self._redis.hset(f"{self._key_prefix}stats", "last_access", self._serialize(datetime.now()))
                    
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                if self._fallback_to_local:
                    logger.warning("Falling back to local in-memory cache.")
                    self._redis = None
                else:
                    raise RuntimeError(f"Failed to connect to Redis and fallback is disabled: {e}")
    
    def _serialize(self, data):
        """Serialize data for Redis storage"""
        try:
            return pickle.dumps(data)
        except Exception as e:
            logger.error(f"Failed to serialize data: {e}")
            return None
    
    def _deserialize(self, data):
        """Deserialize data from Redis storage"""
        if data is None:
            return None
        try:
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize data: {e}")
            return None
    
    def update_tick(self, symbol: str, tick_data: Dict[str, Any]) -> None:
        with self._lock:
            now = datetime.now()
            
            # Update local cache
            self._local_ticks[symbol] = tick_data
            self._local_last_update_time["tick"] = now
            self._local_stats["tick_updates"] += 1
            
            # Update Redis if enabled
            if self._use_redis and self._redis:
                try:
                    # Store tick data with symbol as key
                    key = f"{self._key_prefix}tick:{symbol}"
                    self._redis.set(key, self._serialize(tick_data))
                    
                    # Apply expiration if configured
                    if self._expire_seconds > 0:
                        self._redis.expire(key, self._expire_seconds)
                    
                    # Update last update time
                    self._redis.hset(f"{self._key_prefix}last_update_time", "tick", self._serialize(now))
                    
                    # Increment stats counter
                    self._redis.hincrby(f"{self._key_prefix}stats", "tick_updates", 1)
                    
                    # Add symbol to list of available symbols
                    self._redis.sadd(f"{self._key_prefix}symbols", symbol)
                except Exception as e:
                    logger.error(f"Redis error in update_tick: {e}")
    
    def update_ohlc(self, symbol: str, interval: str, ohlc_data: Dict[str, Any]) -> None:
        with self._lock:
            now = datetime.now()
            
            # Update local cache
            if symbol not in self._local_ohlc:
                self._local_ohlc[symbol] = {}
            self._local_ohlc[symbol][interval] = ohlc_data
            self._local_last_update_time["ohlc"] = now
            self._local_stats["ohlc_updates"] += 1
            
            # Update Redis if enabled
            if self._use_redis and self._redis:
                try:
                    # Store OHLC data with symbol and interval as key
                    key = f"{self._key_prefix}ohlc:{symbol}:{interval}"
                    self._redis.set(key, self._serialize(ohlc_data))
                    
                    # Apply expiration if configured
                    if self._expire_seconds > 0:
                        self._redis.expire(key, self._expire_seconds)
                    
                    # Update last update time
                    self._redis.hset(f"{self._key_prefix}last_update_time", "ohlc", self._serialize(now))
                    
                    # Increment stats counter
                    self._redis.hincrby(f"{self._key_prefix}stats", "ohlc_updates", 1)
                    
                    # Add symbol and interval to sets for tracking
                    self._redis.sadd(f"{self._key_prefix}symbols", symbol)
                    self._redis.sadd(f"{self._key_prefix}intervals:{symbol}", interval)
                except Exception as e:
                    logger.error(f"Redis error in update_ohlc: {e}")
    
    def update_metrics(self, symbol: str, metrics: SymbolMetrics) -> None:
        with self._lock:
            now = datetime.now()
            
            # Update local cache
            self._local_metrics[symbol] = metrics
            self._local_last_update_time["metrics"] = now
            self._local_stats["metrics_updates"] += 1
            
            # Update Redis if enabled
            if self._use_redis and self._redis:
                try:
                    # Store metrics data
                    key = f"{self._key_prefix}metrics:{symbol}"
                    self._redis.set(key, self._serialize(metrics))
                    
                    # Apply expiration if configured
                    if self._expire_seconds > 0:
                        self._redis.expire(key, self._expire_seconds)
                    
                    # Update last update time
                    self._redis.hset(f"{self._key_prefix}last_update_time", "metrics", self._serialize(now))
                    
                    # Increment stats counter
                    self._redis.hincrby(f"{self._key_prefix}stats", "metrics_updates", 1)
                    
                    # Add symbol to set
                    self._redis.sadd(f"{self._key_prefix}symbols", symbol)
                except Exception as e:
                    logger.error(f"Redis error in update_metrics: {e}")
    
    def update_snapshot(self, snapshot: MarketSnapshot) -> None:
        with self._lock:
            now = datetime.now()
            
            # Update local cache
            self._local_snapshot = snapshot
            self._local_last_update_time["snapshot"] = now
            self._local_stats["snapshot_updates"] += 1
            
            # Update Redis if enabled
            if self._use_redis and self._redis:
                try:
                    # Store snapshot
                    key = f"{self._key_prefix}snapshot"
                    self._redis.set(key, self._serialize(snapshot))
                    
                    # Apply expiration if configured
                    if self._expire_seconds > 0:
                        self._redis.expire(key, self._expire_seconds)
                    
                    # Update last update time
                    self._redis.hset(f"{self._key_prefix}last_update_time", "snapshot", self._serialize(now))
                    
                    # Increment stats counter
                    self._redis.hincrby(f"{self._key_prefix}stats", "snapshot_updates", 1)
                except Exception as e:
                    logger.error(f"Redis error in update_snapshot: {e}")
    
    def get_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            now = datetime.now()
            
            # Update access time
            self._local_stats["last_access"] = now
            
            # Try Redis first if enabled
            if self._use_redis and self._redis:
                try:
                    self._redis.hset(f"{self._key_prefix}stats", "last_access", self._serialize(now))
                    data = self._redis.get(f"{self._key_prefix}tick:{symbol}")
                    if data:
                        return self._deserialize(data)
                except Exception as e:
                    logger.error(f"Redis error in get_tick: {e}")
            
            # Fall back to local cache
            return self._local_ticks.get(symbol)
    
    def get_all_ticks(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            now = datetime.now()
            self._local_stats["last_access"] = now
            result = {}
            
            # Try Redis first if enabled
            if self._use_redis and self._redis:
                try:
                    self._redis.hset(f"{self._key_prefix}stats", "last_access", self._serialize(now))
                    symbols = self._redis.smembers(f"{self._key_prefix}symbols")
                    for symbol in symbols:
                        symbol_str = symbol.decode('utf-8') if isinstance(symbol, bytes) else symbol
                        data = self._redis.get(f"{self._key_prefix}tick:{symbol_str}")
                        if data:
                            result[symbol_str] = self._deserialize(data)
                    
                    # If we got data from Redis, return it
                    if result:
                        return result
                except Exception as e:
                    logger.error(f"Redis error in get_all_ticks: {e}")
            
            # Fall back to local cache
            return {k: v.copy() for k, v in self._local_ticks.items()}
    
    def get_ohlc(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            now = datetime.now()
            self._local_stats["last_access"] = now
            
            # Try Redis first if enabled
            if self._use_redis and self._redis:
                try:
                    self._redis.hset(f"{self._key_prefix}stats", "last_access", self._serialize(now))
                    data = self._redis.get(f"{self._key_prefix}ohlc:{symbol}:{interval}")
                    if data:
                        return self._deserialize(data)
                except Exception as e:
                    logger.error(f"Redis error in get_ohlc: {e}")
            
            # Fall back to local cache
            return self._local_ohlc.get(symbol, {}).get(interval)
    
    def get_all_ohlc(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        with self._lock:
            now = datetime.now()
            self._local_stats["last_access"] = now
            result = {}
            
            # Try Redis first if enabled
            if self._use_redis and self._redis:
                try:
                    self._redis.hset(f"{self._key_prefix}stats", "last_access", self._serialize(now))
                    symbols = self._redis.smembers(f"{self._key_prefix}symbols")
                    
                    for symbol in symbols:
                        symbol_str = symbol.decode('utf-8') if isinstance(symbol, bytes) else symbol
                        intervals = self._redis.smembers(f"{self._key_prefix}intervals:{symbol_str}")
                        
                        if intervals:
                            result[symbol_str] = {}
                            for interval in intervals:
                                interval_str = interval.decode('utf-8') if isinstance(interval, bytes) else interval
                                data = self._redis.get(f"{self._key_prefix}ohlc:{symbol_str}:{interval_str}")
                                if data:
                                    result[symbol_str][interval_str] = self._deserialize(data)
                    
                    # If we got data from Redis, return it
                    if result:
                        return result
                except Exception as e:
                    logger.error(f"Redis error in get_all_ohlc: {e}")
            
            # Fall back to local cache
            result = {}
            for symbol, intervals in self._local_ohlc.items():
                result[symbol] = {}
                for interval, data in intervals.items():
                    result[symbol][interval] = data.copy()
            return result
    
    def get_metrics(self, symbol: str) -> Optional[SymbolMetrics]:
        with self._lock:
            now = datetime.now()
            self._local_stats["last_access"] = now
            
            # Try Redis first if enabled
            if self._use_redis and self._redis:
                try:
                    self._redis.hset(f"{self._key_prefix}stats", "last_access", self._serialize(now))
                    data = self._redis.get(f"{self._key_prefix}metrics:{symbol}")
                    if data:
                        return self._deserialize(data)
                except Exception as e:
                    logger.error(f"Redis error in get_metrics: {e}")
            
            # Fall back to local cache
            return self._local_metrics.get(symbol)
    
    def get_all_metrics(self) -> Dict[str, SymbolMetrics]:
        with self._lock:
            now = datetime.now()
            self._local_stats["last_access"] = now
            result = {}
            
            # Try Redis first if enabled
            if self._use_redis and self._redis:
                try:
                    self._redis.hset(f"{self._key_prefix}stats", "last_access", self._serialize(now))
                    symbols = self._redis.smembers(f"{self._key_prefix}symbols")
                    
                    for symbol in symbols:
                        symbol_str = symbol.decode('utf-8') if isinstance(symbol, bytes) else symbol
                        data = self._redis.get(f"{self._key_prefix}metrics:{symbol_str}")
                        if data:
                            result[symbol_str] = self._deserialize(data)
                    
                    # If we got data from Redis, return it
                    if result:
                        return result
                except Exception as e:
                    logger.error(f"Redis error in get_all_metrics: {e}")
            
            # Fall back to local cache
            return self._local_metrics.copy()
    
    def get_snapshot(self) -> Optional[MarketSnapshot]:
        with self._lock:
            now = datetime.now()
            self._local_stats["last_access"] = now
            
            # Try Redis first if enabled
            if self._use_redis and self._redis:
                try:
                    self._redis.hset(f"{self._key_prefix}stats", "last_access", self._serialize(now))
                    data = self._redis.get(f"{self._key_prefix}snapshot")
                    if data:
                        return self._deserialize(data)
                except Exception as e:
                    logger.error(f"Redis error in get_snapshot: {e}")
            
            # Fall back to local cache
            return self._local_snapshot
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            now = datetime.now()
            self._local_stats["last_access"] = now
            
            # Try Redis first if enabled
            if self._use_redis and self._redis:
                try:
                    self._redis.hset(f"{self._key_prefix}stats", "last_access", self._serialize(now))
                    
                    # Get stats from Redis
                    stats = {}
                    redis_stats = self._redis.hgetall(f"{self._key_prefix}stats")
                    for k, v in redis_stats.items():
                        k_str = k.decode('utf-8') if isinstance(k, bytes) else k
                        if k_str in ["tick_updates", "ohlc_updates", "metrics_updates", "snapshot_updates"]:
                            stats[k_str] = int(v) if isinstance(v, bytes) else v
                        else:
                            stats[k_str] = self._deserialize(v)
                    
                    # Add counts
                    symbols = self._redis.smembers(f"{self._key_prefix}symbols")
                    symbol_count = len(symbols)
                    
                    # Count OHLC entries
                    ohlc_count = 0
                    for symbol in symbols:
                        symbol_str = symbol.decode('utf-8') if isinstance(symbol, bytes) else symbol
                        intervals = self._redis.smembers(f"{self._key_prefix}intervals:{symbol_str}")
                        ohlc_count += len(intervals)
                    
                    # Get last update times
                    last_update_times = {}
                    redis_update_times = self._redis.hgetall(f"{self._key_prefix}last_update_time")
                    for k, v in redis_update_times.items():
                        k_str = k.decode('utf-8') if isinstance(k, bytes) else k
                        time_obj = self._deserialize(v)
                        if time_obj:
                            last_update_times[k_str] = time_obj.isoformat()
                    
                    stats.update({
                        "tick_count": symbol_count,
                        "ohlc_count": ohlc_count,
                        "metrics_count": symbol_count,
                        "last_update_times": last_update_times,
                        "using_redis": True
                    })
                    
                    return stats
                    
                except Exception as e:
                    logger.error(f"Redis error in get_stats: {e}")
            
            # Fall back to local cache
            stats = self._local_stats.copy()
            stats.update({
                "tick_count": len(self._local_ticks),
                "ohlc_count": sum(len(intervals) for intervals in self._local_ohlc.values()),
                "metrics_count": len(self._local_metrics),
                "last_update_times": {k: v.isoformat() for k, v in self._local_last_update_time.items()},
                "using_redis": False
            })
            return stats
    
    def flush_cache(self) -> bool:
        """
        Clear all cached data both in Redis and local memory
        
        Returns:
            bool: True if operation was successful
        """
        with self._lock:
            # Clear local cache
            self._local_ticks = {}
            self._local_ohlc = {}
            self._local_metrics = {}
            self._local_snapshot = None
            self._local_last_update_time = {}
            self._local_stats = {
                "tick_updates": 0,
                "ohlc_updates": 0,
                "metrics_updates": 0,
                "snapshot_updates": 0,
                "last_access": datetime.now()
            }
            
            # Clear Redis cache if enabled
            if self._use_redis and self._redis:
                try:
                    # Get all keys with our prefix and delete them
                    keys = self._redis.keys(f"{self._key_prefix}*")
                    if keys:
                        self._redis.delete(*keys)
                    
                    # Reinitialize stats in Redis
                    self._redis.hset(f"{self._key_prefix}stats", "tick_updates", 0)
                    self._redis.hset(f"{self._key_prefix}stats", "ohlc_updates", 0)
                    self._redis.hset(f"{self._key_prefix}stats", "metrics_updates", 0)
                    self._redis.hset(f"{self._key_prefix}stats", "snapshot_updates", 0)
                    self._redis.hset(f"{self._key_prefix}stats", "last_access", self._serialize(datetime.now()))
                    
                    return True
                except Exception as e:
                    logger.error(f"Failed to flush Redis cache: {e}")
                    return False
            return True
    
    def check_redis_connection(self) -> bool:
        """
        Check if Redis connection is active and working
        
        Returns:
            bool: True if connection is working
        """
        if not self._use_redis:
            return False
            
        try:
            if self._redis:
                self._redis.ping()
                return True
            else:
                self._setup_redis_connection()
                return self._redis is not None
        except Exception as e:
            logger.error(f"Redis connection check failed: {e}")
            return False
    
    def get_serializable_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics in a JSON-serializable format
        
        Returns:
            dict: Cache statistics with all values converted to JSON-serializable types
        """
        stats = self.get_stats()
        return self._make_serializable(stats)
    
    def _make_serializable(self, obj):
        """
        Convert an object to a JSON-serializable format by handling datetime objects
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(i) for i in obj]
        elif hasattr(obj, 'isoformat'):  # datetime objects have isoformat method
            return obj.isoformat()
        else:
            return obj
        
    def get_redis_info(self) -> Dict[str, Any]:
        """
        Get information about the Redis connection and cache configuration
        
        Returns:
            dict: Information about Redis connection and configuration
        """
        info = {
            "use_redis": self._use_redis,
            "redis_url": self._redis_url.replace(":".join(self._redis_url.split(":")[1:2]), ":*****"),  # Hide password
            "key_prefix": self._key_prefix,
            "expire_seconds": self._expire_seconds,
            "fallback_to_local": self._fallback_to_local,
            "connected": False
        }
        
        if self._use_redis and self._redis:
            try:
                redis_info = self._redis.info()
                info.update({
                    "connected": True,
                    "redis_version": redis_info.get("redis_version", "unknown"),
                    "used_memory_human": redis_info.get("used_memory_human", "unknown"),
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "uptime_in_seconds": redis_info.get("uptime_in_seconds", 0)
                })
            except Exception as e:
                logger.error(f"Failed to get Redis info: {e}")
        
        return info

class AggregatorWorker:
    
    def __init__(self, callback: Callable[[Dict[str, Any]], None], max_queue_size: int = 10000, name: str = "market_aggregator_worker"):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.running = False
        self.name = name
        self.callback = callback
        self.processed_count = 0
        self.dropped_count = 0
        self.last_processed_time = None
        self.worker_status = "idle"
        
        # Thread pool for handling aggregation tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"{name}_pool")
        
        # Cache reference
        self.cache = InMemoryCache.get_instance()
    
    def start(self) -> bool:
        if self.running:
            logger.warning("Worker is already running")
            return False
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name=self.name)
        self.worker_thread.start()
        logger.info(f"Market aggregator worker thread '{self.name}' started")
        return True
    
    def is_alive(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.is_alive()
    
    def stop(self):
        if not self.running:
            return
            
        self.running = False
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
            if self.worker_thread.is_alive():
                logger.warning("Worker thread did not terminate gracefully")
        
        logger.info(f"Market aggregator worker stopped. Stats: processed={self.processed_count}, dropped={self.dropped_count}")
    
    def add_tick(self, data: Dict[str, Any]):
        try:
            self.queue.put(data, block=False)
        except queue.Full:
            self.dropped_count += 1
            
            if self.dropped_count % 100 == 0:
                logger.warning(f"Market data queue full, dropped {self.dropped_count} items so far")
    
    def add_ohlc(self, data: Dict[str, Any]):
        self.add_tick(data)
    
    def get_status(self) -> Dict[str, Any]:
        status = {
            "running": self.running,
            "queue_size": self.queue.qsize(),
            "queue_full_percent": (self.queue.qsize() / self.queue.maxsize) * 100 if self.queue.maxsize > 0 else 0,
            "processed_count": self.processed_count,
            "dropped_count": self.dropped_count,
            "status": self.worker_status,
            "last_processed": self.last_processed_time.isoformat() if self.last_processed_time else None
        }
        
        # Add cache statistics
        cache_stats = self.cache.get_stats()
        status["cache"] = cache_stats
        
        return status
        
    def get_market_data(self) -> Dict[str, Any]:
        snapshot = self.cache.get_snapshot()
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "symbols": {},
            "market_summary": {
                "top_gainers": [],
                "top_losers": [],
                "top_volume": []
            }
        }
        
        if snapshot:
            result["market_summary"]["top_gainers"] = snapshot.top_gainers
            result["market_summary"]["top_losers"] = snapshot.top_losers
            result["market_summary"]["top_volume"] = snapshot.top_volume
            
            for symbol, metrics in snapshot.symbols.items():
                result["symbols"][symbol] = metrics.dict()
                
                tick_data = self.cache.get_tick(symbol)
                if tick_data:
                    result["symbols"][symbol]["last_tick"] = tick_data
                
                ohlc_data = {}
                for interval in ["1m", "5m", "15m", "1h"]:
                    ohlc = self.cache.get_ohlc(symbol, interval)
                    if ohlc:
                        ohlc_data[interval] = ohlc
                
                if ohlc_data:
                    result["symbols"][symbol]["ohlc"] = ohlc_data
        
        return result
    
    def get_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        metrics = self.cache.get_metrics(symbol)
        if not metrics:
            return None
            
        result = metrics.dict()
        
        tick_data = self.cache.get_tick(symbol)
        if tick_data:
            result["last_tick"] = tick_data
        
        ohlc_data = {}
        for interval in ["1m", "5m", "15m", "1h"]:
            ohlc = self.cache.get_ohlc(symbol, interval)
            if ohlc:
                ohlc_data[interval] = ohlc
        
        if ohlc_data:
            result["ohlc"] = ohlc_data
            
        return result
    
    def _worker_loop(self):
        last_snapshot_time = datetime.now()
        snapshot_interval = 5.0  # Generate snapshot every 5 seconds
        
        while self.running:
            try:
                self.worker_status = "waiting"
                
                try:
                    data = self.queue.get(timeout=1.0)
                    self.worker_status = "processing"
                except queue.Empty:
                    now = datetime.now()
                    if (now - last_snapshot_time).total_seconds() >= snapshot_interval:
                        self._generate_market_snapshot()
                        last_snapshot_time = now
                    continue
                
                try:
                    self._process_market_data(data)
                    if self.callback:
                        self.callback(data)
                    
                    self.processed_count += 1
                    self.last_processed_time = datetime.now()
                    self.queue.task_done()
                    now = datetime.now()
                    if (now - last_snapshot_time).total_seconds() >= snapshot_interval:
                        self._generate_market_snapshot()
                        last_snapshot_time = now
                        
                except Exception as e:
                    logger.error(f"Error processing market data: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            except Exception as e:
                logger.error(f"Unexpected error in market data worker: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(0.1)
                
        # Clean up
        self.thread_pool.shutdown(wait=True)
        logger.info("Market data worker thread stopped")
        
    def _process_market_data(self, data: Dict[str, Any]) -> None:
        if "tick" in data:
            tick_data = data["tick"]
            symbol = tick_data.get("symbol")
            if symbol:
                self.cache.update_tick(symbol, tick_data)
                self.thread_pool.submit(self._update_metrics_from_tick, symbol, tick_data)
                
        elif "ohlc" in data:
            ohlc_data = data["ohlc"]
            symbol = ohlc_data.get("symbol")
            interval = str(ohlc_data.get("granularity", "1m"))
            if symbol and interval:
                self.cache.update_ohlc(symbol, interval, ohlc_data)
                self.thread_pool.submit(self._update_metrics_from_ohlc, symbol, interval, ohlc_data)
                
    def _update_metrics_from_tick(self, symbol: str, tick_data: Dict[str, Any]) -> None:
        try:
            metrics = self.cache.get_metrics(symbol)
            if not metrics:
                metrics = SymbolMetrics(
                    symbol={"base": symbol, "quote": "USD", "original": symbol, "display": symbol, "asset_name": symbol},
                    last_price=tick_data.get("quote", 0.0),
                    last_updated=datetime.now()
                )
            else:
                metrics.last_price = tick_data.get("quote", metrics.last_price)
                metrics.last_updated = datetime.now()
                prev_price = metrics.last_price
                new_price = tick_data.get("quote", prev_price)
                if new_price > prev_price:
                    metrics.status = "up"
                elif new_price < prev_price:
                    metrics.status = "down"
                
            self.cache.update_metrics(symbol, metrics)
            
        except Exception as e:
            logger.error(f"Error updating metrics from tick for {symbol}: {e}")
            
    def _update_metrics_from_ohlc(self, symbol: str, interval: str, ohlc_data: Dict[str, Any]) -> None:
        try:
            metrics = self.cache.get_metrics(symbol)
            if not metrics:
                metrics = SymbolMetrics(
                    symbol={"base": symbol, "quote": "USD", "original": symbol, "display": symbol, "asset_name": symbol},
                    last_price=ohlc_data.get("close", 0.0),
                    last_updated=datetime.now()
                )
            
            open_price = ohlc_data.get("open", 0.0)
            close_price = ohlc_data.get("close", 0.0)
            
            if open_price > 0:
                pct_change = ((close_price - open_price) / open_price) * 100
                if interval == "1m":
                    metrics.price_change_1m = pct_change
                elif interval == "5m":
                    metrics.price_change_5m = pct_change
                elif interval == "15m":
                    metrics.price_change_15m = pct_change
                elif interval == "1h":
                    metrics.price_change_1h = pct_change
                
                volume = ohlc_data.get("volume", 0.0)
                if interval == "1m":
                    metrics.volume_1m = volume
                elif interval == "5m":
                    metrics.volume_5m = volume
                elif interval == "15m":
                    metrics.volume_15m = volume
                    
                # Simple directional bias based on price changes
                if metrics.price_change_1h > 0.5:
                    metrics.directional_bias = DirectionalBias.BULL
                elif metrics.price_change_1h < -0.5:
                    metrics.directional_bias = DirectionalBias.BEAR
                else:
                    metrics.directional_bias = DirectionalBias.NEUTRAL
                    
                # Update volatility (simple high-low range as percentage of open)
                if open_price > 0:
                    high = ohlc_data.get("high", open_price)
                    low = ohlc_data.get("low", open_price)
                    metrics.volatility = ((high - low) / open_price) * 100
            
            # Save updated metrics to cache
            self.cache.update_metrics(symbol, metrics)
            
        except Exception as e:
            logger.error(f"Error updating metrics from OHLC for {symbol}: {e}")
            
    def _generate_market_snapshot(self) -> None:
        """
        Generate a market snapshot from current metrics
        """
        try:
            all_metrics = self.cache.get_all_metrics()
            if not all_metrics:
                return
                
            # Create lists for top gainers, losers, and volume
            symbols_list = list(all_metrics.keys())
            
            # Sort symbols by metrics
            gainers = sorted(symbols_list, 
                             key=lambda s: all_metrics[s].price_change_1h, 
                             reverse=True)[:5]
            
            losers = sorted(symbols_list, 
                            key=lambda s: all_metrics[s].price_change_1h)[:5]
            
            volume_leaders = sorted(symbols_list, 
                                   key=lambda s: all_metrics[s].volume_15m, 
                                   reverse=True)[:5]
            
            # Create and store snapshot
            snapshot = MarketSnapshot(
                timestamp=datetime.now(),
                symbols={symbol: metrics for symbol, metrics in all_metrics.items()},
                top_gainers=gainers,
                top_losers=losers,
                top_volume=volume_leaders
            )
            
            self.cache.update_snapshot(snapshot)
            
        except Exception as e:
            logger.error(f"Error generating market snapshot: {e}")
            import traceback
            logger.error(traceback.format_exc())

# Public API function for accessing market data
def get_market_data() -> Dict[str, Any]:
    """
    Get consolidated market data from the cache.
    This is the main function for accessing market data by other components.
    
    Returns:
        Dict[str, Any]: Consolidated market data from cache
    """
    try:
        processor = MarketAggregatorProcessor.get_instance()
        return processor.get_market_data()
    except RuntimeError:
        logger.error("MarketAggregatorProcessor not initialized")
        return {
            "error": "Market data processor not initialized",
            "timestamp": datetime.now().isoformat()
        }

class MarketAggregatorProcessor:
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance of the processor"""
        if cls._instance is None:
            raise RuntimeError("MarketAggregatorProcessor not initialized")
        return cls._instance
    
    @classmethod
    def initialize(cls, market_stream: MarketStream, process_callback: Optional[Callable[[Dict[str, Any]], None]] = None, 
                 worker_name: str = "market_aggregator_worker"):
        """Initialize the singleton instance"""
        if cls._instance is None:
            cls._instance = MarketAggregatorProcessor(market_stream, process_callback, worker_name)
        return cls._instance
    
    def __init__(self, market_stream: MarketStream, process_callback: Optional[Callable[[Dict[str, Any]], None]] = None, 
                worker_name: str = "market_aggregator_worker"):
        """
        Initialize the market aggregator processor
        
        Args:
            market_stream: The market stream to subscribe to
            process_callback: Optional callback function for processing market data
            worker_name: Name of the worker for registration
        """
        self.market_stream = market_stream
        self.worker = AggregatorWorker(process_callback, name=worker_name)
        self.worker_name = worker_name
        self.cache = InMemoryCache.get_instance()
        
        # Set as singleton instance
        if MarketAggregatorProcessor._instance is None:
            MarketAggregatorProcessor._instance = self
        
    def start(self) -> bool:
        """
        Start the processor
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if not self.worker.start():
            return False

        # Subscribe to market events
        self.market_stream.add_callback("tick", self._handle_tick)
        self.market_stream.add_callback("ohlc", self._handle_ohlc)

        # Register with worker manager
        worker_manager = WorkerManager.get_instance()
        worker_manager.register_worker(self.worker_name, self.worker)
        
        logger.info("Market aggregator processor started")
        return True
    
    def is_alive(self) -> bool:
        """Check if worker is alive"""
        return self.worker.is_alive()
        
    def stop(self):
        """Stop the worker and remove callbacks"""
        # Remove callbacks
        self.market_stream.remove_callback("tick", self._handle_tick)
        self.market_stream.remove_callback("ohlc", self._handle_ohlc)
        
        # Stop worker
        self.worker.stop()
        
        logger.info("Market aggregator processor stopped")
    
    def _handle_tick(self, data: Dict[str, Any]):
        """Handle tick data from market stream"""
        # Add to worker queue for processing
        self.worker.add_tick(data)
    
    def _handle_ohlc(self, data: Dict[str, Any]):
        """Handle OHLC data from market stream"""
        # Add to worker queue for processing
        self.worker.add_ohlc(data)
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the processor"""
        return self.worker.get_status()
    
    def get_market_data(self) -> Dict[str, Any]:
        """
        Get consolidated market data - main API for other components
        
        Returns:
            Dict[str, Any]: Consolidated market data including ticks, OHLC, and metrics
        """
        return self.worker.get_market_data()
    
    def get_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get data for a specific symbol
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            Optional[Dict[str, Any]]: Symbol data or None if not found
        """
        return self.worker.get_symbol_data(symbol)