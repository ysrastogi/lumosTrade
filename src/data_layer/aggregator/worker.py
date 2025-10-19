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
    
    _instance = None
    _redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    _use_redis = os.environ.get("USE_REDIS_CACHE", "true").lower() == "true"
    _key_prefix = os.environ.get("REDIS_KEY_PREFIX", "lumos:cache:")
    _expire_seconds = int(os.environ.get("REDIS_EXPIRE_SECONDS", "0"))
    _fallback_to_local = os.environ.get("CACHE_FALLBACK_TO_LOCAL", "true").lower() == "true"
    _connection_timeout = 5  # Redis connection timeout in seconds
    
    @classmethod
    def get_instance(cls, force_new=False):
        
        if cls._instance is None or force_new:
            cls._instance = InMemoryCache()
        return cls._instance
    
    @classmethod
    def configure(cls, redis_url=None, use_redis=None, key_prefix=None, expire_seconds=None, fallback_to_local=None):

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
        
        if cls._instance is not None:
            cls._instance._setup_redis_connection()
    
    def __init__(self):
        self._lock = threading.RLock()
        
        self._redis = None
        self._setup_redis_connection()
    
    def _setup_redis_connection(self):

        if self._use_redis:
            try:
                self._redis = redis.from_url(
                    self._redis_url, 
                    socket_connect_timeout=self._connection_timeout,
                    socket_timeout=self._connection_timeout
                )
                self._redis.ping()
                logger.info(f"Connected to Redis at {self._redis_url}")
                
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
        try:
            return pickle.dumps(data)
        except Exception as e:
            logger.error(f"Failed to serialize data: {e}")
            return None
    
    def _deserialize(self, data):
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
            self._local_ticks[symbol] = tick_data
            self._local_last_update_time["tick"] = now
            self._local_stats["tick_updates"] += 1

            if self._use_redis and self._redis:
                try:
                    key = f"{self._key_prefix}tick:{symbol}"
                    self._redis.set(key, self._serialize(tick_data))
                    
                    if self._expire_seconds > 0:
                        self._redis.expire(key, self._expire_seconds)
                    
                    self._redis.hset(f"{self._key_prefix}last_update_time", "tick", self._serialize(now))
                    
                    self._redis.hincrby(f"{self._key_prefix}stats", "tick_updates", 1)
                    self._redis.sadd(f"{self._key_prefix}symbols", symbol)
                except Exception as e:
                    logger.error(f"Redis error in update_tick: {e}")
    
    def update_ohlc(self, symbol: str, interval: str, ohlc_data: Dict[str, Any]) -> None:
        with self._lock:
            now = datetime.now()
            if symbol not in self._local_ohlc:
                self._local_ohlc[symbol] = {}
            self._local_ohlc[symbol][interval] = ohlc_data
            self._local_last_update_time["ohlc"] = now
            self._local_stats["ohlc_updates"] += 1
            
            if self._use_redis and self._redis:
                try:
                    key = f"{self._key_prefix}ohlc:{symbol}:{interval}"
                    self._redis.set(key, self._serialize(ohlc_data))
                    
                    if self._expire_seconds > 0:
                        self._redis.expire(key, self._expire_seconds)
                
                    self._redis.hset(f"{self._key_prefix}last_update_time", "ohlc", self._serialize(now))
                    self._redis.hincrby(f"{self._key_prefix}stats", "ohlc_updates", 1)
                    self._redis.sadd(f"{self._key_prefix}symbols", symbol)
                    self._redis.sadd(f"{self._key_prefix}intervals:{symbol}", interval)
                except Exception as e:
                    logger.error(f"Redis error in update_ohlc: {e}")
    
    def update_metrics(self, symbol: str, metrics: SymbolMetrics) -> None:
        with self._lock:
            now = datetime.now()
            self._local_metrics[symbol] = metrics
            self._local_last_update_time["metrics"] = now
            self._local_stats["metrics_updates"] += 1
            
            if self._use_redis and self._redis:
                try:

                    key = f"{self._key_prefix}metrics:{symbol}"
                    self._redis.set(key, self._serialize(metrics))
                    if self._expire_seconds > 0:
                        self._redis.expire(key, self._expire_seconds)

                    self._redis.hset(f"{self._key_prefix}last_update_time", "metrics", self._serialize(now))
                    self._redis.hincrby(f"{self._key_prefix}stats", "metrics_updates", 1)
                    self._redis.sadd(f"{self._key_prefix}symbols", symbol)
                except Exception as e:
                    logger.error(f"Redis error in update_metrics: {e}")
    
    def update_snapshot(self, snapshot: MarketSnapshot) -> None:
        with self._lock:
            now = datetime.now()
            
            self._local_snapshot = snapshot
            self._local_last_update_time["snapshot"] = now
            self._local_stats["snapshot_updates"] += 1
            
            if self._use_redis and self._redis:
                try:
                    key = f"{self._key_prefix}snapshot"
                    self._redis.set(key, self._serialize(snapshot))
            
                    if self._expire_seconds > 0:
                        self._redis.expire(key, self._expire_seconds)
                
                    self._redis.hset(f"{self._key_prefix}last_update_time", "snapshot", self._serialize(now))
                    self._redis.hincrby(f"{self._key_prefix}stats", "snapshot_updates", 1)
                except Exception as e:
                    logger.error(f"Redis error in update_snapshot: {e}")
    
    def get_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            now = datetime.now()
            self._local_stats["last_access"] = now
    
            if self._use_redis and self._redis:
                try:
                    self._redis.hset(f"{self._key_prefix}stats", "last_access", self._serialize(now))
                    data = self._redis.get(f"{self._key_prefix}tick:{symbol}")
                    if data:
                        return self._deserialize(data)
                except Exception as e:
                    logger.error(f"Redis error in get_tick: {e}")
            
    
    def get_all_ticks(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            now = datetime.now()
            self._local_stats["last_access"] = now
            result = {}
            

            if self._use_redis and self._redis:
                try:
                    self._redis.hset(f"{self._key_prefix}stats", "last_access", self._serialize(now))
                    symbols = self._redis.smembers(f"{self._key_prefix}symbols")
                    for symbol in symbols:
                        symbol_str = symbol.decode('utf-8') if isinstance(symbol, bytes) else symbol
                        data = self._redis.get(f"{self._key_prefix}tick:{symbol_str}")
                        if data:
                            result[symbol_str] = self._deserialize(data)
                
                    if result:
                        return result
                except Exception as e:
                    logger.error(f"Redis error in get_all_ticks: {e}")
            
    
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
    
    def get_all_ohlc(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        with self._lock:
            now = datetime.now()
            self._local_stats["last_access"] = now
            result = {}
            
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
                    if result:
                        return result
                except Exception as e:
                    logger.error(f"Redis error in get_all_ohlc: {e}")
    
    def get_metrics(self, symbol: str) -> Optional[SymbolMetrics]:
        with self._lock:
            now = datetime.now()
            self._local_stats["last_access"] = now

            if self._use_redis and self._redis:
                try:
                    self._redis.hset(f"{self._key_prefix}stats", "last_access", self._serialize(now))
                    data = self._redis.get(f"{self._key_prefix}metrics:{symbol}")
                    if data:
                        return self._deserialize(data)
                except Exception as e:
                    logger.error(f"Redis error in get_metrics: {e}")
            
            return self._local_metrics.get(symbol)
    
    def get_all_metrics(self) -> Dict[str, SymbolMetrics]:
        with self._lock:
            now = datetime.now()
            self._local_stats["last_access"] = now
            result = {}
            
            if self._use_redis and self._redis:
                try:
                    self._redis.hset(f"{self._key_prefix}stats", "last_access", self._serialize(now))
                    symbols = self._redis.smembers(f"{self._key_prefix}symbols")
                    
                    for symbol in symbols:
                        symbol_str = symbol.decode('utf-8') if isinstance(symbol, bytes) else symbol
                        data = self._redis.get(f"{self._key_prefix}metrics:{symbol_str}")
                        if data:
                            result[symbol_str] = self._deserialize(data)
        
                    if result:
                        return result
                except Exception as e:
                    logger.error(f"Redis error in get_all_metrics: {e}")
    
    def get_snapshot(self) -> Optional[MarketSnapshot]:
        with self._lock:
            now = datetime.now()
            self._local_stats["last_access"] = now
            
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
            
            if self._use_redis and self._redis:
                try:
                    self._redis.hset(f"{self._key_prefix}stats", "last_access", self._serialize(now))
                
                    stats = {}
                    redis_stats = self._redis.hgetall(f"{self._key_prefix}stats")
                    for k, v in redis_stats.items():
                        k_str = k.decode('utf-8') if isinstance(k, bytes) else k
                        if k_str in ["tick_updates", "ohlc_updates", "metrics_updates", "snapshot_updates"]:
                            stats[k_str] = int(v) if isinstance(v, bytes) else v
                        else:
                            stats[k_str] = self._deserialize(v)
                    
                    symbols = self._redis.smembers(f"{self._key_prefix}symbols")
                    symbol_count = len(symbols)
                    
                    ohlc_count = 0
                    for symbol in symbols:
                        symbol_str = symbol.decode('utf-8') if isinstance(symbol, bytes) else symbol
                        intervals = self._redis.smembers(f"{self._key_prefix}intervals:{symbol_str}")
                        ohlc_count += len(intervals)
                    
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
            
    
    def flush_cache(self) -> bool:
        with self._lock:
            if self._use_redis and self._redis:
                try:
                    keys = self._redis.keys(f"{self._key_prefix}*")
                    if keys:
                        self._redis.delete(*keys)

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
    
    def get_serializable_stats(self) -> Dict[str, Any]:
        stats = self.get_stats()
        return self._make_serializable(stats)
    
    def _make_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(i) for i in obj]
        elif hasattr(obj, 'isoformat'):  # datetime objects have isoformat method
            return obj.isoformat()
        else:
            return obj
        

class AggregatorWorker:
    
    def __init__(self, callback: Callable[[Dict[str, Any]], None], max_queue_size: int = 100000, name: str = "market_aggregator_worker"):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.running = False
        self.name = name
        self.callback = callback
        self.processed_count = 0
        self.dropped_count = 0
        self.last_processed_time = None
        self.worker_status = "idle"
        
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"{name}_pool")
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
        snapshot_interval = 5.0
        
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

            if not ohlc_data:
                logger.warning(f"Empty OHLC data for {symbol}")
                return
                
            metrics = self.cache.get_metrics(symbol)
            if not metrics:
                metrics = SymbolMetrics(
                    symbol={"base": symbol, "quote": "USD", "original": symbol, "display": symbol, "asset_name": symbol},
                    last_price=float(ohlc_data.get("close", 0.0)),
                    last_updated=datetime.now()
                )
            
            try:
                open_price = float(ohlc_data.get("open", 0.0))
                close_price = float(ohlc_data.get("close", 0.0))
                high_price = float(ohlc_data.get("high", 0.0))
                low_price = float(ohlc_data.get("low", 0.0))
                volume = float(ohlc_data.get("volume", 0.0))
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting OHLC values to float for {symbol}: {e}")
                open_price = 0.0
                close_price = 0.0
                high_price = 0.0
                low_price = 0.0
                volume = 0.0
            
            if open_price > 0:
                pct_change = ((close_price - open_price) / open_price) * 100
                
                if isinstance(interval, (int, float)) or interval.isdigit():
                    granularity_map = {
                        '60': "1m",
                        '300': "5m",
                        '900': "15m",
                        '3600': "1h",
                        '14400': "4h",
                        '86400': "1d"
                    }
                    interval = granularity_map.get(str(interval), "1m")
                
                if interval == "1m":
                    metrics.price_change_1m = pct_change
                    metrics.volume_1m = volume
                elif interval == "5m":
                    metrics.price_change_5m = pct_change
                    metrics.volume_5m = volume
                elif interval == "15m":
                    metrics.price_change_15m = pct_change
                    metrics.volume_15m = volume
                elif interval == "1h":
                    metrics.price_change_1h = pct_change
                
                if metrics.price_change_1h > 0.5:
                    metrics.directional_bias = DirectionalBias.BULL
                elif metrics.price_change_1h < -0.5:
                    metrics.directional_bias = DirectionalBias.BEAR
                else:
                    metrics.directional_bias = DirectionalBias.NEUTRAL
                    
                if open_price > 0:
                    high = float(ohlc_data.get("high", open_price))
                    low = float(ohlc_data.get("low", open_price))
                    metrics.volatility = ((high - low) / open_price) * 100
            
            self.cache.update_metrics(symbol, metrics)
            
        except Exception as e:
            logger.error(f"Error updating metrics from OHLC for {symbol}: {e}")
            
    def _generate_market_snapshot(self) -> None:

        try:
            all_metrics = self.cache.get_all_metrics()
            if not all_metrics:
                return
                
            symbols_list = list(all_metrics.keys())
            gainers = sorted(symbols_list, 
                             key=lambda s: all_metrics[s].price_change_1h, 
                             reverse=True)[:5]
            
            losers = sorted(symbols_list, 
                            key=lambda s: all_metrics[s].price_change_1h)[:5]
            
            volume_leaders = sorted(symbols_list, 
                                   key=lambda s: all_metrics[s].volume_15m, 
                                   reverse=True)[:5]

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

        if cls._instance is None:
            raise RuntimeError("MarketAggregatorProcessor not initialized")
        return cls._instance
    
    @classmethod
    def initialize(cls, market_stream: MarketStream, process_callback: Optional[Callable[[Dict[str, Any]], None]] = None, 
                 worker_name: str = "market_aggregator_worker"):

        if cls._instance is None:
            cls._instance = MarketAggregatorProcessor(market_stream, process_callback, worker_name)
        return cls._instance
    
    def __init__(self, market_stream: MarketStream, process_callback: Optional[Callable[[Dict[str, Any]], None]] = None, 
                worker_name: str = "market_aggregator_worker"):

        self.market_stream = market_stream
        self.worker = AggregatorWorker(process_callback, name=worker_name)
        self.worker_name = worker_name
        self.cache = InMemoryCache.get_instance()
        
        if MarketAggregatorProcessor._instance is None:
            MarketAggregatorProcessor._instance = self
        
    def start(self) -> bool:

        if not self.worker.start():
            return False

        self.market_stream.add_callback("tick", self._handle_tick)
        self.market_stream.add_callback("ohlc", self._handle_ohlc)
        worker_manager = WorkerManager.get_instance()
        worker_manager.register_worker(self.worker_name, self.worker)
        
        logger.info("Market aggregator processor started")
        return True
    
    def is_alive(self) -> bool:

        return self.worker.is_alive()
        
    def stop(self):

        self.market_stream.remove_callback("tick", self._handle_tick)
        self.market_stream.remove_callback("ohlc", self._handle_ohlc)
        
        # Stop worker
        self.worker.stop()
        
        logger.info("Market aggregator processor stopped")
    
    def _handle_tick(self, data: Dict[str, Any]):

        self.worker.add_tick(data)
    
    def _handle_ohlc(self, data: Dict[str, Any]):

        self.worker.add_ohlc(data)
    
    def get_status(self) -> Dict[str, Any]:

        return self.worker.get_status()
    
    def get_market_data(self) -> Dict[str, Any]:

        return self.worker.get_market_data()
    
    def get_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:

        return self.worker.get_symbol_data(symbol)