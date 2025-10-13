"""
Cache layer for Market Stream Aggregator.

This module implements an efficient in-memory cache for storing
aggregated market data and providing fast access for dashboard APIs.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict

from src.data_layer.aggregator.models import (
    MarketSnapshot, 
    AICommentaryData,
    TradingSetup,
    MarketMapEntry,
    MarketMapResponse,
    AICommentaryResponse,
    TradeSetupResponse,
    TopSetupsResponse
)

logger = logging.getLogger(__name__)


class CacheEntry:
    """A single cache entry with value and expiration time"""
    def __init__(self, key: str, value: Any, ttl: int = 300):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl  # TTL in seconds
    
    def is_expired(self) -> bool:
        """Check if the entry is expired"""
        return time.time() > self.expires_at
    
    def __str__(self) -> str:
        return f"CacheEntry(key={self.key}, expires_in={self.expires_at - time.time():.1f}s)"


class MarketDataCache:
    """
    In-memory cache for market data and API responses.
    
    This cache stores pre-computed responses for API endpoints and
    provides fast access with automatic expiration and refresh.
    """
    
    def __init__(self):
        """Initialize the cache"""
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
        # Default TTL values
        self._ttl = {
            "market_map": 5,         # 5 seconds TTL for market map
            "ai_commentary": 300,    # 5 minutes TTL for AI commentary
            "top_setups": 300,       # 5 minutes TTL for top setups
            "snapshots": 60,         # 1 minute TTL for snapshots
            "exports": 3600,         # 1 hour TTL for exports
        }
        
        # Start cleanup thread
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._run_cleanup, daemon=True)
        self._cleanup_thread.start()
    
    def stop(self):
        """Stop the cache and cleanup thread"""
        self._running = False
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value in the cache with optional custom TTL"""
        if ttl is None:
            # Determine TTL based on key type
            for category, default_ttl in self._ttl.items():
                if key.startswith(category):
                    ttl = default_ttl
                    break
            else:
                ttl = 300  # Default TTL (5 minutes)
        
        with self._lock:
            self._cache[key] = CacheEntry(key, value, ttl)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache"""
        with self._lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired():
                return entry.value
            elif entry:
                # Remove expired entry
                del self._cache[key]
            return None
    
    def delete(self, key: str):
        """Delete a value from the cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def clear(self):
        """Clear the entire cache"""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache"""
        with self._lock:
            total_entries = len(self._cache)
            categories = {}
            
            for key, entry in self._cache.items():
                category = key.split(':', 1)[0] if ':' in key else 'other'
                if category not in categories:
                    categories[category] = 0
                categories[category] += 1
            
            return {
                "total_entries": total_entries,
                "categories": categories
            }
    
    def _run_cleanup(self):
        """Background thread to cleanup expired entries"""
        while self._running:
            try:
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
            
            # Sleep for 10 seconds before next cleanup
            time.sleep(10)
    
    def _cleanup_expired(self):
        """Remove expired entries from the cache"""
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            # Remove expired entries
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.debug(f"Removed {len(expired_keys)} expired cache entries")


# Cache keys
MARKET_MAP_KEY = "market_map:latest"
AI_COMMENTARY_KEY = "ai_commentary:latest"
TOP_SETUPS_KEY = "top_setups:latest"
SNAPSHOT_KEY_PREFIX = "snapshot:"


class AggregatorCache:
    """
    Higher-level cache for the aggregator service.
    
    This class provides domain-specific caching functionality
    on top of the generic MarketDataCache.
    """
    
    def __init__(self, cache: Optional[MarketDataCache] = None):
        """Initialize with an optional cache instance"""
        self._cache = cache or MarketDataCache()
    
    def get_market_map(self) -> Optional[MarketMapResponse]:
        """Get the cached market map response"""
        return self._cache.get(MARKET_MAP_KEY)
    
    def set_market_map(self, response: MarketMapResponse):
        """Cache a market map response"""
        self._cache.set(MARKET_MAP_KEY, response)
    
    def get_ai_commentary(self) -> Optional[AICommentaryResponse]:
        """Get the cached AI commentary response"""
        return self._cache.get(AI_COMMENTARY_KEY)
    
    def set_ai_commentary(self, response: AICommentaryResponse):
        """Cache an AI commentary response"""
        self._cache.set(AI_COMMENTARY_KEY, response)
    
    def get_top_setups(self) -> Optional[TopSetupsResponse]:
        """Get the cached top setups response"""
        return self._cache.get(TOP_SETUPS_KEY)
    
    def set_top_setups(self, response: TopSetupsResponse):
        """Cache a top setups response"""
        self._cache.set(TOP_SETUPS_KEY, response)
    
    def get_snapshot(self, timestamp: Optional[str] = None) -> Optional[MarketSnapshot]:
        """Get a cached snapshot by timestamp or the latest"""
        if timestamp:
            key = f"{SNAPSHOT_KEY_PREFIX}{timestamp}"
            return self._cache.get(key)
        else:
            return self._cache.get(f"{SNAPSHOT_KEY_PREFIX}latest")
    
    def set_snapshot(self, snapshot: MarketSnapshot):
        """Cache a market snapshot"""
        timestamp = snapshot.timestamp.isoformat()
        # Store with timestamp key
        self._cache.set(f"{SNAPSHOT_KEY_PREFIX}{timestamp}", snapshot)
        # Also store as latest
        self._cache.set(f"{SNAPSHOT_KEY_PREFIX}latest", snapshot)
    
    def invalidate_all(self):
        """Invalidate all cached data"""
        self._cache.clear()
    
    def refresh(self):
        """Invalidate only time-sensitive caches"""
        self._cache.delete(MARKET_MAP_KEY)
        # Leave AI commentary and top setups as they don't change as frequently


# Singleton instance
_cache_instance = None

def get_cache_instance() -> AggregatorCache:
    """Get or create the global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = AggregatorCache()
    return _cache_instance