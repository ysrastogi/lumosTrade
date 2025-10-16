from typing import Dict, Any
from src.data_layer.aggregator.worker import InMemoryCache

def get_market_data_from_cache(symbol: str, interval: str, limit: int = 100) -> Dict[str, Any]:
    """Fetch market data from the in-memory cache."""
    cache = InMemoryCache()
    ohlc_data = cache.get_ohlc(symbol, interval)
    tick_data = cache.get_tick(symbol)
    
    # Limit the number of OHLC entries returned
    if ohlc_data and 'candles' in ohlc_data:
        ohlc_data['candles'] = ohlc_data['candles'][-limit:]
    
    return {
        'ohlc': ohlc_data,
        'tick': tick_data
    }