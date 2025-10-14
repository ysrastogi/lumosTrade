import logging
import time
from datetime import datetime, timedelta
from threading import Thread, Lock
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import asyncio
import json
import statistics
from collections import deque, defaultdict

from src.data_layer.market_stream.stream import MarketStream
from src.data_layer.aggregator.worker import MarketAggregatorProcessor
from src.data_layer.aggregator.models import (
    DirectionalBias,
    NormalizedSymbol,
    RawMarketTick,
    NormalizedMarketTick,
    SymbolMetrics,
    MarketSnapshot,
    AICommentaryData,
    SetupType,
    TradingSetup,
)

# Configure logging
logger = logging.getLogger(__name__)


class MarketDataAggregator:
    """
    Aggregates real-time market data from the MarketStream.
    
    Responsibilities:
    - Data normalization from multiple sources
    - Computing time-window metrics (1m, 5m, 15m changes)
    - Calculating derived indicators (volatility, sentiment)
    - Maintaining real-time cache of latest aggregated data
    - Providing snapshot data to dashboard APIs
    """
    
    # Symbol mapping for known assets
    ASSET_NAMES = {
        "BTC": "Bitcoin",
        "ETH": "Ethereum",
        "SOL": "Solana", 
        "DOGE": "Dogecoin",
        "AVAX": "Avalanche",
        "BNB": "BNB",
        "ADA": "Cardano",
        "DOT": "Polkadot",
        "XRP": "XRP",
        "MATIC": "Polygon",
        "LINK": "Chainlink",
        "UNI": "Uniswap",
        "AAVE": "Aave",
        "COMP": "Compound",
        "ATOM": "Cosmos",
        # Add more as needed
    }
    
    def __init__(self, market_stream: Optional[MarketStream] = None):
        self.market_stream = market_stream or MarketStream()
        
        # Cache storage for aggregated data
        self._lock = Lock()
        self._symbols_cache: Dict[str, SymbolMetrics] = {}  # Latest metrics by symbol
        self._historical_cache: Dict[str, Dict[str, deque]] = {}  # Initialize as empty dict, not defaultdict
        
        # Timeframes for metrics calculation (in seconds)
        self._timeframes = {
            "1m": 60,
            "5m": 300, 
            "15m": 900,
            "1h": 3600
        }
        
        self._history_limits = {
            "ticks": 1000,  # Keep 1000 ticks per symbol
            "1m": 120,    # Keep 2 hours of 1m data (increased from 60)
            "5m": 120,    # Keep 10 hours of 5m data (increased from 72)
            "15m": 120,   # Keep 30 hours of 15m data (increased from 96)
            "1h": 168,    # Keep 7 days of 1h data
            "4h": 60,     # Keep 10 days of 4h data
            "1d": 60      # Keep 60 days of daily data
        }
        self._snapshots: List[MarketSnapshot] = []
        self._last_commentary: Optional[AICommentaryData] = None
        self._worker_processor: Optional[MarketAggregatorProcessor] = None
        self._last_top_setups: List[TradingSetup] = []
        self._initialize_from_config()

        self._running = False
        self._aggregation_thread: Optional[Thread] = None
        self._snapshot_thread: Optional[Thread] = None
        self._worker_processor: Optional[MarketAggregatorProcessor] = None
    
    def start(self) -> bool:
        if self._running:
            logger.warning("Aggregator is already running")
            return True
            
        if  not self.market_stream.connect():
            logger.error("Failed to connect to market stream")
            return False
        
        self._worker_processor = MarketAggregatorProcessor(
            self.market_stream,
            self._process_worker_data
        )
        
        if not self._worker_processor.start():
            logger.error("Failed to start market aggregator worker")
            return False
    
        self._running = True
        self._aggregation_thread = Thread(target=self._run_aggregation_loop, daemon=True)
        self._aggregation_thread.start()
        self._snapshot_thread = Thread(target=self._run_snapshot_loop, daemon=True)
        self._snapshot_thread.start()
        self._subscribe_to_market_data()
        
        logger.info("Market data aggregator started successfully")
        return True
    
    def _process_worker_data(self, data: Dict[str, Any]):
        """Process data from the worker thread"""
        # Determine the data type
        if "tick" in data:
            self._process_tick(data)
        elif "ohlc" in data:
            self._process_ohlc(data)
        else:
            logger.warning(f"Unknown data type received from worker: {data.keys()}")
        
    def _initialize_from_config(self):
        """Initialize data structures from configuration"""
        try:
            # Get config from market stream if available
            config = getattr(self.market_stream, 'config', None)
            
            if not config:
                logger.warning("No config available in market stream, using default symbols")
                symbols = ["R_10", "R_50", "R_100", "EURUSD", "GBPUSD", "USDJPY"]
            else:
                # Get symbols from configuration
                symbols = config.get('market_data', {}).get('symbols', [])
                if not symbols:
                    logger.warning("No symbols found in config, using default symbols")
                    symbols = ["R_10", "R_50", "R_100", "EURUSD", "GBPUSD", "USDJPY"]
            
            # Initialize history storage for each symbol
            for symbol in symbols:
                self._initialize_symbol_history(symbol)
            
            # Initialize snapshots storage with a capacity limit
            self._snapshots_capacity = 288  # Store 24h of 5-min snapshots
            
            logger.info(f"Initialized market data structures for {len(symbols)} symbols")
        
        except Exception as e:
            logger.error(f"Error initializing from config: {str(e)}")
            # Initialize with default symbols in case of error
            default_symbols = ["R_10", "R_50", "R_100"]
            for symbol in default_symbols:
                self._initialize_symbol_history(symbol)
    
    def _subscribe_to_market_data(self):
        """Subscribe to market data streams for all configured symbols"""
        try:
            # Get config from market stream
            config = self.market_stream.config
            
            if not config:
                logger.error("No configuration found in market stream")
                return
                
            # Get symbols and stream types from config
            symbols = config.get('market_data', {}).get('symbols', [])
            stream_types = config.get('market_data', {}).get('stream_types', ['tick'])
            candle_intervals = config.get('market_data', {}).get('candle_intervals', ['1m'])
            
            if not symbols:
                logger.warning("No symbols defined in configuration")
                # Use some default symbols
                symbols = ["R_10", "R_50", "R_100", "EURUSD", "GBPUSD", "USDJPY"]
                
            logger.info(f"Subscribing to market data for {len(symbols)} symbols")
            
            # Subscribe to all configured symbols and stream types
            for symbol in symbols:
                logger.info(f"Subscribing to data streams for {symbol}")
                
                if 'tick' in stream_types:
                    try:
                        self.market_stream.subscribe_ticks(symbol)
                        logger.info(f"Successfully subscribed to tick data for {symbol}")
                    except Exception as e:
                        logger.error(f"Error subscribing to tick data for {symbol}: {e}")
                    
                if 'ohlc' in stream_types:
                    for interval in candle_intervals:
                        try:
                            self.market_stream.subscribe_ohlc(symbol, interval)
                            logger.info(f"Successfully subscribed to OHLC data for {symbol} with interval {interval}")
                        except Exception as e:
                            logger.error(f"Error subscribing to OHLC data for {symbol} with interval {interval}: {e}")
                        
                if 'candles' in stream_types:
                    for interval in candle_intervals:
                        try:
                            self.market_stream.subscribe_candles(symbol, interval)
                            logger.info(f"Successfully subscribed to candle data for {symbol} with interval {interval}")
                        except Exception as e:
                            logger.error(f"Error subscribing to candle data for {symbol} with interval {interval}: {e}")
            
            logger.info("Successfully subscribed to all market data streams")
        except Exception as e:
            logger.error(f"Error subscribing to market data: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def stop(self):
        """Stop the aggregator and clean up resources"""
        if not self._running:
            return
            
        self._running = False
        
        # Stop the worker processor first
        if self._worker_processor:
            self._worker_processor.stop()
            self._worker_processor = None
        
        # Wait for threads to terminate
        if self._aggregation_thread and self._aggregation_thread.is_alive():
            self._aggregation_thread.join(timeout=2.0)
            
        if self._snapshot_thread and self._snapshot_thread.is_alive():
            self._snapshot_thread.join(timeout=2.0)
        
        logger.info("Market data aggregator stopped")
    
    def _normalize_symbol(self, raw_symbol: str) -> NormalizedSymbol:
        """
        Convert raw symbols into normalized format
        
        Examples:
        - BTCUSD -> BTC/USD
        - BTC-USD -> BTC/USD
        - ETHBTC -> ETH/BTC
        """
        # Handle common formats
        if "-" in raw_symbol:
            base, quote = raw_symbol.split("-", 1)
        elif "/" in raw_symbol:
            base, quote = raw_symbol.split("/", 1)
        elif len(raw_symbol) >= 6:
            # Try to detect format like "BTCUSD"
            # Common quote currencies
            quotes = ["USD", "USDT", "USDC", "EUR", "GBP", "JPY", "BTC", "ETH"]
            
            for quote in quotes:
                if raw_symbol.endswith(quote):
                    base = raw_symbol[:-len(quote)]
                    return NormalizedSymbol(
                        base=base,
                        quote=quote,
                        original=raw_symbol,
                        display=f"{base}/{quote}",
                        asset_name=self.ASSET_NAMES.get(base, base)
                    )
            
            # Default handling for unknown patterns - take first 3 chars as base
            base = raw_symbol[:3]
            quote = raw_symbol[3:6]
        else:
            # Handle short symbols like R_50, R_100 from Deriv
            base = raw_symbol
            quote = "USD"  # Default quote
        
        # Create normalized symbol
        return NormalizedSymbol(
            base=base,
            quote=quote,
            original=raw_symbol,
            display=f"{base}/{quote}",
            asset_name=self.ASSET_NAMES.get(base, base)
        )
    
    def _process_tick(self, data: Dict[str, Any]):
        """Process incoming tick data from the market stream"""
        try:
            tick_data = data.get('tick', {})
            if not tick_data:
                logger.warning("Received empty tick data")
                return
                
            # Extract fields
            symbol = tick_data.get('symbol')
            price = tick_data.get('quote')
            timestamp = tick_data.get('epoch')
            pip_size = tick_data.get('pip_size')
            
            if not all([symbol, price, timestamp]):
                logger.warning(f"Incomplete tick data: {tick_data}")
                return
                
            # Log the tick for debugging
            current_time = datetime.now()
            logger.debug(f"Processing tick: {symbol} @ {price} [{current_time}]")
            
            # Create raw tick object
            raw_tick = RawMarketTick(
                symbol=symbol,
                price=price,
                timestamp=timestamp,
                pip_size=pip_size,
                ask=tick_data.get('ask'),
                bid=tick_data.get('bid'),
                volume=1.0  # Set default volume since Deriv doesn't provide it
            )
            
            # Process the tick
            self._handle_market_tick(raw_tick)
            
            # Force metric update for this symbol
            with self._lock:
                if symbol in self._symbols_cache:
                    self._update_metrics_for_symbol_now(symbol)
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # The rest of the implementation remains unchanged

    def _update_metrics_for_symbol_now(self, symbol: str):
        """Immediately update metrics for a symbol after receiving a tick"""
        try:
            if symbol not in self._symbols_cache:
                logger.warning(f"Cannot update metrics for unknown symbol: {symbol}")
                return
                
            # Normalize the symbol for display and lookup
            norm_symbol = self._normalize_symbol(symbol)
            display_symbol = norm_symbol.display
            
            metrics = self._symbols_cache.get(display_symbol)
            if not metrics:
                logger.warning(f"No metrics found for symbol: {display_symbol}")
                return
                
            # Update price changes and volumes
            current_time = datetime.now()
            self._update_metrics_for_symbol(display_symbol, metrics, current_time)
            
            # Calculate volatility
            self._calculate_volatility(display_symbol, metrics)
            
            # Update directional bias
            if metrics.price_change_15m > 0.5:
                metrics.directional_bias = DirectionalBias.BULL
            elif metrics.price_change_15m < -0.5:
                metrics.directional_bias = DirectionalBias.BEAR
            else:
                metrics.directional_bias = DirectionalBias.NEUTRAL
                
            logger.debug(f"Updated metrics for {display_symbol}: price={metrics.last_price}, change_5m={metrics.price_change_5m}%")
        except Exception as e:
            logger.error(f"Error updating metrics for symbol {symbol}: {e}")
    
    def _process_ohlc(self, data: Dict[str, Any]):
        """Process incoming OHLC data from the market stream"""
        # Check for both old and new format OHLC data
        ohlc_data = data.get('ohlc', {})
        
        # If no ohlc data directly, check if it's in the new format with candles from ticks_history
        if not ohlc_data and data.get('candles') and data.get('echo_req', {}).get('ticks_history'):
            # Extract from the most recent candle
            candles = data.get('candles', [])
            if candles:
                latest_candle = candles[-1]
                symbol = data.get('echo_req', {}).get('ticks_history')
                granularity = data.get('echo_req', {}).get('granularity', 60)
                
                # Create synthetic OHLC data
                ohlc_data = {
                    'symbol': symbol,
                    'open': latest_candle.get('open'),
                    'high': latest_candle.get('high'),
                    'low': latest_candle.get('low'),
                    'close': latest_candle.get('close'),
                    'epoch': latest_candle.get('epoch'),
                    'granularity': granularity
                }
                
        if not ohlc_data:
            logger.debug(f"No OHLC data found in message: {data.keys()}")
            return
            
        # Extract fields
        symbol = ohlc_data.get('symbol')
        close = ohlc_data.get('close')
        open_price = ohlc_data.get('open')
        high = ohlc_data.get('high')
        low = ohlc_data.get('low')
        timestamp = ohlc_data.get('epoch')
        granularity = ohlc_data.get('granularity', 60)
        
        if not all([symbol, close, timestamp]):
            logger.warning(f"Incomplete OHLC data: {ohlc_data}")
            return
            
        # Extract volume if available and convert to float
        volume = float(ohlc_data.get('volume', 0))
        
        # Convert price data to float
        try:
            close = float(close)
            open_price = float(open_price) if open_price is not None else None
            high = float(high) if high is not None else None
            low = float(low) if low is not None else None
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting OHLC data to float: {e}")
            return
            
        # Create raw tick from OHLC close price
        raw_tick = RawMarketTick(
            symbol=symbol,
            price=close,
            timestamp=timestamp,
            volume=volume,
            ask=None,
            bid=None,
            pip_size=None,
            open=open_price,
            high=high,
            low=low
        )
        
        # Process the tick
        self._handle_market_tick(raw_tick)
        
        # Store in history
        with self._lock:
            norm_symbol = self._normalize_symbol(symbol).display
            if norm_symbol in self._historical_cache:
                # Convert granularity (seconds) to timeframe string
                if isinstance(granularity, (int, float)):
                    granularity_map = {
                        60: "1m",
                        300: "5m",
                        900: "15m",
                        3600: "1h",
                        14400: "4h",
                        86400: "1d"
                    }
                    # Get string representation
                    timeframe = granularity_map.get(int(granularity), "1m")
                    # Also keep the numeric representation
                    granularity_str = str(int(granularity))
                else:
                    timeframe = "1m"  # Default if granularity is not valid
                    granularity_str = "60"  # Default 60 seconds
                
                # Create OHLC data object with complete information
                ohlc_data = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'epoch': timestamp,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    'granularity': timeframe
                }
                
                # Store in both the timeframe and granularity caches if they exist
                stored = False
                
                # Try to store in the timeframe-named cache (e.g., "1m")
                if timeframe in self._historical_cache[norm_symbol]:
                    before_count = len(self._historical_cache[norm_symbol][timeframe])
                    self._historical_cache[norm_symbol][timeframe].append(ohlc_data)
                    after_count = len(self._historical_cache[norm_symbol][timeframe])
                    
                    if after_count > before_count:
                        logger.info(f"✅ OHLC stored for {symbol} [{timeframe}]. Cache count: {after_count}")
                        stored = True
                    else:
                        logger.warning(f"⚠️ OHLC not stored in {timeframe} cache for {symbol}")
                
                # Also try to store in the granularity-named cache (e.g., "60")
                if granularity_str in self._historical_cache[norm_symbol]:
                    before_count = len(self._historical_cache[norm_symbol][granularity_str])
                    self._historical_cache[norm_symbol][granularity_str].append(ohlc_data)
                    after_count = len(self._historical_cache[norm_symbol][granularity_str])
                    
                    if after_count > before_count:
                        logger.info(f"✅ OHLC stored for {symbol} [{granularity_str}]. Cache count: {after_count}")
                        stored = True
                    else:
                        logger.warning(f"⚠️ OHLC not stored in {granularity_str} cache for {symbol}")
                
                # Always add a copy to 1m cache to ensure we have enough data for analysis
                # This is to address the "minimum 20 bars required" error
                if "1m" in self._historical_cache[norm_symbol] and timeframe != "1m":
                    # Only add if we don't have at least 20 entries yet
                    if len(self._historical_cache[norm_symbol]["1m"]) < 25:
                        one_min_data = ohlc_data.copy()
                        one_min_data['granularity'] = "1m"
                        self._historical_cache[norm_symbol]["1m"].append(one_min_data)
                        logger.info(f"✅ OHLC also stored in 1m cache for {symbol}. Cache count: {len(self._historical_cache[norm_symbol]['1m'])}")
                        stored = True
                
                if not stored:
                    logger.warning(f"⚠️ OHLC received but not stored for {symbol}. No matching cache found.")
    
    def _handle_market_tick(self, raw_tick: RawMarketTick):
        """Process a normalized market tick"""
        try:
            # Normalize the tick data
            norm_tick = self._normalize_tick(raw_tick)
            display_symbol = norm_tick.symbol.display
            
            # Store in historical cache
            with self._lock:
                if display_symbol not in self._historical_cache:
                    # Initialize history for this symbol
                    self._initialize_symbol_history(display_symbol)
                    
                # Add tick to history
                self._historical_cache[display_symbol]["ticks"].append(norm_tick)
                
                # Update or create symbol metrics
                if display_symbol in self._symbols_cache:
                    # Update existing metrics
                    metrics = self._symbols_cache[display_symbol]
                    metrics.last_price = raw_tick.price
                    metrics.last_updated = norm_tick.timestamp_dt
                    metrics.status = self._determine_status(display_symbol)
                
                else:
                    # Create new metrics for this symbol with default values
                    self._symbols_cache[display_symbol] = SymbolMetrics(
                        symbol=norm_tick.symbol,
                        last_price=raw_tick.price,
                        last_updated=norm_tick.timestamp_dt,
                        status="neutral",  # Initial status is neutral
                        price_change_1m=0.0,
                        price_change_5m=0.0,
                        price_change_15m=0.0,
                        price_change_1h=0.0,
                        volume_1m=1.0,
                        volume_5m=1.0,
                        volume_15m=1.0,
                        volatility=0.0,
                        directional_bias=DirectionalBias.NEUTRAL,
                        sentiment_score=0.0
                    )
                    logger.info(f"Created new symbol metrics for {display_symbol}")
                
                # Calculate price changes based on historical data
                self._calculate_price_changes(display_symbol)
            
            # Trigger a snapshot creation if this is a new symbol or periodically
            if display_symbol not in self._symbols_cache or len(self._historical_cache[display_symbol]["ticks"]) % 50 == 0:
                # Create a snapshot for new symbols or every 50 ticks
                self._create_snapshot()
        except Exception as e:
            logger.error(f"Error handling market tick for {raw_tick.symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    def _run_aggregation_loop(self):
        """Background thread that updates metrics on regular intervals"""
        while self._running:
            try:
                self._update_all_metrics()
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
            
            # Sleep for 1 second before next update
            time.sleep(1)
    
    def _run_snapshot_loop(self):
        """Background thread that creates periodic snapshots"""
        while self._running:
            try:
                # Create a snapshot every 5 minutes
                self._create_snapshot()
                
                # Generate AI commentary every 5 minutes
                self._generate_ai_commentary()
                
                # Update top setups
                self._calculate_top_setups()
            except Exception as e:
                logger.error(f"Error in snapshot loop: {e}")
            
            # Sleep for 5 minutes before next snapshot
            for _ in range(300):  # 5 minutes in seconds
                if not self._running:
                    break
                time.sleep(1)
    
    def _normalize_tick(self, raw_tick: RawMarketTick) -> NormalizedMarketTick:
        """Convert raw tick to normalized format"""
        # Normalize symbol
        norm_symbol = self._normalize_symbol(raw_tick.symbol)
        
        # Convert timestamp to datetime
        if isinstance(raw_tick.timestamp, (int, float)):
            timestamp_dt = datetime.fromtimestamp(raw_tick.timestamp)
        else:
            # If already datetime or None, use as is
            timestamp_dt = raw_tick.timestamp or datetime.now()
        
        # Create normalized tick
        return NormalizedMarketTick(
            symbol=norm_symbol,
            price=raw_tick.price,
            timestamp=raw_tick.timestamp,
            timestamp_dt=timestamp_dt,
            volume=raw_tick.volume,
            ask=raw_tick.ask,
            bid=raw_tick.bid,
            spread=raw_tick.ask - raw_tick.bid if raw_tick.ask and raw_tick.bid else None,
            source="derived"  # Set default source
        )
        
    def _initialize_symbol_history(self, symbol: str):
        """Initialize history for a new symbol"""
        self._historical_cache[symbol] = {
            "ticks": deque(maxlen=self._history_limits.get("ticks", 1000)),
            "1m": deque(maxlen=self._history_limits.get("1m", 120)),
            "5m": deque(maxlen=self._history_limits.get("5m", 120)),
            "15m": deque(maxlen=self._history_limits.get("15m", 120)),
            "1h": deque(maxlen=self._history_limits.get("1h", 168)),
            "4h": deque(maxlen=self._history_limits.get("4h", 60)),
            "1d": deque(maxlen=self._history_limits.get("1d", 60)),
            # Also add some aliases used in the code
            "60": deque(maxlen=self._history_limits.get("1m", 120)),    # 1 minute
            "300": deque(maxlen=self._history_limits.get("5m", 120)),   # 5 minutes
            "900": deque(maxlen=self._history_limits.get("15m", 120)),  # 15 minutes
            "3600": deque(maxlen=self._history_limits.get("1h", 168)),  # 1 hour
            "14400": deque(maxlen=self._history_limits.get("4h", 60)),  # 4 hours
            "86400": deque(maxlen=self._history_limits.get("1d", 60))   # 1 day
        }
        logger.info(f"Initialized history cache for symbol: {symbol}")
        
    def _calculate_price_changes(self, symbol: str):
        """Calculate price changes for different time frames based on tick data"""
        try:
            if symbol not in self._historical_cache or symbol not in self._symbols_cache:
                logger.warning(f"Cannot calculate price changes for unknown symbol: {symbol}")
                return
                
            ticks = self._historical_cache[symbol]["ticks"]
            if not ticks:
                logger.debug(f"No tick data available for {symbol}")
                return
                
            current_price = ticks[-1].price
            current_time = ticks[-1].timestamp_dt
            
            # Calculate price changes for each timeframe
            for timeframe, seconds in self._timeframes.items():
                # Find the oldest tick within the timeframe
                reference_time = current_time - timedelta(seconds=seconds)
                reference_price = current_price  # Default to current if no historical data
                
                # Find the closest tick to the reference time
                for tick in ticks:
                    if tick.timestamp_dt >= reference_time:
                        reference_price = tick.price
                        break
                
                # Calculate the price change percentage
                if reference_price > 0:
                    price_change = ((current_price - reference_price) / reference_price) * 100
                else:
                    price_change = 0.0
                    
                # Update the metrics with the calculated change
                if timeframe == "1m":
                    self._symbols_cache[symbol].price_change_1m = price_change
                elif timeframe == "5m":
                    self._symbols_cache[symbol].price_change_5m = price_change
                elif timeframe == "15m":
                    self._symbols_cache[symbol].price_change_15m = price_change
                elif timeframe == "1h":
                    self._symbols_cache[symbol].price_change_1h = price_change
            
            # Calculate directional bias based on price changes
            metrics = self._symbols_cache[symbol]
            self._calculate_volatility(symbol, metrics)
            
        except Exception as e:
            logger.error(f"Error calculating price changes for {symbol}: {e}")
    
    def _determine_status(self, symbol: str) -> str:
        """Determine the status of a symbol based on recent price changes"""
        if symbol not in self._symbols_cache:
            return "neutral"
        
        metrics = self._symbols_cache[symbol]
        
        # Determine status based on short-term price change
        if metrics.price_change_5m > 0.2:
            return "up"
        elif metrics.price_change_5m < -0.2:
            return "down"
        else:
            return "neutral"
    
    def _update_all_metrics(self):
        """Update metrics for all symbols"""
        current_time = datetime.now()
        
        with self._lock:
            for symbol, metrics in list(self._symbols_cache.items()):
                try:
                    # Skip if last update was too recent (within 1 second)
                    if metrics.last_updated and (current_time - metrics.last_updated).total_seconds() < 1:
                        continue
                        
                    # Update metrics for this symbol
                    self._update_metrics_for_symbol(symbol, metrics, current_time)
                except Exception as e:
                    logger.error(f"Error updating metrics for {symbol}: {e}")
    
    def _update_metrics_for_symbol(self, symbol: str, metrics: SymbolMetrics, current_time: datetime):
        """Update metrics for a single symbol"""
        # Check if we have history for this symbol
        if symbol not in self._historical_cache:
            return
            
        # Get ticks for this symbol
        ticks = self._historical_cache[symbol].get("ticks", [])
        if not ticks:
            return
            
        # Calculate price changes
        for timeframe, seconds in self._timeframes.items():
            reference_time = current_time - timedelta(seconds=seconds)
            
            # Find the closest tick to the reference time
            reference_price = None
            for tick in ticks:
                if tick.timestamp_dt >= reference_time:
                    if isinstance(tick, dict):
                        reference_price = tick['price']
                    else:
                        reference_price = tick.price
                    break
            
            if reference_price is None:
                # No data for this timeframe
                continue
                
            # Get current price
            if ticks and len(ticks) > 0:
                if isinstance(ticks[-1], dict):
                    current_price = ticks[-1]['price']
                else:
                    current_price = ticks[-1].price
            else:
                current_price = metrics.last_price
            
            # Calculate price change
            if reference_price > 0:
                price_change_pct = ((current_price - reference_price) / reference_price) * 100
            else:
                price_change_pct = 0.0
            
            # Update the appropriate metric
            if timeframe == "1m":
                metrics.price_change_1m = price_change_pct
            elif timeframe == "5m":
                metrics.price_change_5m = price_change_pct
            elif timeframe == "15m":
                metrics.price_change_15m = price_change_pct
            elif timeframe == "1h":
                metrics.price_change_1h = price_change_pct
        
        # Calculate volatility (standard deviation of price changes over last 20 ticks)
        if len(ticks) > 5:
            recent_ticks = list(ticks)[-20:]
            price_changes = []
            
            last_price = None
            for tick in recent_ticks:
                if isinstance(tick, dict):
                    price = tick['price']
                else:
                    price = tick.price
                    
                if last_price is not None:
                    price_changes.append((price - last_price) / last_price * 100)
                last_price = price
                
            # Now calculate the volatility
            self._calculate_volatility_from_changes(metrics, price_changes)
    
    def _calculate_volatility(self, symbol: str, metrics: SymbolMetrics):
        """Calculate volatility for a symbol based on recent ticks"""
        if symbol not in self._historical_cache:
            return
            
        ticks = self._historical_cache[symbol].get("ticks", [])
        if len(ticks) < 5:
            return  # Not enough data
            
        # Get last 20 ticks (or all if fewer)
        recent_ticks = list(ticks)[-20:]
        price_changes = []
        
        last_price = None
        for tick in recent_ticks:
            if hasattr(tick, 'price'):
                price = tick.price
                if last_price is not None:
                    try:
                        price_change = (price - last_price) / last_price * 100
                        price_changes.append(price_change)
                    except (ZeroDivisionError, TypeError):
                        pass  # Skip invalid calculations
                last_price = price
        
        self._calculate_volatility_from_changes(metrics, price_changes)
    
    def _calculate_volatility_from_changes(self, metrics: SymbolMetrics, price_changes: List[float]):
        """Calculate volatility from a list of price changes"""
        if not price_changes:
            return
            
        try:
            # Calculate standard deviation of price changes
            std_dev = statistics.stdev(price_changes) if len(price_changes) > 1 else 0
            metrics.volatility = std_dev
            
            # Determine directional bias based on recent price changes
            avg_change = sum(price_changes) / len(price_changes) if price_changes else 0
            if avg_change > 0.2:
                metrics.directional_bias = DirectionalBias.BULL
            elif avg_change < -0.2:
                metrics.directional_bias = DirectionalBias.BEAR
            else:
                metrics.directional_bias = DirectionalBias.NEUTRAL
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
                
            if price_changes:
                volatility = statistics.stdev(price_changes) if len(price_changes) > 1 else 0
                metrics.volatility = volatility
        
        # Determine directional bias based on recent price changes
        if metrics.price_change_15m > 0.5:
            metrics.directional_bias = DirectionalBias.BULL
        elif metrics.price_change_15m < -0.5:
            metrics.directional_bias = DirectionalBias.BEAR
        else:
            metrics.directional_bias = DirectionalBias.NEUTRAL
        
        # Calculate synthetic sentiment score (-1.0 to 1.0)
        # This is a simple heuristic - in a real system this might come from external data
        price_signals = [
            metrics.price_change_1m / 2,  # Half weight to 1m change
            metrics.price_change_5m,
            metrics.price_change_15m * 1.5,  # 50% more weight to 15m change
        ]
        
        # Average the signals and normalize to -1 to 1 range
        avg_signal = sum(price_signals) / len(price_signals)
        sentiment = max(min(avg_signal / 5.0, 1.0), -1.0)  # Cap at -1.0 to 1.0
        metrics.sentiment_score = sentiment
    
    def _create_snapshot(self) -> MarketSnapshot:
        """Create a snapshot of the current market state"""
        with self._lock:
            # Check if we have any symbols
            if not self._symbols_cache:
                logger.warning("Attempting to create snapshot with no symbols available")
                # Create empty snapshot
                return MarketSnapshot(
                    timestamp=datetime.now(),
                    symbols={},
                    top_gainers=[],
                    top_losers=[],
                    high_volatility=[]
                )
                
            # Copy current metrics
            snapshot = MarketSnapshot(
                timestamp=datetime.now(),
                symbols=dict(self._symbols_cache),  # Make a copy
                top_gainers=[],
                top_losers=[],
                high_volatility=[]
            )
            
            # Find top gainers, losers, and high volatility symbols
            symbols_list = list(self._symbols_cache.values())
            
            # Sort by 15-min price change for gainers/losers
            sorted_by_change = sorted(symbols_list, key=lambda x: x.price_change_15m, reverse=True)
            
            # Top 5 gainers - get just the symbol names
            snapshot.top_gainers = [s.symbol.display for s in sorted_by_change[:5]]
            
            # Top 5 losers - get just the symbol names
            snapshot.top_losers = [s.symbol.display for s in sorted_by_change[-5:]] if len(sorted_by_change) >= 5 else []
            
            # Sort by volatility
            sorted_by_volatility = sorted(symbols_list, key=lambda x: x.volatility, reverse=True)
            
            # Top 5 high volatility - get just the symbol names
            snapshot.high_volatility = [s.symbol.display for s in sorted_by_volatility[:5]]
            
            # Store the snapshot
            self._snapshots.append(snapshot)
            
            # Limit the number of stored snapshots
            while len(self._snapshots) > self._snapshots_capacity:
                self._snapshots.pop(0)  # Remove oldest
            
            return snapshot

    def _generate_ai_commentary(self):
        """Generate AI commentary for current market conditions"""
        # This is a placeholder for actual AI commentary generation
        # In a real system, this would call an LLM or other analysis system
        
        try:
            # Get the latest snapshot
            if not self._snapshots:
                return None
                
            latest_snapshot = self._snapshots[-1]
            timestamp = latest_snapshot.timestamp
            
            # Generate simple commentary based on top movers
            insights = []
            
            # Add insights based on top gainers
            if latest_snapshot.top_gainers:
                top_gainer = latest_snapshot.top_gainers[0]
                insights.append(f"{top_gainer.symbol.asset_name} leading with {top_gainer.price_change_15m:.2f}% gain in the last 15 minutes.")
            
            # Add insights based on top losers
            if latest_snapshot.top_losers:
                top_loser = latest_snapshot.top_losers[-1]  # Last one is the biggest loser
                insights.append(f"{top_loser.symbol.asset_name} showing weakness with {top_loser.price_change_15m:.2f}% drop in the last 15 minutes.")
            
            # Add volatility insight
            if latest_snapshot.high_volatility:
                vol_symbol = latest_snapshot.high_volatility[0]
                insights.append(f"Watch for increased volatility in {vol_symbol.symbol.asset_name} (volatility: {vol_symbol.volatility:.2f}).")
            
            # Build the commentary object
            commentary = AICommentaryData(
                timestamp=timestamp,
                insights=insights,
                market_summary="Market showing mixed signals with some volatility.",
                key_levels={},
                sentiment_overview="Neutral market sentiment with pockets of strength."
            )
            
            # Store the commentary
            self._last_commentary = commentary
            
            return commentary
            
        except Exception as e:
            logger.error(f"Error generating AI commentary: {e}")
            return None

    def _calculate_top_setups(self) -> List[TradingSetup]:
        """Calculate potential trading setups based on market conditions"""
        # This is a placeholder for actual setup detection logic
        try:
            if not self._symbols_cache:
                return []
                
            setups = []
            current_time = datetime.now()
            
            # Look for potential setups in all symbols
            for symbol, metrics in self._symbols_cache.items():
                # Example logic for bullish setup
                if metrics.directional_bias == DirectionalBias.BULL and metrics.price_change_15m > 1.0:
                    # Simple bullish continuation setup
                    setups.append(TradingSetup(
                        symbol=metrics.symbol,
                        setup_type=SetupType.BULLISH_CONTINUATION,
                        entry_price=metrics.last_price,
                        stop_loss=metrics.last_price * 0.99,  # 1% stop loss
                        take_profit=metrics.last_price * 1.02,  # 2% take profit
                        timeframe="15m",
                        confidence_score=min(metrics.price_change_15m / 2, 0.9),  # Max 0.9
                        detection_time=current_time,
                        expiration_time=current_time + timedelta(hours=1)
                    ))
                
                # Example logic for bearish setup
                elif metrics.directional_bias == DirectionalBias.BEAR and metrics.price_change_15m < -1.0:
                    # Simple bearish continuation setup
                    setups.append(TradingSetup(
                        symbol=metrics.symbol,
                        setup_type=SetupType.BEARISH_CONTINUATION,
                        entry_price=metrics.last_price,
                        stop_loss=metrics.last_price * 1.01,  # 1% stop loss
                        take_profit=metrics.last_price * 0.98,  # 2% take profit
                        timeframe="15m",
                        confidence_score=min(abs(metrics.price_change_15m) / 2, 0.9),  # Max 0.9
                        detection_time=current_time,
                        expiration_time=current_time + timedelta(hours=1)
                    ))
                
                # Example logic for volatility breakout
                elif metrics.volatility > 0.5:
                    # High volatility breakout setup
                    setups.append(TradingSetup(
                        symbol=metrics.symbol,
                        setup_type=SetupType.VOLATILITY_BREAKOUT,
                        entry_price=metrics.last_price,
                        stop_loss=metrics.last_price * (1 - metrics.volatility/100),
                        take_profit=metrics.last_price * (1 + metrics.volatility/50),
                        timeframe="5m",
                        confidence_score=min(metrics.volatility / 3, 0.85),  # Max 0.85
                        detection_time=current_time,
                        expiration_time=current_time + timedelta(minutes=30)
                    ))
            
            # Sort by confidence score
            setups.sort(key=lambda x: x.confidence_score, reverse=True)
            
            # Keep top 5
            top_setups = setups[:5]
            
            # Store latest setups
            self._last_top_setups = top_setups
            
            return top_setups
            
        except Exception as e:
            logger.error(f"Error calculating top setups: {e}")
            return []
    
    # === API Methods ===
    
    def get_latest_snapshot(self) -> MarketSnapshot:
        """Get the latest market snapshot"""
        with self._lock:
            if not self._snapshots:
                # Create a new snapshot if none exists
                return self._create_snapshot()
            return self._snapshots[-1]
    
    def get_symbol_metrics(self, symbol: str) -> Optional[SymbolMetrics]:
        """Get metrics for a specific symbol"""
        with self._lock:
            # Try exact match first
            if symbol in self._symbols_cache:
                return self._symbols_cache[symbol]
                
            # Try normalized symbol format
            norm_symbol = self._normalize_symbol(symbol)
            display_symbol = norm_symbol.display
            
            if display_symbol in self._symbols_cache:
                return self._symbols_cache[display_symbol]
                
            return None
    
    def get_all_symbols(self) -> List[str]:
        """Get list of all available symbols"""
        with self._lock:
            return list(self._symbols_cache.keys())
    
    def get_historical_snapshots(self, limit: int = 12) -> List[MarketSnapshot]:
        """Get historical snapshots, limited to the specified number"""
        with self._lock:
            limit = min(limit, len(self._snapshots))
            return list(self._snapshots)[-limit:]
    
    def get_historical_ticks(self, symbol: str, limit: int = 100) -> List[NormalizedMarketTick]:
        """Get historical ticks for a symbol"""
        with self._lock:
            # Normalize the symbol
            norm_symbol = self._normalize_symbol(symbol).display
            
            if norm_symbol not in self._historical_cache:
                return []
                
            ticks = self._historical_cache[norm_symbol].get("ticks", [])
            limit = min(limit, len(ticks))
            return list(ticks)[-limit:]
    
    def get_historical_ohlc(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> List[Dict]:
        """Get historical OHLC data for a symbol"""
        with self._lock:
            # Normalize the symbol
            norm_symbol = self._normalize_symbol(symbol).display
            
            if norm_symbol not in self._historical_cache:
                return []
                
            if timeframe not in self._historical_cache[norm_symbol]:
                return []
                
            ohlc_data = self._historical_cache[norm_symbol][timeframe]
            limit = min(limit, len(ohlc_data))
            return list(ohlc_data)[-limit:]
    
    def get_ai_commentary(self) -> Optional[AICommentaryData]:
        """Get the latest AI commentary"""
        return self._last_commentary
    
    def get_trading_setups(self) -> List[TradingSetup]:
        """Get current trading setups"""
        return self._last_top_setups

    def get_worker_status(self) -> Dict[str, Any]:
        """Get status of the worker thread"""
        if not self._worker_processor:
            return {
                "running": False,
                "status": "not initialized"
            }
            
        return self._worker_processor.get_status()

    def generate_demo_market_data(self) -> bool:
        """Generate demo market data for testing"""
        try:
            # Demo symbols
            demo_symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"]
            
            # Current time
            current_time = datetime.now()
            
            # Generate data for each symbol
            for symbol_str in demo_symbols:
                # Normalize symbol
                norm_symbol = self._normalize_symbol(symbol_str)
                
                # Generate a price based on symbol
                base = norm_symbol.base
                if base == "BTC":
                    price = 40000.0 + (hash(str(current_time) + base) % 2000)
                elif base == "ETH":
                    price = 3000.0 + (hash(str(current_time) + base) % 300)
                elif base == "SOL":
                    price = 150.0 + (hash(str(current_time) + base) % 20)
                elif base == "DOGE":
                    price = 0.25 + (hash(str(current_time) + base) % 100) / 1000.0
                else:
                    # Random price between 1 and 100
                    price = 1.0 + (hash(str(current_time) + base) % 100)
                
                # Create metrics for this symbol
                self._symbols_cache[symbol_str] = SymbolMetrics(
                    symbol=norm_symbol,
                    last_price=price,
                    last_updated=current_time,
                    price_change_1m=((hash(str(current_time) + base + "1m") % 100) - 50) / 10.0,
                    price_change_5m=((hash(str(current_time) + base + "5m") % 100) - 50) / 10.0,
                    price_change_15m=((hash(str(current_time) + base + "15m") % 100) - 50) / 5.0,
                    price_change_1h=((hash(str(current_time) + base + "1h") % 100) - 50) / 3.0,
                    volume_1m=(hash(str(current_time) + base + "v1m") % 100) / 10.0,
                    volume_5m=(hash(str(current_time) + base + "v5m") % 100) / 5.0,
                    volume_15m=(hash(str(current_time) + base + "v15m") % 100) / 2.0,
                    volatility=(hash(str(current_time) + base + "vol") % 50) / 100.0,
                    status="neutral",
                    directional_bias=DirectionalBias.NEUTRAL,
                    sentiment_score=((hash(str(current_time) + base + "sent") % 200) - 100) / 100.0
                )
                
                # Set directional bias based on price changes
                metrics = self._symbols_cache[symbol_str]
                if metrics.price_change_15m > 0.5:
                    metrics.directional_bias = DirectionalBias.BULL
                    metrics.status = "up"
                elif metrics.price_change_15m < -0.5:
                    metrics.directional_bias = DirectionalBias.BEAR
                    metrics.status = "down"
            
            # Create snapshot, commentary and setups
            self._create_snapshot()
            self._generate_ai_commentary()
            self._calculate_top_setups()
            
            logger.info(f"Generated demo data for {len(demo_symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error generating demo market data: {e}")
            return False


# Singleton instance
_aggregator_instance = None

def get_aggregator_instance(market_stream: Optional[MarketStream] = None) -> MarketDataAggregator:
    """Get or create the global aggregator instance"""
    global _aggregator_instance
    if _aggregator_instance is None:
        _aggregator_instance = MarketDataAggregator(market_stream=market_stream)
    return _aggregator_instance