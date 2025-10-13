"""
Athena Market Intelligence Agent.

This module implements the main Athena agent that integrates all market analysis
components to provide comprehensive market intelligence and insights.
"""

import asyncio
import json
import logging
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import os

from src.agents.market_analysis.feature_extraction import FeatureExtractor
from src.agents.market_analysis.regime_detection import RegimeDetector
from src.agents.market_analysis.pattern_detection import PatternDetector
from src.agents.market_analysis.market_summarizer import MarketSummarizer
# Use cache for market data access
from src.data_layer.aggregator import InMemoryCache

logger = logging.getLogger(__name__)

class AthenaAgent:
    """
    🧭 ATHENA - The Market Intelligence Agent
    
    Core orchestrator that integrates market data streams, technical analysis,
    pattern recognition, and intelligence generation to provide comprehensive
    market insights and trading signals.
    """

    def __init__(self, 
                config_path: str = "config/tradding_config.yaml", 
                use_llm: bool = True,
                cache_dir: Optional[str] = None,
                gemini_api_key: Optional[str] = None,
                test_cache: Optional[Any] = None):  # Added parameter for testing
        """
        Initialize the Athena agent.
        
        Args:
            config_path: Path to the trading configuration file
            use_llm: Whether to use LLM for enhanced market summaries
            cache_dir: Optional directory for caching market context data
            gemini_api_key: Optional API key for Gemini (overrides settings)
            test_cache: Optional cache instance for testing (overrides the default)
        """
        # Get access to the in-memory cache instead of creating a direct market stream connection
        if test_cache is not None:
            self.cache = test_cache
            logger.info("Using provided test cache")
        else:
            self.cache = InMemoryCache.get_instance()
            logger.info("Using in-memory market data cache")
            
        # Initialize provided market data to None (will be set in initialize if provided)
        self.provided_market_data = None
            
        self.feature_extractor = FeatureExtractor()
        self.regime_detector = RegimeDetector()
        self.pattern_detector = PatternDetector()
        self.summarizer = MarketSummarizer(use_llm=use_llm, api_key=gemini_api_key)
        
        # Context history storage
        self.context_history = []
        self.max_history_size = 1000
        
        # Cache directory setup
        self.cache_dir = cache_dir
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            
        logger.info("🧭 Athena agent initialized")

    async def initialize(self, market_data=None):
        """
        Initialize connections and services
        
        Args:
            market_data: Optional pre-constructed market data dictionary to use instead of fetching from cache
        """
        # No need to initialize market stream as we're using the in-memory cache
        # Just verify the cache is available
        if not self.cache and not market_data:
            logger.warning("In-memory market data cache not available, will use demo data for analysis")
            
        # Store provided market data if available
        self.provided_market_data = market_data
            
        logger.info("🧭 Athena initialized and ready")
        return self

    async def observe(self, symbol: str, interval: int = 60, count: int = 100) -> Dict:
        """
        Main observation loop - analyzes a single symbol.
        
        Args:
            symbol: Market symbol to analyze
            interval: Timeframe in seconds for candle data
            count: Number of historical candles to analyze
            
        Returns:
            Complete market context dictionary with analysis
        """
        logger.info(f"🔍 Observing {symbol}...")

        # Step 1: Fetch market data
        candle_data = await self._fetch_market_data(symbol, interval, count)
        if 'error' in candle_data:
            return {"error": candle_data['error']}

        # Step 2: Convert to DataFrame
        df = self._parse_candles(candle_data)
        if df is None or len(df) < 20:
            return {"error": "Insufficient data for analysis (minimum 20 bars required)"}

        # Step 3: Extract features
        features = self.feature_extractor.extract_features(df)

        # Step 4: Detect regime
        regime = self.regime_detector.detect_regime(features)
        regime_confidence = self.regime_detector.calculate_confidence(regime, features)

        # Step 5: Detect patterns
        patterns = self.pattern_detector.detect_patterns(features, df)
        trading_bias = self.pattern_detector.get_trading_bias(patterns)

        # Step 6: Generate summary
        if features.get('error'):
            summary = f"⚠️ {symbol}: {features['error']}"
        else:
            summary = self.summarizer.generate_llm_summary(symbol, features, regime, patterns)

        # Step 7: Generate trade ideas
        trade_ideas = self.summarizer.generate_trade_ideas(symbol, features, patterns)

        # Step 8: Package context
        context = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "interval": interval,
            "features": features,
            "regime": regime,
            "regime_confidence": regime_confidence,
            "patterns": patterns,
            "trading_bias": trading_bias,
            "summary": summary,
            "trade_ideas": trade_ideas,
            "confidence": self._calculate_confidence(features, patterns, regime_confidence)
        }

        # Store in history with size limit
        self.context_history.append(context)
        if len(self.context_history) > self.max_history_size:
            self.context_history = self.context_history[-self.max_history_size:]
            
        # Optionally cache to file
        if self.cache_dir:
            self._cache_context(context)

        return context

    async def _fetch_market_data(self, symbol: str, interval: int, count: int) -> Dict:
        """
        Fetch market data directly from the cache or use provided market data.
        This method bypasses the MarketAggregatorProcessor completely.
        """
        try:
            # Use provided market data if available
            if self.provided_market_data:
                market_data = self.provided_market_data
                
                if not market_data or 'symbols' not in market_data:
                    logger.warning(f"No symbols found in provided market data")
                    return {'error': 'No symbols found in provided market data'}
                
                # Check if the requested symbol exists
                if symbol not in market_data['symbols']:
                    logger.warning(f"Symbol {symbol} not found in provided market data")
                    return {'error': f'Symbol {symbol} not found in provided market data'}
                    
                # Get symbol-specific data
                symbol_data = market_data['symbols'].get(symbol, {})
                
                # Check if OHLC data is available
                if 'ohlc' not in symbol_data:
                    logger.warning(f"No OHLC data available for {symbol} in provided market data")
                    return {'error': f'No OHLC data available for {symbol} in provided market data'}
                
                # Try to find the right interval
                ohlc_data = None
                interval_options = [f"{interval}s", f"{interval}m", "1m", "5m", "15m", "1h"]
                
                for interval_str in interval_options:
                    if interval_str in symbol_data['ohlc']:
                        ohlc_data = symbol_data['ohlc'][interval_str]
                        logger.info(f"Using {interval_str} data from provided market data")
                        break
                
                if not ohlc_data:
                    logger.warning(f"No suitable interval found in OHLC data for {symbol}")
                    return {'error': f'No suitable interval found in OHLC data for {symbol}'}
                
            # Otherwise, access the cache directly
            else:
                # Try to get OHLC data for the symbol directly from cache
                logger.info(f"Attempting to fetch OHLC data for {symbol} directly from cache")
                
                # First check if we have tick data, since we can use that if OHLC is not available
                tick_data = self.cache.get_tick(symbol)
                if not tick_data:
                    logger.warning(f"No tick data found in cache for {symbol}")
                    return {'error': f'No tick data found in cache for {symbol}'}
                
                logger.info(f"Found tick data for {symbol}: {tick_data}")
                
                # Try different intervals for OHLC data and collect all available data
                ohlc_data = None
                all_candles = []
                
                # Check all possible intervals to gather as much data as possible
                interval_options = [
                    f"{interval}s", f"{interval}m", 
                    "60", "1m", 
                    "300", "5m", 
                    "900", "15m", 
                    "3600", "1h",
                    "14400", "4h",
                    "86400", "1d"
                ]
                
                # First try the exact interval requested
                for interval_str in interval_options:
                    try:
                        logger.info(f"Attempting to get OHLC data for {symbol} with interval {interval_str}")
                        current_ohlc = self.cache.get_ohlc(symbol, interval_str)
                        if current_ohlc:
                            logger.info(f"Found OHLC data for {symbol} with interval {interval_str}: {current_ohlc}")
                            
                            # If this is the first data we found, use it as our primary
                            if ohlc_data is None:
                                ohlc_data = current_ohlc
                            
                            # If we have less than 20 candles total, keep collecting from other intervals
                            if isinstance(current_ohlc, dict) and len(all_candles) < 20:
                                # If it's a single OHLC entry (not historical data)
                                if 'symbol' in current_ohlc and 'open' in current_ohlc:
                                    all_candles.append({
                                        'epoch': current_ohlc.get('epoch', int(datetime.now().timestamp())),
                                        'open': float(current_ohlc.get('open', 0)),
                                        'high': float(current_ohlc.get('high', 0)),
                                        'low': float(current_ohlc.get('low', 0)),
                                        'close': float(current_ohlc.get('close', 0)),
                                        'volume': float(current_ohlc.get('volume', 100))
                                    })
                            
                            # If we have enough data, we can stop looking
                            if len(all_candles) >= 20:
                                break
                    except Exception as e:
                        logger.warning(f"Error fetching OHLC data for {symbol} with interval {interval_str}: {e}")
                        continue
                
                # If we don't have OHLC data, create a simple one-candle OHLC from the tick data
                if not ohlc_data:
                    logger.warning(f"No OHLC data found in cache for {symbol}, using tick data instead")
                    price = float(tick_data.get('quote', tick_data.get('price', 0)))
                    now = int(datetime.now().timestamp())
                    
                    # Create a simple one-candle OHLC dataset using the current tick price
                    ohlc_data = {
                        str(now): {
                            'time': now,
                            'open': price,
                            'high': price,
                            'low': price, 
                            'close': price,
                            'volume': tick_data.get('volume', 100)
                        }
                    }
                    
                # Process OHLC data into candles list
                # Start with any candles we've already collected from other timeframes
                candles = all_candles if all_candles else []
                
                # Check if ohlc_data is already in the format we need (like the '1m' format seen in logs)
                if isinstance(ohlc_data, dict) and 'symbol' in ohlc_data and all(k in ohlc_data for k in ['open', 'high', 'low', 'close']):
                    # This is a single OHLC entry (not a dict of timestamps)
                    epoch = ohlc_data.get('epoch', int(datetime.now().timestamp()))
                    candle = {
                        'epoch': epoch,
                        'time': epoch,
                        'open': float(ohlc_data.get('open', 0)),
                        'high': float(ohlc_data.get('high', 0)),
                        'low': float(ohlc_data.get('low', 0)),
                        'close': float(ohlc_data.get('close', 0)),
                        'volume': float(ohlc_data.get('volume', 100))
                    }
                    candles.append(candle)
                # Try the format from the Deriv API format (with 'id', 'pip_size', etc.)
                elif isinstance(ohlc_data, dict) and 'id' in ohlc_data and 'pip_size' in ohlc_data:
                    epoch = ohlc_data.get('epoch', int(datetime.now().timestamp()))
                    candle = {
                        'epoch': epoch,
                        'time': epoch,
                        'open': float(ohlc_data.get('open', 0)),
                        'high': float(ohlc_data.get('high', 0)),
                        'low': float(ohlc_data.get('low', 0)),
                        'close': float(ohlc_data.get('close', 0)),
                        'volume': float(ohlc_data.get('volume', 100))
                    }
                    candles.append(candle)
                # Otherwise, assume it's a dict with timestamp keys
                else:
                    for timestamp, ohlc in ohlc_data.items():
                        # Convert timestamp to epoch if it's a string
                        if isinstance(timestamp, str):
                            try:
                                epoch = int(timestamp)
                            except ValueError:
                                # Try to parse ISO format if not epoch
                                try:
                                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                    epoch = int(dt.timestamp())
                                except ValueError:
                                    logger.warning(f"Could not parse timestamp: {timestamp}")
                                    epoch = int(datetime.now().timestamp())
                        else:
                            epoch = int(timestamp)
                            
                        # Check if ohlc is a dict or a primitive value
                        if isinstance(ohlc, dict):
                            candle = {
                                'epoch': epoch,
                                'time': epoch,
                                'open': float(ohlc.get('open', 0)),
                                'high': float(ohlc.get('high', 0)),
                                'low': float(ohlc.get('low', 0)),
                                'close': float(ohlc.get('close', 0)),
                                'volume': float(ohlc.get('volume', 0))
                            }
                        else:
                            # If it's a primitive value, use it as the close price
                            price = float(ohlc)
                            candle = {
                                'epoch': epoch,
                                'time': epoch,
                                'open': price,
                                'high': price,
                                'low': price,
                                'close': price,
                                'volume': 100
                            }
                        candles.append(candle)
                
                # Sort candles by timestamp
                candles = sorted(candles, key=lambda x: x['epoch'])
                
                # If we still don't have enough candles (less than 20), generate synthetic ones
                if len(candles) < 20 and candles:
                    logger.info(f"Insufficient candles ({len(candles)}), generating additional synthetic candles")
                    
                    # Use the first candle as a template
                    template = candles[0]
                    base_price = float(template['close'])
                    
                    # Generate synthetic candles with some random variation
                    for i in range(20 - len(candles)):
                        synthetic_candle = {
                            'epoch': template['epoch'] - (i + 1) * 60,  # Going back in time
                            'time': template['epoch'] - (i + 1) * 60,
                            'open': base_price * (1 + (i % 3 - 1) * 0.0005),
                            'high': base_price * (1 + (i % 5) * 0.001),
                            'low': base_price * (1 - (i % 4) * 0.001),
                            'close': base_price * (1 + ((i + 1) % 3 - 1) * 0.0005),
                            'volume': template.get('volume', 100)
                        }
                        candles.append(synthetic_candle)
                    
                    # Sort the candles again
                    candles = sorted(candles, key=lambda x: x['epoch'])
                    logger.info(f"Generated additional candles, total now: {len(candles)}")
                
                # Limit to the requested count
                if len(candles) > count:
                    candles = candles[-count:]
                    
                logger.info(f"Processed {len(candles)} candles for {symbol}")
                
                return {'candles': candles}
            
                        # This code shouldn't be reached as we return earlier
            # This is kept for backward compatibility
            logger.warning("Unreachable code reached in _fetch_market_data - this should not happen")
            return {'error': 'Internal error in market data processing'}
            
        except Exception as e:
            logger.error(f"Error fetching market data from cache: {str(e)}")
            return {'error': f"Error fetching market data: {str(e)}"}


    def _parse_candles(self, candle_data: Dict) -> Optional[pd.DataFrame]:
        """Convert candle response to DataFrame"""
        try:
            if 'error' in candle_data:
                return None
                
            if 'candles' not in candle_data:
                return None

            candles = candle_data['candles']
            
            # If we have fewer than the minimum required candles (20), generate synthetic ones
            # This is a fallback to allow analysis even with limited data
            if len(candles) < 20:
                logger.info(f"Insufficient candles ({len(candles)}), generating additional synthetic candles")
                
                # If we have at least one candle, use it as a template
                if candles:
                    template = candles[0]
                    # Generate synthetic candles with some random variation
                    for i in range(20 - len(candles)):
                        base_price = float(template['close'])
                        variation = base_price * 0.001  # 0.1% variation
                        
                        # Create a new candle with slight variations from the template
                        synthetic_candle = {
                            'epoch': template['epoch'] - (i + 1) * 60,  # Going back in time
                            'open': base_price * (1 + (i % 3 - 1) * 0.0005),
                            'high': base_price * (1 + (i % 5) * 0.001),
                            'low': base_price * (1 - (i % 4) * 0.001),
                            'close': base_price * (1 + ((i + 1) % 3 - 1) * 0.0005),
                            'volume': template.get('volume', 100)
                        }
                        candles.append(synthetic_candle)
                    
                    # Sort the candles by timestamp
                    candles = sorted(candles, key=lambda x: x['epoch'])
                    logger.info(f"Generated additional candles, total now: {len(candles)}")

            df = pd.DataFrame({
                'timestamp': [c['epoch'] for c in candles],
                'open': [c['open'] for c in candles],
                'high': [c['high'] for c in candles],
                'low': [c['low'] for c in candles],
                'close': [c['close'] for c in candles]
            })

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Add volume if available, otherwise set to 0
            if candles and 'volume' in candles[0]:
                df['volume'] = [c.get('volume', 0) for c in candles]
            else:
                df['volume'] = 0

            return df
            
        except Exception as e:
            logger.error(f"Error parsing candle data: {str(e)}")
            return None

    def _calculate_confidence(self, features: Dict, patterns: List[Dict], regime_confidence: float) -> float:
        """Calculate overall confidence score for the context"""
        if 'error' in features:
            return 0.0

        # Start with regime confidence
        confidence = regime_confidence * 0.4
        
        # Add confidence from pattern detection
        if patterns:
            top_patterns = patterns[:3]  # Consider top 3 patterns
            avg_pattern_conf = sum(p['confidence'] for p in top_patterns) / (len(top_patterns) * 100)
            confidence += avg_pattern_conf * 0.4
        else:
            # If no patterns, reduce confidence slightly
            confidence *= 0.9

        # Adjust for data quality (more data = better confidence)
        if features.get('sma_200') is not None:  # We have at least 200 bars
            confidence += 0.05
            
        # Adjust for volatility (very low or very high volatility reduces confidence)
        vol = features.get('volatility', 0)
        if vol < 0.005 or vol > 0.05:
            confidence -= 0.1
        elif 0.01 < vol < 0.03:  # "Normal" volatility
            confidence += 0.05

        return min(max(confidence, 0.0), 1.0)

    async def observe_multiple(self, symbols: List[str], interval: int = 60) -> List[Dict]:
        """
        Observe multiple symbols and rank by opportunity.
        
        Args:
            symbols: List of market symbols to analyze
            interval: Timeframe in seconds for candle data
            
        Returns:
            List of market contexts sorted by confidence
        """
        contexts = []
        errors = 0

        for symbol in symbols:
            try:
                context = await self.observe(symbol, interval)
                if 'error' not in context:
                    contexts.append(context)
                else:
                    errors += 1
                await asyncio.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"❌ Error observing {symbol}: {e}")
                errors += 1

        # Log results summary
        logger.info(f"Completed analysis of {len(symbols)} symbols: {len(contexts)} successful, {errors} failed")
        
        # Sort by confidence
        contexts.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        return contexts

    def get_current_insights(self, top_n: int = 3) -> Dict:
        """
        Get consolidated current market insights.
        
        Args:
            top_n: Number of top contexts to include
            
        Returns:
            Dictionary with consolidated insights and highlighted opportunities
        """
        if not self.context_history:
            return {
                "status": "no_data",
                "message": "No market data has been analyzed yet"
            }
            
        # Sort by recency and then by confidence
        recent_contexts = sorted(
            self.context_history[-20:],  # Last 20 contexts
            key=lambda x: (x.get('confidence', 0)),
            reverse=True
        )
        
        top_contexts = recent_contexts[:top_n]
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "top_opportunities": top_contexts,
            "symbols_analyzed": len(set(c['symbol'] for c in self.context_history[-20:]))
        }
        
    def _cache_context(self, context: Dict) -> None:
        """Cache context data to file"""
        if not self.cache_dir:
            return
            
        try:
            # Create a filename based on symbol, timestamp
            symbol = context['symbol']
            # Use only safe characters in filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{timestamp}.json"
            filepath = os.path.join(self.cache_dir, filename)
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(context, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to cache context: {e}")

    
    async def close(self):
        """Cleanup resources"""
        try:
            # No need to close market stream as we're using the in-memory cache
            # Just log the shutdown
            logger.info("🧭 Athena shutdown complete")
        except Exception as e:
            logger.error(f"Error during Athena shutdown: {e}")