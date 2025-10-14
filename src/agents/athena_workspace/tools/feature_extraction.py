import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
import logging
import ta


class FeatureExtractor:
    
    @staticmethod
    def _convert_numpy_types(obj: Any) -> Any:
        """
        Convert numpy types to Python native types for JSON serialization
        
        Args:
            obj: Any object that might contain numpy types
            
        Returns:
            Object with numpy types converted to Python native types
        """
        if isinstance(obj, dict):
            return {k: FeatureExtractor._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [FeatureExtractor._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    @staticmethod
    def extract_features(df: pd.DataFrame) -> Dict:
        """Extract technical features from OHLC dataframe."""
        if len(df) < 20:  # Not enough data for reliable features
            # Return a simplified set of features that don't require longer lookback periods
            return FeatureExtractor._extract_minimal_features(df)
            
        features = {}
        
        # Basic price information
        features['close'] = df['close'].iloc[-1]
        features['open'] = df['open'].iloc[-1]
        features['high'] = df['high'].iloc[-1]
        features['low'] = df['low'].iloc[-1]
        
        # Calculate simple metrics
        features['daily_range'] = features['high'] - features['low']
        features['daily_change'] = features['close'] - features['open']
        features['daily_change_pct'] = (features['daily_change'] / features['open']) * 100
        
        # Simple Moving Averages
        features['sma_5'] = ta.trend.sma_indicator(df['close'], window=5).iloc[-1]
        features['sma_10'] = ta.trend.sma_indicator(df['close'], window=10).iloc[-1]
        features['sma_20'] = ta.trend.sma_indicator(df['close'], window=20).iloc[-1]
        
        # Exponential Moving Averages
        features['ema_5'] = ta.trend.ema_indicator(df['close'], window=5).iloc[-1]
        features['ema_10'] = ta.trend.ema_indicator(df['close'], window=10).iloc[-1]
        features['ema_20'] = ta.trend.ema_indicator(df['close'], window=20).iloc[-1]
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        features['bb_high'] = bb.bollinger_hband().iloc[-1]
        features['bb_mid'] = bb.bollinger_mavg().iloc[-1]
        features['bb_low'] = bb.bollinger_lband().iloc[-1]
        features['bb_width'] = (features['bb_high'] - features['bb_low']) / features['bb_mid']
        
        # RSI
        features['rsi'] = ta.momentum.rsi(df['close']).iloc[-1]
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        features['macd'] = macd.macd().iloc[-1]
        features['macd_signal'] = macd.macd_signal().iloc[-1]
        features['macd_diff'] = macd.macd_diff().iloc[-1]
        
        # ADX - Only calculate if we have enough data
        if len(df) > 20:  # ADX typically uses 14 periods
            try:
                features['adx'] = ta.trend.adx(df['high'], df['low'], df['close']).iloc[-1]
            except Exception:
                features['adx'] = 25.0  # Default to neutral ADX if calculation fails
        else:
            features['adx'] = 25.0  # Default to neutral ADX
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        features['stoch_k'] = stoch.stoch().iloc[-1]
        features['stoch_d'] = stoch.stoch_signal().iloc[-1]
        
        # ATR - Average True Range (volatility)
        features['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close']).iloc[-1]
        
        # Volume (if available)
        if 'volume' in df.columns:
            features['volume'] = df['volume'].iloc[-1]
            features['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=5).iloc[-1]
        else:
            features['volume'] = 0
            features['volume_sma'] = 0
        
        # Recent price action
        features['price_change_1'] = df['close'].iloc[-1] - df['close'].iloc[-2]
        if len(df) > 5:
            features['price_change_5'] = df['close'].iloc[-1] - df['close'].iloc[-6]
        else:
            features['price_change_5'] = 0
            
        features = FeatureExtractor._convert_numpy_types(features)

        return features
        
    @staticmethod
    def _extract_minimal_features(df: pd.DataFrame) -> Dict:
        """Extract a minimal set of features when we have limited data."""
        features = {}
        
        # Basic price information
        features['close'] = df['close'].iloc[-1]
        features['open'] = df['open'].iloc[-1]
        features['high'] = df['high'].iloc[-1]
        features['low'] = df['low'].iloc[-1]
        
        # Calculate simple metrics that don't require long lookback periods
        features['daily_range'] = features['high'] - features['low']
        features['daily_change'] = features['close'] - features['open']
        features['daily_change_pct'] = (features['daily_change'] / features['open']) * 100
        
        # Simple Moving Averages - limited by data length
        max_window = min(5, len(df)-1)
        if max_window > 1:
            features['sma_5'] = ta.trend.sma_indicator(df['close'], window=max_window).iloc[-1]
        else:
            features['sma_5'] = features['close']
            
        features['sma_10'] = features['close']  # Not enough data
        features['sma_20'] = features['close']  # Not enough data
        
        # Exponential Moving Averages - limited by data length
        if max_window > 1:
            features['ema_5'] = ta.trend.ema_indicator(df['close'], window=max_window).iloc[-1]
        else:
            features['ema_5'] = features['close']
            
        features['ema_10'] = features['close']  # Not enough data
        features['ema_20'] = features['close']  # Not enough data
        
        # Defaults for other technical indicators
        features['bb_high'] = features['high']
        features['bb_mid'] = features['close']
        features['bb_low'] = features['low']
        features['bb_width'] = (features['high'] - features['low']) / features['close']
        
        # Placeholder values for other indicators
        features['rsi'] = 50.0  # Neutral
        features['macd'] = 0.0  # Neutral
        features['macd_signal'] = 0.0  # Neutral
        features['macd_diff'] = 0.0  # Neutral
        features['adx'] = 25.0  # Neutral
        features['stoch_k'] = 50.0  # Neutral
        features['stoch_d'] = 50.0  # Neutral
        features['atr'] = features['daily_range']  # Approximation
        
        # Volume (if available)
        if 'volume' in df.columns:
            features['volume'] = df['volume'].iloc[-1]
            features['volume_sma'] = features['volume']
        else:
            features['volume'] = 0
            features['volume_sma'] = 0
        
        # Recent price action - limited by data length
        if len(df) > 1:
            features['price_change_1'] = df['close'].iloc[-1] - df['close'].iloc[-2]
        else:
            features['price_change_1'] = 0
        features['price_change_5'] = 0  # Not enough data
        
        features = FeatureExtractor._convert_numpy_types(features)

        return features

    @staticmethod
    def _calculate_slope(series: pd.Series, window: int = 5) -> float:
        """Calculate the slope of the recent price movement"""
        if len(series) < window:
            return 0
            
        y = series.iloc[-window:].values
        x = np.array(range(window))
    
        x_mean = x.mean()
        y_mean = y.mean()
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0
            
        return numerator / denominator