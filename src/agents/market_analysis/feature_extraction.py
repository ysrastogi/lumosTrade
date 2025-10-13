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
        if len(df) < 20:
            return {"error": "Insufficient data for feature extraction (minimum 20 bars required)"}

        features = {}
        features['close_price'] = df['close'].iloc[-1]
        features['open_price'] = df['open'].iloc[-1]
        features['high_price'] = df['high'].iloc[-1]
        features['low_price'] = df['low'].iloc[-1]
        features['price_change'] = df['close'].iloc[-1] - df['close'].iloc[-2]
        features['price_change_pct'] = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
        
        features['last_5_returns'] = df['close'].pct_change().iloc[-5:].tolist()
        features['last_5_change'] = ((df['close'].iloc[-1] / df['close'].iloc[-6]) - 1) * 100 if len(df) >= 6 else None

        features['volatility'] = df['close'].pct_change().std()
        features['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14).iloc[-1]
        features['daily_range'] = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1] * 100

        features['rsi'] = ta.momentum.rsi(df['close'], window=14).iloc[-1]
        features['macd'] = ta.trend.macd_diff(df['close']).iloc[-1]
        features['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close']).iloc[-1]
        features['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close']).iloc[-1]

        features['sma_20'] = ta.trend.sma_indicator(df['close'], window=20).iloc[-1]
        features['sma_50'] = ta.trend.sma_indicator(df['close'], window=50).iloc[-1] if len(df) >= 50 else None
        features['sma_200'] = ta.trend.sma_indicator(df['close'], window=200).iloc[-1] if len(df) >= 200 else None
        features['ema_12'] = ta.trend.ema_indicator(df['close'], window=12).iloc[-1]
        features['ema_26'] = ta.trend.ema_indicator(df['close'], window=26).iloc[-1]

        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        features['bb_upper'] = bb.bollinger_hband().iloc[-1]
        features['bb_lower'] = bb.bollinger_lband().iloc[-1]
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / df['close'].iloc[-1]
        features['bb_percent'] = (df['close'].iloc[-1] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

        if 'volume' in df.columns and df['volume'].sum() > 0:
            features['volume'] = df['volume'].iloc[-1]
            features['obv'] = ta.volume.on_balance_volume(df['close'], df['volume']).iloc[-1]
            features['volume_sma'] = df['volume'].rolling(window=20).mean().iloc[-1]
            features['volume_change'] = df['volume'].iloc[-1] / features['volume_sma'] if features['volume_sma'] > 0 else 0

        features['price_to_sma_20'] = (df['close'].iloc[-1] / features['sma_20'] - 1) * 100
        features['price_to_bb_upper'] = (features['close_price'] - features['bb_upper']) / features['bb_upper'] * 100
        features['price_to_bb_lower'] = (features['close_price'] - features['bb_lower']) / features['bb_lower'] * 100
        
        features['adx'] = ta.trend.adx(df['high'], df['low'], df['close']).iloc[-1]
        features['slope_5'] = FeatureExtractor._calculate_slope(df['close'], 5)

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