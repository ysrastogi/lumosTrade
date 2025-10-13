from typing import Dict, List, Optional, Any
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class PatternDetector:
    
    @staticmethod
    def _convert_numpy_types(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: PatternDetector._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [PatternDetector._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    @staticmethod
    def detect_patterns(features: Dict, df: pd.DataFrame) -> List[Dict]:
        """
        Detect trading patterns and setups from technical indicators.
        
        Args:
            features: Dictionary of technical indicators from FeatureExtractor
            df: DataFrame with price data (for patterns needing historical context)
            
        Returns:
            List of detected patterns with confidence scores and descriptions
        """
        patterns = []

        if 'error' in features:
            logger.warning(f"Unable to detect patterns: {features['error']}")
            return patterns

        # Extract key metrics for pattern detection
        rsi = features.get('rsi', 50)
        bb_width = features.get('bb_width', 0)
        price_to_upper = features.get('price_to_bb_upper', 0)
        price_to_lower = features.get('price_to_bb_lower', 0)
        macd = features.get('macd', 0)
        stoch_k = features.get('stoch_k', 50)
        stoch_d = features.get('stoch_d', 50)
        adx = features.get('adx', 20)
        close_price = features.get('close_price', 0)

        # 1. Oversold Rebound Setup
        if rsi < 30:
            strength = min((30 - rsi) / 30, 1.0)
            patterns.append({
                "type": "oversold_rebound",
                "confidence": min(strength * 100, 95),
                "description": f"RSI oversold at {rsi:.1f} - potential bounce setup",
                "bias": "bullish",
                "key_metrics": {
                    "rsi": rsi,
                    "stoch_k": stoch_k
                }
            })

        # 2. Overbought Reversal Setup
        if rsi > 70:
            strength = min((rsi - 70) / 30, 1.0)
            patterns.append({
                "type": "overbought_reversal",
                "confidence": min(strength * 100, 95),
                "description": f"RSI overbought at {rsi:.1f} - potential reversal setup",
                "bias": "bearish",
                "key_metrics": {
                    "rsi": rsi,
                    "stoch_k": stoch_k
                }
            })

        # 3. Bollinger Squeeze Breakout
        if bb_width < 0.02:
            patterns.append({
                "type": "volatility_squeeze",
                "confidence": min((0.02 - bb_width) / 0.02 * 100, 90),
                "description": "Bollinger squeeze - breakout imminent",
                "bias": "neutral",
                "key_metrics": {
                    "bb_width": bb_width
                }
            })

        # 4. Bollinger Band Bounce
        if -2 < price_to_lower < 0:
            bounce_strength = min(abs(price_to_lower) / 2, 1.0)
            patterns.append({
                "type": "bb_lower_bounce",
                "confidence": bounce_strength * 75,
                "description": f"Price near lower Bollinger Band - bounce potential ({abs(price_to_lower):.1f}% away)",
                "bias": "bullish",
                "key_metrics": {
                    "price_to_lower": price_to_lower,
                    "bb_width": bb_width
                }
            })

        # 5. Bollinger Band Rejection
        if 0 < price_to_upper < 2:
            rejection_strength = min(abs(price_to_upper) / 2, 1.0)
            patterns.append({
                "type": "bb_upper_rejection",
                "confidence": rejection_strength * 75,
                "description": f"Price near upper Bollinger Band - rejection potential ({abs(price_to_upper):.1f}% away)",
                "bias": "bearish",
                "key_metrics": {
                    "price_to_upper": price_to_upper,
                    "bb_width": bb_width
                }
            })

        # 6. MACD Momentum Shift
        if macd > 0 and df['close'].iloc[-1] > df['close'].iloc[-5]:
            momentum_strength = min(macd * 10, 1.0) if macd > 0 else 0.5
            patterns.append({
                "type": "macd_bullish_momentum",
                "confidence": 55 + momentum_strength * 20,
                "description": "MACD positive with upward price action - bullish momentum",
                "bias": "bullish",
                "key_metrics": {
                    "macd": macd,
                    "price_change": features.get('price_change_pct', 0)
                }
            })

        if macd < 0 and df['close'].iloc[-1] < df['close'].iloc[-5]:
            momentum_strength = min(abs(macd) * 10, 1.0) if macd < 0 else 0.5
            patterns.append({
                "type": "macd_bearish_momentum",
                "confidence": 55 + momentum_strength * 20,
                "description": "MACD negative with downward price action - bearish momentum",
                "bias": "bearish",
                "key_metrics": {
                    "macd": macd,
                    "price_change": features.get('price_change_pct', 0)
                }
            })

        # 7. Mean Reversion Setup
        if 'sma_20' in features:
            sma_20 = features['sma_20']
            distance_from_mean = abs((close_price - sma_20) / sma_20) * 100
            if distance_from_mean > 2:
                reversion_bias = "bearish" if close_price > sma_20 else "bullish"
                patterns.append({
                    "type": "mean_reversion",
                    "confidence": min(distance_from_mean * 10, 80),
                    "description": f"Price extended {distance_from_mean:.1f}% from mean - potential reversion",
                    "bias": reversion_bias,
                    "key_metrics": {
                        "distance_from_mean": distance_from_mean,
                        "volatility": features.get('volatility', 0)
                    }
                })

        # 8. Divergences (if RSI available)
        if len(df) > 14 and 'rsi' in features:
            price_higher_high = df['close'].iloc[-1] > df['close'].iloc[-5] and df['close'].iloc[-5] > df['close'].iloc[-10]
            price_lower_low = df['close'].iloc[-1] < df['close'].iloc[-5] and df['close'].iloc[-5] < df['close'].iloc[-10]
            
            try:
                # Calculate RSI values for comparison periods
                rsi_now = features['rsi']
                import ta
                rsi_5_bars_ago = ta.momentum.rsi(df['close'].shift(5), window=14).iloc[-1]
                rsi_10_bars_ago = ta.momentum.rsi(df['close'].shift(10), window=14).iloc[-1]
                
                # Bearish divergence
                if price_higher_high and rsi_now < rsi_5_bars_ago:
                    patterns.append({
                        "type": "bearish_divergence",
                        "confidence": 70,
                        "description": "Bearish RSI divergence detected - price making higher highs while RSI shows weakness",
                        "bias": "bearish",
                        "key_metrics": {
                            "rsi": rsi,
                            "rsi_5_ago": rsi_5_bars_ago
                        }
                    })
                    
                # Bullish divergence
                if price_lower_low and rsi_now > rsi_5_bars_ago:
                    patterns.append({
                        "type": "bullish_divergence",
                        "confidence": 70,
                        "description": "Bullish RSI divergence detected - price making lower lows while RSI shows strength",
                        "bias": "bullish",
                        "key_metrics": {
                            "rsi": rsi,
                            "rsi_5_ago": rsi_5_bars_ago
                        }
                    })
            except Exception as e:
                # Skip divergence calculation if there's an error
                logger.debug(f"Could not calculate divergences: {str(e)}")
                pass

        # Sort patterns by confidence
        patterns.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Convert numpy types to Python native types
        patterns = PatternDetector._convert_numpy_types(patterns)
        
        return patterns

    @staticmethod
    def get_trading_bias(patterns: List[Dict]) -> Dict:
        """
        Determine overall trading bias from detected patterns.
        
        Args:
            patterns: List of detected patterns from detect_patterns()
            
        Returns:
            Dictionary with overall bias assessment
        """
        if not patterns:
            return {
                "bias": "neutral",
                "confidence": 0,
                "description": "No clear patterns detected"
            }
            
        # Count patterns by bias
        bullish_count = sum(1 for p in patterns if p['bias'] == 'bullish')
        bearish_count = sum(1 for p in patterns if p['bias'] == 'bearish')
        
        # Weighted confidence calculation
        bullish_confidence = sum(p['confidence'] for p in patterns if p['bias'] == 'bullish')
        bearish_confidence = sum(p['confidence'] for p in patterns if p['bias'] == 'bearish')
        
        # Determine bias
        if bullish_confidence > bearish_confidence:
            bias = "bullish"
            confidence = min(bullish_confidence / 100, 0.95)
            strength = "strong" if bullish_confidence > bearish_confidence * 2 else "moderate"
            description = f"{strength.capitalize()} bullish bias ({bullish_count} bullish patterns vs {bearish_count} bearish)"
        elif bearish_confidence > bullish_confidence:
            bias = "bearish"
            confidence = min(bearish_confidence / 100, 0.95)
            strength = "strong" if bearish_confidence > bullish_confidence * 2 else "moderate"
            description = f"{strength.capitalize()} bearish bias ({bearish_count} bearish patterns vs {bullish_count} bullish)"
        else:
            bias = "neutral"
            confidence = 0.5
            description = "Mixed signals - no clear directional bias"
            
        result = {
            "bias": bias,
            "confidence": confidence,
            "description": description
        }
        
        # Convert any numpy types to Python native types
        return PatternDetector._convert_numpy_types(result)