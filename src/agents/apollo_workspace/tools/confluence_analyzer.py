from typing import List, Tuple
from src.agents.apollo_workspace.models import Signal

class ConfluenceAnalyzer:
    """
    Identifies and quantifies supporting/conflicting factors
    """
    
    def analyze(self, signal: Signal, market_data: dict) -> Tuple[List[str], List[str]]:
        """
        Returns: (supporting_factors, risk_factors)
        """
        supporting = []
        risks = []
        
        features = market_data['features']
        patterns = market_data['patterns']
        regime = market_data['regime']
        
        # Technical indicator analysis
        supporting.extend(self._analyze_indicators(signal, features))
        
        # Pattern confluence
        supporting.extend(self._analyze_patterns(signal, patterns))
        
        # Regime analysis
        regime_factors, regime_risks = self._analyze_regime(signal, regime, features)
        supporting.extend(regime_factors)
        risks.extend(regime_risks)
        
        # Price structure
        supporting.extend(self._analyze_price_structure(signal, features))
        
        # Volume analysis
        supporting.extend(self._analyze_volume(signal, features))
        
        # Risk factors
        risks.extend(self._identify_risks(signal, features, market_data))
        
        return supporting, risks
    
    def _analyze_indicators(self, signal: Signal, features: dict) -> List[str]:
        factors = []
        
        rsi = features.get('rsi', 50)
        stoch_k = features.get('stoch_k', 0.5)
        bb_width = features.get('bb_width', 1)
        
        if signal.direction == 'buy':
            if rsi < 30:
                factors.append(f"RSI deeply oversold at {rsi:.1f} (extreme)")
            elif rsi < 40:
                factors.append(f"RSI oversold at {rsi:.1f}")
            
            if stoch_k < 0.2:
                factors.append(f"Stochastic oversold at {stoch_k*100:.1f}%")
            
            if bb_width > 2:
                factors.append("High volatility environment - larger moves possible")
        
        else:  # sell
            if rsi > 70:
                factors.append(f"RSI overbought at {rsi:.1f}")
            
            if stoch_k > 0.8:
                factors.append(f"Stochastic overbought at {stoch_k*100:.1f}%")
        
        # MACD
        macd = features.get('macd')
        macd_signal = features.get('macd_signal')
        
        if macd is not None and macd_signal is not None:
            if signal.direction == 'buy' and macd > macd_signal:
                factors.append("MACD bullish crossover confirmed")
            elif signal.direction == 'sell' and macd < macd_signal:
                factors.append("MACD bearish crossover confirmed")
        
        return factors
    
    def _analyze_patterns(self, signal: Signal, patterns: List[dict]) -> List[str]:
        factors = []
        
        aligned_patterns = [p for p in patterns if p['bias'] == signal.direction]
        
        if len(aligned_patterns) >= 2:
            pattern_names = [p['type'] for p in aligned_patterns]
            factors.append(f"Multiple confirming patterns: {', '.join(pattern_names)}")
        
        # High confidence patterns
        high_conf = [p for p in aligned_patterns if p['confidence'] >= 85]
        if high_conf:
            factors.append(f"{len(high_conf)} high-confidence pattern(s) aligned")
        
        return factors
    
    def _analyze_regime(self, signal: Signal, regime: str, 
                       features: dict) -> Tuple[List[str], List[str]]:
        supporting = []
        risks = []
        
        if 'ranging' in regime:
            if signal.pattern in ['mean_reversion', 'oversold_rebound', 'overbought_reversal']:
                supporting.append(f"Pattern optimal for {regime} regime")
            else:
                risks.append(f"Breakout pattern in ranging market - lower success rate")
        
        if 'trending' in regime:
            if signal.pattern in ['breakout', 'trend_continuation']:
                supporting.append(f"Pattern optimal for {regime} regime")
            else:
                risks.append(f"Counter-trend setup in strong trend")
        
        if 'low_volatility' in regime:
            atr = features.get('atr', 0)
            if atr > 0:
                supporting.append(f"Low volatility regime - controlled risk environment")
        
        return supporting, risks
    
    def _analyze_price_structure(self, signal: Signal, features: dict) -> List[str]:
        factors = []
        
        close = features.get('close', 0)
        bb_high = features.get('bb_high', 0)
        bb_low = features.get('bb_low', 0)
        bb_mid = features.get('bb_mid', 0)
        
        if close > 0 and bb_high > bb_low:
            bb_position = (close - bb_low) / (bb_high - bb_low)
            
            if signal.direction == 'buy' and bb_position < 0.2:
                factors.append(f"Price at lower Bollinger Band ({bb_position*100:.0f}% position)")
            elif signal.direction == 'sell' and bb_position > 0.8:
                factors.append(f"Price at upper Bollinger Band ({bb_position*100:.0f}% position)")
        
        # Distance from moving averages
        sma_20 = features.get('sma_20', 0)
        if close > 0 and sma_20 > 0:
            distance_pct = ((close - sma_20) / sma_20) * 100
            
            if signal.direction == 'buy' and distance_pct < -2:
                factors.append(f"Price {abs(distance_pct):.1f}% below 20 SMA")
            elif signal.direction == 'sell' and distance_pct > 2:
                factors.append(f"Price {distance_pct:.1f}% above 20 SMA")
        
        return factors
    
    def _analyze_volume(self, signal: Signal, features: dict) -> List[str]:
        factors = []
        
        volume = features.get('volume', 0)
        volume_sma = features.get('volume_sma', 0)
        
        if volume > 0 and volume_sma > 0:
            vol_ratio = volume / volume_sma
            
            if vol_ratio > 1.5:
                factors.append(f"Above-average volume ({vol_ratio:.1f}x) confirms interest")
            elif vol_ratio < 0.5:
                factors.append(f"Low volume ({vol_ratio:.1f}x) - wait for confirmation")
        
        return factors
    
    def _identify_risks(self, signal: Signal, features: dict, 
                       market_data: dict) -> List[str]:
        risks = []
        
        # Check for conflicting patterns
        patterns = market_data['patterns']
        opposing = [p for p in patterns if p['bias'] != signal.direction and p['confidence'] > 70]
        
        if opposing:
            risks.append(f"{len(opposing)} opposing pattern(s) detected")
        
        # ADX check for trend strength
        adx = features.get('adx', 25)
        if adx < 20 and 'breakout' in signal.pattern:
            risks.append("Weak trend (ADX < 20) - breakout may fail")
        
        # Extreme volatility
        bb_width = features.get('bb_width', 1)
        if bb_width > 3:
            risks.append("Extreme volatility - wider stops recommended")
        
        # Price change momentum
        price_change_5 = features.get('price_change_5', 0)
        if abs(price_change_5) > 10:
            risks.append("Recent strong move - potential exhaustion")
        
        return risks