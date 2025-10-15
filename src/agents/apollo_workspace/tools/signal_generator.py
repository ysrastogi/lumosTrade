from datetime import datetime
from typing import List
from src.agents.apollo_workspace.models import Signal

class SignalGenerator:
    """
    Multi-stage signal generation with quality filtering
    """
    
    def __init__(self, min_confidence: float = 0.65):
        self.min_confidence = min_confidence
        self.signal_counter = 0
        
    def generate_signals(self, market_data: dict) -> List[Signal]:
        """
        Primary signal generation pipeline
        
        Stages:
        1. Pattern validation
        2. Regime compatibility check
        3. Confluence scoring
        4. Risk management validation
        5. Quality assessment
        """
        signals = []
        
        for trade_idea in market_data.get('trade_ideas', []):
            # Stage 1: Validate pattern integrity
            if not self._validate_pattern(trade_idea, market_data):
                continue
                
            # Stage 2: Regime compatibility
            regime_score = self._check_regime_compatibility(
                trade_idea['pattern'], 
                market_data['regime']
            )
            
            # Stage 3: Calculate confluence
            confluence_score = self._calculate_confluence(
                trade_idea, 
                market_data['patterns'],
                market_data['features']
            )
            
            # Stage 4: Risk validation
            if not self._validate_risk_params(trade_idea):
                continue
            
            # Stage 5: Adjust confidence
            adjusted_confidence = self._adjust_confidence(
                trade_idea['confidence'],
                regime_score,
                confluence_score,
                market_data['regime_confidence']
            )
            
            if adjusted_confidence >= self.min_confidence:
                signal = self._create_signal(
                    trade_idea, 
                    adjusted_confidence,
                    regime_score,
                    confluence_score
                )
                signals.append(signal)
                
        return signals
    
    def _validate_pattern(self, trade_idea: dict, market_data: dict) -> bool:
        """Ensures pattern metrics are logical"""
        # Check for price anomalies
        if trade_idea['entry'] == 0:
            return False
        
        # Validate stop loss distance
        sl_distance = abs(trade_idea['entry'] - trade_idea['stop_loss'])
        atr = market_data['features'].get('atr', 0)
        
        if atr > 0 and sl_distance < 0.5 * atr:
            return False  # Stop too tight
        
        if atr > 0 and sl_distance > 3 * atr:
            return False  # Stop too wide
            
        return True
    
    def _check_regime_compatibility(self, pattern: str, regime: str) -> float:
        """
        Score pattern-regime alignment
        
        Compatibility Matrix:
        - Oversold/Overbought: Best in ranging markets
        - Breakouts: Best in trending markets
        - Mean reversion: Best in low volatility
        """
        compatibility_map = {
            'oversold_rebound': {
                'low_volatility_ranging_tight_range': 0.9,
                'moderate_volatility_ranging': 0.8,
                'high_volatility_trending': 0.4,
            },
            'mean_reversion': {
                'low_volatility_ranging_tight_range': 0.95,
                'moderate_volatility_ranging': 0.7,
                'high_volatility_trending': 0.3,
            },
            'breakout': {
                'high_volatility_trending': 0.9,
                'moderate_volatility_ranging': 0.6,
                'low_volatility_ranging_tight_range': 0.3,
            }
        }
        
        return compatibility_map.get(pattern, {}).get(regime, 0.5)
    
    def _calculate_confluence(self, trade_idea: dict, 
                             patterns: List[dict], 
                             features: dict) -> float:
        """
        Multi-factor confluence scoring
        
        Factors:
        - Number of supporting patterns
        - Technical indicator alignment
        - Price action confirmation
        """
        score = 0.0
        
        # Pattern confluence
        same_bias_patterns = [p for p in patterns 
                             if p['bias'] == trade_idea['direction']]
        score += min(len(same_bias_patterns) * 0.15, 0.3)
        
        # RSI confluence
        rsi = features.get('rsi', 50)
        if trade_idea['direction'] == 'buy' and rsi < 30:
            score += 0.2
        elif trade_idea['direction'] == 'sell' and rsi > 70:
            score += 0.2
        
        # Stochastic confluence
        stoch_k = features.get('stoch_k', 0.5)
        if trade_idea['direction'] == 'buy' and stoch_k < 0.2:
            score += 0.15
        elif trade_idea['direction'] == 'sell' and stoch_k > 0.8:
            score += 0.15
        
        # Price vs MA confluence
        close = features.get('close', 0)
        ema_20 = features.get('ema_20', close)
        
        if close > 0 and ema_20 > 0:
            if trade_idea['direction'] == 'buy' and close < ema_20:
                score += 0.1  # Buying dip
            elif trade_idea['direction'] == 'sell' and close > ema_20:
                score += 0.1  # Selling rally
        
        return min(score, 1.0)
    
    def _validate_risk_params(self, trade_idea: dict) -> bool:
        """Validate risk/reward and position sizing"""
        # Minimum R:R ratio
        if trade_idea['risk_reward'] < 1.5:
            return False
        
        # Maximum R:R (unrealistic targets)
        if trade_idea['risk_reward'] > 5:
            return False
            
        return True
    
    def _adjust_confidence(self, base_confidence: float,
                          regime_score: float,
                          confluence_score: float,
                          regime_confidence: float) -> float:
        """
        Bayesian-inspired confidence adjustment
        """
        # Convert to probabilities
        base_prob = base_confidence / 100.0
        
        # Weighted adjustment
        adjusted = (
            base_prob * 0.5 +           # Base pattern confidence
            regime_score * 0.2 +         # Regime compatibility
            confluence_score * 0.2 +     # Multi-factor confluence
            regime_confidence * 0.1      # Regime detection certainty
        )
        
        return adjusted * 100
    
    def _create_signal(self, trade_idea: dict, 
                      confidence: float,
                      regime_score: float,
                      confluence_score: float) -> Signal:
        """Package signal with full context"""
        self.signal_counter += 1
        
        return Signal(
            id=f"SIG_{trade_idea['symbol']}_{self.signal_counter}",
            timestamp=datetime.fromisoformat(trade_idea['timestamp']),
            symbol=trade_idea['symbol'],
            pattern=trade_idea['pattern'],
            direction=trade_idea['direction'],
            confidence=confidence,
            entry=trade_idea['entry'],
            stop_loss=trade_idea['stop_loss'],
            target=trade_idea['target'],
            risk_reward=trade_idea['risk_reward'],
            reasoning="",  # Filled by ReasoningEngine
            invalidation_criteria=[],  # Filled by ReasoningEngine
            supporting_factors=[],  # Filled by ConfluenceAnalyzer
            similar_historical_count=0,  # Filled by BacktestValidator
            historical_win_rate=0.0,  # Filled by BacktestValidator
        )