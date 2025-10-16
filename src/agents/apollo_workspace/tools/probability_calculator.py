from typing import List, Dict, Tuple
from src.agents.apollo_workspace.models import Signal
import numpy as np

class ProbabilityCalculator:
    """
    Statistical analysis of signal probability
    """
    
    def __init__(self, memory_system):
        self.memory = memory_system
        
    def calculate_probabilities(self, signal: Signal, 
                               market_data: dict) -> Dict[str, float]:
        """
        Returns comprehensive probability metrics
        """
        # Find similar historical patterns
        similar_signals = self.memory.find_similar_patterns(
            pattern=signal.pattern,
            symbol=signal.symbol,
            regime=market_data['regime'],
            direction=signal.direction,
            limit=100
        )
        
        if not similar_signals:
            return self._default_probabilities(signal)
        
        # Calculate base win rate
        win_rate = self._calculate_win_rate(similar_signals)
        
        # Adjust for confidence level
        confidence_adjusted_wr = self._adjust_for_confidence(
            win_rate, 
            signal.confidence,
            similar_signals
        )
        
        # Calculate expected value
        avg_win, avg_loss = self._calculate_avg_outcomes(similar_signals)
        expected_value = (confidence_adjusted_wr * avg_win) - \
                        ((1 - confidence_adjusted_wr) * avg_loss)
        
        # Calculate profit factor
        total_wins = sum(s.pnl for s in similar_signals if s.pnl and s.pnl > 0)
        total_losses = abs(sum(s.pnl for s in similar_signals if s.pnl and s.pnl < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Regime-specific win rate
        regime_wr = self._calculate_regime_win_rate(
            similar_signals,
            market_data['regime']
        )
        
        return {
            'base_win_rate': win_rate,
            'adjusted_win_rate': confidence_adjusted_wr,
            'expected_value': expected_value,
            'profit_factor': profit_factor,
            'regime_win_rate': regime_wr,
            'sample_size': len(similar_signals),
            'avg_win_r': avg_win,
            'avg_loss_r': avg_loss,
        }
    
    def _calculate_win_rate(self, signals: List[Signal]) -> float:
        """Calculate historical win rate"""
        completed = [s for s in signals if s.outcome in ['win', 'loss']]
        if not completed:
            return 0.5
        
        wins = len([s for s in completed if s.outcome == 'win'])
        return wins / len(completed)
    
    def _adjust_for_confidence(self, base_wr: float, 
                              confidence: float,
                              signals: List[Signal]) -> float:
        """
        Adjust win rate based on signal confidence
        
        Higher confidence signals historically perform better
        """
        # Segment by confidence levels
        high_conf = [s for s in signals if s.confidence >= 80]
        med_conf = [s for s in signals if 65 <= s.confidence < 80]
        low_conf = [s for s in signals if s.confidence < 65]
        
        if confidence >= 80 and high_conf:
            return self._calculate_win_rate(high_conf)
        elif 65 <= confidence < 80 and med_conf:
            return self._calculate_win_rate(med_conf)
        elif low_conf:
            return self._calculate_win_rate(low_conf)
        
        return base_wr
    
    def _calculate_avg_outcomes(self, signals: List[Signal]) -> Tuple[float, float]:
        """Calculate average win/loss in R multiples"""
        wins = [s.pnl for s in signals if s.pnl and s.pnl > 0]
        losses = [abs(s.pnl) for s in signals if s.pnl and s.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 1.0
        avg_loss = np.mean(losses) if losses else 1.0
        
        return avg_win, avg_loss
    
    def _calculate_regime_win_rate(self, signals: List[Signal], 
                                   current_regime: str) -> float:
        """Win rate specifically in this market regime"""
        # This requires regime data to be stored with signals
        # For now, return base win rate
        return self._calculate_win_rate(signals)
    
    def _default_probabilities(self, signal: Signal) -> Dict[str, float]:
        """Default probabilities when no historical data exists"""
        base_wr = 0.5 + (signal.confidence - 50) / 100 * 0.2
        
        return {
            'base_win_rate': base_wr,
            'adjusted_win_rate': base_wr,
            'expected_value': (base_wr * signal.risk_reward) - (1 - base_wr),
            'profit_factor': signal.risk_reward,
            'regime_win_rate': base_wr,
            'sample_size': 0,
            'avg_win_r': signal.risk_reward,
            'avg_loss_r': 1.0,
        }