class BacktestValidator:
    """
    Rapid historical validation of signals
    """
    
    def __init__(self, memory_system):
        self.memory = memory_system
        
    def validate_signal(self, signal: Signal, market_data: dict) -> Dict[str, any]:
        """
        Quick backtest validation
        
        Returns metrics on similar historical signals
        """
        # Find similar patterns
        similar = self.memory.find_similar_patterns(
            pattern=signal.pattern,
            symbol=signal.symbol,
            regime=market_data['regime'],
            direction=signal.direction,
            limit=200
        )
        
        if len(similar) < 10:
            return {
                'validated': False,
                'reason': 'Insufficient historical data',
                'sample_size': len(similar),
                'recommendation': 'PAPER_TRADE'
            }
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(similar)
        
        # Validation criteria
        validated = (
            metrics['win_rate'] >= 0.45 and
            metrics['profit_factor'] >= 1.2 and
            metrics['avg_win'] / metrics['avg_loss'] >= 1.5
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(metrics, validated)
        
        return {
            'validated': validated,
            'metrics': metrics,
            'sample_size': len(similar),
            'recommendation': recommendation,
            'confidence_impact': self._calculate_confidence_impact(metrics),
            'similar_signals': similar[:5]  # Top 5 for review
        }
    
    def _calculate_performance_metrics(self, signals: List[Signal]) -> Dict:
        """Calculate comprehensive performance metrics"""
        completed = [s for s in signals if s.outcome in ['win', 'loss']]
        
        if not completed:
            return {}
        
        wins = [s for s in completed if s.outcome == 'win']
        losses = [s for s in completed if s.outcome == 'loss']
        
        total_pnl = sum(s.pnl for s in completed if s.pnl)
        win_pnl = sum(s.pnl for s in wins if s.pnl)
        loss_pnl = abs(sum(s.pnl for s in losses if s.pnl))
        
        return {
            'win_rate': len(wins) / len(completed),
            'total_trades': len(completed),
            'wins': len(wins),
            'losses': len(losses),
            'avg_win': np.mean([s.pnl for s in wins if s.pnl]) if wins else 0,
            'avg_loss': np.mean([abs(s.pnl) for s in losses if s.pnl]) if losses else 1,
            'profit_factor': win_pnl / loss_pnl if loss_pnl > 0 else 0,
            'total_return': total_pnl,
            'max_consecutive_wins': self._max_consecutive(wins),
            'max_consecutive_losses': self._max_consecutive(losses),
        }
    
    def _max_consecutive(self, signals: List[Signal]) -> int:
        """Calculate maximum consecutive wins/losses"""
        if not signals:
            return 0
        
        max_streak = 1
        current_streak = 1
        
        for i in range(1, len(signals)):
            if signals[i].timestamp > signals[i-1].timestamp:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        
        return max_streak
    
    def _generate_recommendation(self, metrics: Dict, validated: bool) -> str:
        """Generate trading recommendation based on backtest"""
        if not metrics:
            return "SKIP - Insufficient data"
        
        win_rate = metrics['win_rate']
        profit_factor = metrics['profit_factor']
        sample_size = metrics['total_trades']
        
        # Exceptional setups
        if win_rate >= 0.65 and profit_factor >= 2.0 and sample_size >= 50:
            return "STRONG_TAKE - Proven high-probability setup"
        
        # Good setups
        if win_rate >= 0.55 and profit_factor >= 1.5 and sample_size >= 30:
            return "TAKE - Solid edge with good sample"
        
        # Marginal setups
        if win_rate >= 0.45 and profit_factor >= 1.2 and sample_size >= 20:
            return "TAKE_SMALL - Positive edge, reduce size"
        
        # Unproven but promising
        if sample_size < 20 and win_rate >= 0.50:
            return "PAPER_TRADE - Monitor for data collection"
        
        # Poor performance
        if win_rate < 0.40 or profit_factor < 1.0:
            return "SKIP - Negative historical edge"
        
        return "MONITOR - Needs more data"
    
    def _calculate_confidence_impact(self, metrics: Dict) -> float:
        """
        Calculate how much to adjust confidence based on backtest
        
        Returns: Multiplier (0.8 - 1.2)
        """
        if not metrics:
            return 1.0
        
        win_rate = metrics['win_rate']
        profit_factor = metrics['profit_factor']
        
        # Boost confidence for strong historical performance
        if win_rate >= 0.60 and profit_factor >= 1.8:
            return 1.15
        
        # Slight boost for good performance
        if win_rate >= 0.50 and profit_factor >= 1.3:
            return 1.05
        
        # Reduce confidence for poor performance
        if win_rate < 0.45 or profit_factor < 1.1:
            return 0.85
        
        return 1.0