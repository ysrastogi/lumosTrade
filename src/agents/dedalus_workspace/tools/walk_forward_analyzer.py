import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import random
from dataclasses import dataclass
from scipy import stats
from datetime import datetime, timedelta

class WalkForwardAnalyzer:
    """Out-of-sample validation using walk-forward analysis"""
    
    def __init__(self):
        self.results_history = []
    
    async def analyze(
        self,
        strategy_function: callable,
        optimization_function: callable,
        data: np.ndarray,
        dates: List[datetime],
        in_sample_days: int,
        out_sample_days: int,
        anchored: bool = False
    ) -> Dict[str, Any]:
        """
        Perform walk-forward analysis
        
        Args:
            strategy_function: Strategy to test
            optimization_function: How to optimize parameters
            data: Price/return data
            dates: Corresponding dates
            in_sample_days: Training period length
            out_sample_days: Testing period length
            anchored: If True, in-sample window expands; if False, it rolls
        
        Returns:
            Dictionary with analysis results
        """
        
        periods = []
        in_sample_results = []
        out_sample_results = []
        
        current_start = 0
        
        while current_start + in_sample_days + out_sample_days <= len(data):
            # Define periods
            if anchored:
                in_sample_start = 0
            else:
                in_sample_start = current_start
            
            in_sample_end = current_start + in_sample_days
            out_sample_end = in_sample_end + out_sample_days
            
            # Split data
            in_sample_data = data[in_sample_start:in_sample_end]
            out_sample_data = data[in_sample_end:out_sample_end]
            
            # Optimize on in-sample
            best_params = await optimization_function(in_sample_data)
            
            # Test on in-sample (to see overfitting)
            in_sample_perf = await strategy_function(in_sample_data, best_params)
            
            # Test on out-sample (true performance)
            out_sample_perf = await strategy_function(out_sample_data, best_params)
            
            periods.append({
                'in_sample_start': dates[in_sample_start],
                'in_sample_end': dates[in_sample_end - 1],
                'out_sample_end': dates[out_sample_end - 1],
                'best_params': best_params
            })
            
            in_sample_results.append(in_sample_perf)
            out_sample_results.append(out_sample_perf)
            
            # Move to next period
            current_start = in_sample_end
        
        # Calculate degradation and consistency
        degradation = self._calculate_degradation(in_sample_results, out_sample_results)
        consistency = self._calculate_consistency(out_sample_results)
        
        self.results_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'periods': periods,
            'in_sample_results': in_sample_results,
            'out_sample_results': out_sample_results,
            'degradation': degradation,
            'consistency': consistency
        })
        
        return {
            'num_periods': len(periods),
            'periods': periods,
            'in_sample_avg': np.mean([r.get('sharpe_ratio', 0) for r in in_sample_results]),
            'out_sample_avg': np.mean([r.get('sharpe_ratio', 0) for r in out_sample_results]),
            'degradation_pct': degradation,
            'consistency_score': consistency,
            'detailed_results': {
                'in_sample': in_sample_results,
                'out_sample': out_sample_results
            }
        }
    
    def _calculate_degradation(
        self,
        in_sample_results: List[Dict],
        out_sample_results: List[Dict]
    ) -> float:
        """Calculate performance degradation from in-sample to out-sample"""
        
        in_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in in_sample_results])
        out_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in out_sample_results])
        
        if in_sharpe == 0:
            return 0.0
        
        degradation = ((in_sharpe - out_sharpe) / abs(in_sharpe)) * 100
        return float(degradation)
    
    def _calculate_consistency(self, results: List[Dict]) -> float:
        """Calculate consistency score (lower variance = higher consistency)"""
        
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in results]
        
        if len(sharpe_ratios) < 2:
            return 0.0
        
        mean_sharpe = np.mean(sharpe_ratios)
        std_sharpe = np.std(sharpe_ratios)
        
        if mean_sharpe == 0:
            return 0.0
        
        # Consistency as coefficient of variation (inverted)
        consistency = 1.0 / (1.0 + (std_sharpe / abs(mean_sharpe)))
        
        return float(consistency * 100)  # Scale to 0-100