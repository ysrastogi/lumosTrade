import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import random
from dataclasses import dataclass
from scipy import stats
from datetime import datetime, timedelta

class PortfolioRebalancer:
    """Optimize portfolio allocation across multiple strategies"""
    
    def __init__(self):
        self.allocation_history: List[Dict] = []
    
    def optimize_allocation(
        self,
        strategy_results: List[SimulationResult],
        method: str = "mean_variance",  # mean_variance, risk_parity, equal_weight, max_sharpe
        target_return: Optional[float] = None,
        max_position: float = 0.3
    ) -> Dict[str, float]:
        """
        Calculate optimal portfolio allocation
        
        Args:
            strategy_results: List of strategy simulation results
            method: Allocation optimization method
            target_return: Target portfolio return (for mean-variance)
            max_position: Maximum weight per strategy
        
        Returns:
            Dictionary of {strategy_id: weight}
        """
        n = len(strategy_results)
        
        # Extract returns from equity curves
        returns_matrix = []
        strategy_ids = []
        
        for result in strategy_results:
            if len(result.equity_curve) > 1:
                equity_series = pd.Series(result.equity_curve)
                returns = equity_series.pct_change().dropna()
                returns_matrix.append(returns.values)
                strategy_ids.append(result.strategy_id)
        
        if not returns_matrix:
            # Equal weight fallback
            weight = 1.0 / n
            return {r.strategy_id: weight for r in strategy_results}
        
        # Pad returns to same length
        max_len = max(len(r) for r in returns_matrix)
        returns_matrix = [
            np.pad(r, (0, max_len - len(r)), constant_values=0)
            for r in returns_matrix
        ]
        
        returns_df = pd.DataFrame(returns_matrix).T
        
        if method == "equal_weight":
            weights = np.ones(len(strategy_ids)) / len(strategy_ids)
        
        elif method == "risk_parity":
            weights = self._risk_parity(returns_df)
        
        elif method == "mean_variance":
            weights = self._mean_variance_optimization(returns_df, target_return)
        
        elif method == "max_sharpe":
            weights = self._max_sharpe_portfolio(returns_df)
        
        else:
            weights = np.ones(len(strategy_ids)) / len(strategy_ids)
        
        # Apply position limits
        weights = np.clip(weights, 0, max_position)
        weights = weights / weights.sum()  # Renormalize
        
        allocation = dict(zip(strategy_ids, weights))
        
        self.allocation_history.append({
            "timestamp": datetime.now(),
            "method": method,
            "allocation": allocation
        })
        
        return allocation
    
    def _risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """Risk parity allocation"""
        volatilities = returns.std()
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        return weights.values
    
    def _mean_variance_optimization(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float] = None
    ) -> np.ndarray:
        """Mean-variance optimization (simplified)"""
        n = len(returns.columns)
        
        # Calculate expected returns and covariance
        mu = returns.mean()
        cov = returns.cov()
        
        # Inverse covariance
        try:
            inv_cov = np.linalg.inv(cov.values)
        except:
            # If singular, use equal weights
            return np.ones(n) / n
        
        if target_return is None:
            # Minimum variance portfolio
            ones = np.ones(n)
            weights = inv_cov @ ones
            weights = weights / weights.sum()
        else:
            # Target return portfolio
            ones = np.ones(n)
            A = inv_cov @ mu.values
            B = inv_cov @ ones
            
            denom = (ones @ A) - target_return * (ones @ B)
            if abs(denom) < 1e-10:
                return np.ones(n) / n
            
            weights = (A - target_return * B) / denom
        
        # Ensure non-negative
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        
        return weights
    
    def _max_sharpe_portfolio(self, returns: pd.DataFrame) -> np.ndarray:
        """Maximum Sharpe ratio portfolio"""
        n = len(returns.columns)
        mu = returns.mean()
        cov = returns.cov()
        
        try:
            inv_cov = np.linalg.inv(cov.values)
            weights = inv_cov @ mu.values
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()
        except:
            weights = np.ones(n) / n
        
        return weights
    
    def calculate_portfolio_metrics(
        self,
        strategy_results: List[SimulationResult],
        allocation: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate aggregated portfolio metrics"""
        # Combine equity curves based on allocation
        combined_equity = None
        
        for result in strategy_results:
            weight = allocation.get(result.strategy_id, 0)
            if weight > 0 and len(result.equity_curve) > 0:
                equity = np.array(result.equity_curve)
                if combined_equity is None:
                    combined_equity = equity * weight
                else:
                    # Pad to same length
                    if len(equity) < len(combined_equity):
                        equity = np.pad(equity, (0, len(combined_equity) - len(equity)), 
                                      constant_values=equity[-1])
                    elif len(equity) > len(combined_equity):
                        combined_equity = np.pad(combined_equity, 
                                                (0, len(equity) - len(combined_equity)),
                                                constant_values=combined_equity[-1])
                    combined_equity += equity * weight
        
        if combined_equity is None:
            return {}
        
        # Calculate metrics
        returns = pd.Series(combined_equity).pct_change().dropna()
        
        total_return = (combined_equity[-1] / combined_equity[0]) - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "volatility": returns.std() * np.sqrt(252),
            "calmar_ratio": total_return / abs(max_dd) if max_dd != 0 else 0
        }