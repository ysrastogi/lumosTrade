import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

class ScenarioGenerator:
    """What-if analysis engine"""
    
    def __init__(self):
        self.scenarios = {}
    
    def create_scenario(
        self,
        name: str,
        base_data: np.ndarray,
        modifications: Dict[str, Any]
    ) -> np.ndarray:
        """
        Create a what-if scenario by modifying market data
        
        Modifications can include:
        - volatility_multiplier: Multiply volatility by factor
        - trend_adjustment: Add/subtract trend
        - crash_simulation: Simulate market crash
        - regime_change: Change market regime
        """
        
        scenario_data = base_data.copy()
        
        if 'volatility_multiplier' in modifications:
            scenario_data = self._adjust_volatility(
                scenario_data,
                modifications['volatility_multiplier']
            )
        
        if 'trend_adjustment' in modifications:
            scenario_data = self._adjust_trend(
                scenario_data,
                modifications['trend_adjustment']
            )
        
        if 'crash_simulation' in modifications:
            scenario_data = self._simulate_crash(
                scenario_data,
                modifications['crash_simulation']
            )
        
        if 'regime_change' in modifications:
            scenario_data = self._simulate_regime_change(
                scenario_data,
                modifications['regime_change']
            )
        
        self.scenarios[name] = {
            'data': scenario_data,
            'modifications': modifications,
            'created_at': datetime.utcnow().isoformat()
        }
        
        return scenario_data
    
    def _adjust_volatility(self, data: np.ndarray, multiplier: float) -> np.ndarray:
        """Adjust volatility by multiplier"""
        returns = np.diff(data) / data[:-1]
        mean_return = np.mean(returns)
        
        # Center returns, scale, then re-add mean
        centered = returns - mean_return
        scaled = centered * multiplier
        adjusted_returns = scaled + mean_return
        
        # Reconstruct price series
        adjusted_data = np.zeros_like(data)
        adjusted_data[0] = data[0]
        for i in range(len(adjusted_returns)):
            adjusted_data[i + 1] = adjusted_data[i] * (1 + adjusted_returns[i])
        
        return adjusted_data
    
    def _adjust_trend(self, data: np.ndarray, trend_pct: float) -> np.ndarray:
        """Add trend to data (positive or negative)"""
        daily_trend = trend_pct / len(data)
        trend_factors = np.exp(np.linspace(0, trend_pct / 100, len(data)))
        return data * trend_factors
    
    def _simulate_crash(self, data: np.ndarray, crash_config: Dict) -> np.ndarray:
        """Simulate market crash at specific point"""
        crash_point = crash_config.get('day', len(data) // 2)
        crash_magnitude = crash_config.get('magnitude_pct', -20)
        recovery_days = crash_config.get('recovery_days', 30)
        
        adjusted_data = data.copy()
        crash_factor = 1 + (crash_magnitude / 100)
        
        # Apply crash
        adjusted_data[crash_point:] *= crash_factor
        
        # Gradual recovery
        if recovery_days > 0:
            recovery_path = np.linspace(0, 1, min(recovery_days, len(data) - crash_point))
            for i, recovery_factor in enumerate(recovery_path):
                idx = crash_point + i
                if idx < len(adjusted_data):
                    adjusted_data[idx] *= (1 + (1 - crash_factor) * recovery_factor)
        
        return adjusted_data
    
    def _simulate_regime_change(self, data: np.ndarray, regime_config: Dict) -> np.ndarray:
        """Simulate market regime change"""
        change_point = regime_config.get('day', len(data) // 2)
        new_volatility = regime_config.get('new_volatility_multiplier', 1.5)
        new_trend = regime_config.get('new_trend_pct', 0)
        
        adjusted_data = data.copy()
        
        # Apply new regime to second half
        second_half = adjusted_data[change_point:]
        second_half = self._adjust_volatility(second_half, new_volatility)
        second_half = self._adjust_trend(second_half, new_trend)
        
        adjusted_data[change_point:] = second_half
        
        return adjusted_data
    
    def get_scenario(self, name: str) -> Optional[Dict]:
        """Retrieve a created scenario"""
        return self.scenarios.get(name)
    
    def list_scenarios(self) -> List[str]:
        """List all created scenario names"""
        return list(self.scenarios.keys())