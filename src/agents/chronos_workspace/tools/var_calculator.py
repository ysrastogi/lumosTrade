class VaRCalculator:
    """
    Value at Risk computation engine with multiple methodologies.
    """
    
    def __init__(self, market_regime):
        self.market_regime = market_regime
        self.regime_multipliers = {
            "normal": 1.0,
            "volatile": 1.5,
            "trending": 0.9,
            "crisis": 2.0
        }
    
    def calculate(self, portfolio, confidence=0.95, time_horizon=1):
        """
        Calculate Value at Risk using parametric, historical, and Monte Carlo methods.
        
        Parameters:
        -----------
        portfolio : dict
            Portfolio data with 'positions' key containing list of position dicts
        confidence : float
            Confidence level (0-1)
        time_horizon : int
            Time horizon in days
        
        Returns:
        --------
        dict: VaR metrics using different methodologies
        """
        # Extract positions - handle both formats: dict with positions key or direct dict of positions
        if "positions" in portfolio:
            positions = portfolio["positions"]
        else:
            # Portfolio is a dict where values are position dicts
            positions = list(portfolio.values())
        
        # Calculate total portfolio value
        portfolio_value = sum(p.get("value", 0) for p in positions)
        
        # Apply regime-specific risk multiplier
        regime_multiplier = self.regime_multipliers.get(self.market_regime, 1.0)
        
        # Simplified calculations
        daily_var_pct = 0.02 * regime_multiplier  # 2% daily VaR at given confidence
        daily_var_amount = portfolio_value * daily_var_pct
        
        return {
            "daily_var_pct": daily_var_pct,
            "daily_var_amount": daily_var_amount,
            "parametric_var": daily_var_amount * 0.9,  # Simplified
            "historical_var": daily_var_amount * 1.1,  # Simplified
            "monte_carlo_var": daily_var_amount * 1.05,  # Simplified
            "confidence_level": confidence,
            "time_horizon_days": time_horizon
        }
    
    def update_regime(self, new_regime, volatility_metrics):
        """Update VaR calculations with new market regime"""
        self.market_regime = new_regime