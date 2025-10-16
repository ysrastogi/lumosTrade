class PositionSizer:
    """
    Position sizing using Kelly Criterion with psychological adjustments.
    """
    
    def __init__(self, trader_profile, risk_tolerance, market_regime):
        self.trader_profile = trader_profile
        self.base_risk_tolerance = risk_tolerance
        self.market_regime = market_regime
        self.psych_factor = self._calculate_psychological_factor()
        self.regime_adjustments = {
            "normal": 1.0,
            "volatile": 0.7,
            "trending": 1.2,
            "crisis": 0.4
        }
    
    def _calculate_psychological_factor(self):
        """Calculate psychological adjustment factor based on trader profile"""
        # Lower factor for emotionally reactive traders
        emotional_factor = 1 - 0.3 * self.trader_profile.get("emotional_reactivity", 0.5)
        # Lower factor for traders with history of overtrading
        discipline_factor = 1 - 0.3 * self.trader_profile.get("overtrading_tendency", 0.5)
        # Lower factor for traders with poor drawdown management
        drawdown_factor = 1 - 0.4 * self.trader_profile.get("drawdown_aversion", 0.5)
        
        return min(emotional_factor, discipline_factor, drawdown_factor)
    
    def calculate_position_size(self, expected_return, probability_win, volatility):
        """
        Calculate optimal position size using the Kelly Criterion with adjustments.
        
        Parameters:
        -----------
        expected_return : float
            Expected return of the trade
        probability_win : float
            Probability of winning the trade (0-1)
        volatility : float
            Expected volatility of the trade
        
        Returns:
        --------
        float: Recommended position size as percentage of portfolio
        """
        # Basic Kelly formula: f* = p - (1-p)/R where:
        # f* = fraction of portfolio to bet
        # p = probability of win
        # R = win/loss ratio
        
        if expected_return <= 0 or probability_win <= 0:
            return 0
        
        win_loss_ratio = (expected_return / 100) / (1 - (expected_return / 100))
        kelly = probability_win - ((1 - probability_win) / win_loss_ratio)
        
        # Apply "half Kelly" as a safety measure
        kelly *= 0.5
        
        # Apply psychological adjustment
        kelly *= self.psych_factor
        
        # Apply market regime adjustment
        kelly *= self.regime_adjustments.get(self.market_regime, 1.0)
        
        # Apply volatility scaling
        vol_scalar = 1.0 / (1.0 + volatility)
        kelly *= vol_scalar
        
        return max(0, min(kelly, self.base_risk_tolerance))
    
    def analyze_current_exposure(self, portfolio):
        """
        Analyze current portfolio position sizing.
        
        Parameters:
        -----------
        portfolio : dict
            Portfolio data with 'positions' key containing list of position dicts
            
        Returns:
        --------
        dict: Analysis of current position sizing
        """
        # Extract positions - handle both formats: dict with positions key or direct dict of positions
        if "positions" in portfolio:
            positions = portfolio["positions"]
        else:
            # Portfolio is a dict where values are position dicts
            positions = list(portfolio.values())
        
        if not positions:
            # Return safe defaults if no positions
            return {
                "largest_position_pct": 0.0,
                "largest_position_name": "",
                "total_exposure": 0.0,
                "optimal_adjustments": {}
            }
        
        # Calculate largest position percentage
        largest_position_pct = max([p.get("allocation", 0) for p in positions])
        
        # Find the largest position name
        largest_position = max(positions, key=lambda p: p.get("allocation", 0))
        largest_position_name = largest_position.get("symbol", largest_position.get("name", "Unknown"))
        
        # Calculate total exposure
        total_exposure = sum(p.get("allocation", 0) for p in positions)
        
        return {
            "largest_position_pct": largest_position_pct,
            "largest_position_name": largest_position_name,
            "total_exposure": total_exposure,
            "optimal_adjustments": {}  # Would contain recommended adjustments
        }
    
    def update_regime(self, new_regime, volatility_metrics):
        """Update position sizer with new market regime"""
        self.market_regime = new_regime
