class StressTestEngine:
    """
    Scenario analysis engine for testing portfolio resilience to extreme events.
    """
    
    def __init__(self):
        self.scenarios = {
            "2008_crisis": {"equity": -0.50, "bonds": 0.15, "gold": 0.05},
            "2020_covid": {"equity": -0.35, "bonds": 0.10, "gold": 0.12},
            "rate_hike": {"equity": -0.15, "bonds": -0.10, "gold": -0.05},
            "inflation_spike": {"equity": -0.10, "bonds": -0.15, "gold": 0.20}
        }
    
    def run_stress_test(self, portfolio, scenario_key=None):
        """
        Run stress test on portfolio using predefined or custom scenarios.
        
        Parameters:
        -----------
        portfolio : list
            List of portfolio positions
        scenario_key : str
            Key for predefined scenario or None for all scenarios
        
        Returns:
        --------
        dict: Stress test results showing portfolio impact
        """

        results = {}
        scenarios_to_test = [scenario_key] if scenario_key else self.scenarios.keys()
        
        for scenario in scenarios_to_test:
            if scenario in self.scenarios:
                # Calculate impact based on asset class exposures
                impact = -0.25  # Simplified: 25% portfolio reduction in worst case
                
                results[scenario] = {
                    "portfolio_impact_pct": impact,
                    "max_drawdown": abs(impact),
                    "most_affected_assets": ["Example Asset 1", "Example Asset 2"]
                }
        
        return results
    
    def update_regime(self, new_regime, volatility_metrics):
        """Update stress test scenarios with new market regime"""
        # Would adjust scenario parameters based on regime
        pass