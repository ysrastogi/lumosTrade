import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

from src.agents.chronos_workspace.tools.position_sizer import PositionSizer
from src.agents.chronos_workspace.tools.drawdown_predictor import DrawdownPredictor
from src.agents.chronos_workspace.tools.var_calculator import VaRCalculator
from src.agents.chronos_workspace.tools.stress_engine import StressTestEngine
from src.agents.chronos_workspace.tools.bias_detector import BiasDetector
from src.agents.chronos_workspace.tools.recovery_planner import RecoveryPlanner
from src.agents.chronos_workspace.tools.correlation_analyzer import CorrelationAnalyzer
from src.agents.chronos_workspace.tools.balance_monitor import BalanceMonitor

class ChronosTools:
    """
    Collection of risk management tools for CHRONOS agent.
    """
    
    def __init__(self, trader_profile, risk_tolerance, market_regime, trading_client=None):
        self.trader_profile = trader_profile
        self.risk_tolerance = risk_tolerance
        self.market_regime = market_regime
        self.trading_client = trading_client
        
        # Initialize component tools
        self.position_sizer = PositionSizer(trader_profile, risk_tolerance, market_regime)
        self.drawdown_predictor = DrawdownPredictor(risk_tolerance)
        self.correlation_analyzer = CorrelationAnalyzer()
        self.var_calculator = VaRCalculator(market_regime)
        self.stress_test_engine = StressTestEngine()
        self.bias_detector = BiasDetector(trader_profile)
        self.recovery_planner = RecoveryPlanner()
        
        # Initialize balance monitor
        initial_balance = 0
        if trading_client:
            # Handle case where balance_info might not be initialized yet
            if hasattr(trading_client, 'balance_info') and trading_client.balance_info:
                initial_balance = trading_client.balance_info.get('balance', 0)
        self.balance_monitor = BalanceMonitor(trading_client, initial_balance)
    
    def update_market_regime(self, new_regime, volatility_metrics):
        """Update all tools with new market regime information"""
        self.market_regime = new_regime
        self.position_sizer.update_regime(new_regime, volatility_metrics)
        self.drawdown_predictor.update_regime(new_regime, volatility_metrics)
        self.var_calculator.update_regime(new_regime, volatility_metrics)
        self.stress_test_engine.update_regime(new_regime, volatility_metrics)
    
    def compile_portfolio_metrics(self, portfolio, market_data=None, behavioral_history=None):
        """
        Compile comprehensive portfolio metrics from all risk management tools.
        
        Parameters:
        -----------
        portfolio : list or dict
            Current portfolio holdings and positions
        market_data : dict, optional
            Market data for calculations including returns, volatility, etc.
        behavioral_history : dict, optional
            Historical trading behavior data for bias detection
        
        Returns:
        --------
        dict: Comprehensive risk metrics and portfolio analysis
        """
        # Get metrics from individual tools
        # Calculate potential drawdowns
        drawdown_metrics = self.drawdown_predictor.forecast(
            portfolio=portfolio, 
            market_data=market_data
        )
        
        # Analyze portfolio concentration risk
        concentration_metrics = self.correlation_analyzer.compute_concentration(
            portfolio=portfolio, 
            returns_data=market_data.get("returns") if market_data else None
        )
        
        # Calculate Value at Risk
        var_metrics = self.var_calculator.calculate(portfolio=portfolio)
        
        # Run stress tests on common scenarios
        stress_test_results = self.stress_test_engine.run_stress_test(portfolio=portfolio)
        
        # Detect behavioral biases if history provided
        bias_metrics = {}
        if behavioral_history:
            bias_metrics["detected_biases"] = self.bias_detector.analyze(behavioral_history)
        
        # Calculate optimal position sizing for reference
        position_sizing = {}
        if market_data and "expected_return" in market_data and "win_probability" in market_data:
            position_sizing["optimal_sizing"] = self.position_sizer.calculate_position_size(
                expected_return=market_data["expected_return"],
                probability_win=market_data["win_probability"],
                volatility=market_data.get("volatility", 0.2)
            )
        
        # Get recovery plan if in drawdown
        recovery_plan = {}
        current_drawdown = market_data.get("current_drawdown", 0) if market_data else 0
        if current_drawdown > 0.05:  # Only generate recovery plan if drawdown > 5%
            recovery_plan = self.recovery_planner.generate_plan(
                current_drawdown=current_drawdown,
                portfolio=portfolio,
                risk_tolerance=self.risk_tolerance
            )
        
        # Compile everything into a comprehensive metrics dictionary
        return {
            "portfolio_summary": {
                "total_value": sum(p.get("value", 0) for p in portfolio) if isinstance(portfolio, list) else portfolio.get("total_value", 0),
                "market_regime": self.market_regime,
                "risk_tolerance_level": self.risk_tolerance
            },
            "risk_metrics": {
                "var": var_metrics,
                "drawdown": drawdown_metrics,
                "concentration": concentration_metrics,
                "stress_tests": stress_test_results
            },
            "behavioral_analysis": bias_metrics,
            "position_sizing": position_sizing,
            "recovery_plan": recovery_plan if recovery_plan else {"status": "not_needed"}
        }
    
