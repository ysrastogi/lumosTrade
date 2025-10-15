from src.agents.chronos_workspace.tools.base import ChronosTools
from src.agents.chronos_workspace.memory_manager import ChronosMemory
from src.agents.chronos_workspace.prompts import ChronosPrompts
from src.llm.client import gemini
import json
import datetime
from typing import Dict, List, Any, Optional

class ChronosLLMEngine:
    """
    LLM integration for CHRONOS risk management agent.
    Handles natural language interaction, explanations, and report generation.
    Uses Gemini for natural language processing.
    """
    
    def __init__(self, model="gemini-2.5-flash"):
        """
        Initialize the LLM engine with Gemini integration.
        
        Parameters:
        -----------
        model : str
            The Gemini model to use (defaults to "gemini-2.5-flash")
        """
        self.model = model
        self.gemini_client = gemini
        self.prompts = ChronosPrompts()
    
    def explain_concept(self, concept, complexity_level="intermediate"):
        """
        Provide clear explanation of risk management concept using Gemini.
        
        Parameters:
        -----------
        concept : str
            Risk management concept to explain
        complexity_level : str
            Level of complexity for explanation ("basic", "intermediate", "advanced")
        
        Returns:
        --------
        str: Natural language explanation of the concept
        """
        prompt = self.prompts.RISK_EXPLANATION.format(
            complexity=complexity_level,
            concept=concept
        )
        
        system_instruction = self.prompts.SYSTEM_INSTRUCTIONS["risk_explanation"]
        
        return self.gemini_client.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2 
        )
    
    def justify_position_reduction(self, current_metrics, psychological_patterns):
        """
        Generate explanation for why position size should be reduced using Gemini.
        
        Parameters:
        -----------
        current_metrics : dict
            Current portfolio and risk metrics
        psychological_patterns : dict
            Recent psychological patterns detected
        
        Returns:
        --------
        str: Natural language explanation
        """
        # Format the psychological patterns for better prompt readability
        psych_patterns_formatted = json.dumps(psychological_patterns, indent=2)
        
        prompt = self.prompts.POSITION_SIZE_JUSTIFICATION.format(
            current_exposure=current_metrics.get("current_exposure", 0),
            optimal_exposure=current_metrics.get("optimal_exposure", 0),
            market_regime=current_metrics.get("market_regime", "unknown"),
            psychological_patterns=psych_patterns_formatted
        )
        
        system_instruction = self.prompts.SYSTEM_INSTRUCTIONS["position_justification"]
        
        return self.gemini_client.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.3  # Moderate temperature for personalized but focused response
        )
    
    def generate_report(self, report_data, timeframe="weekly"):
        """
        Generate comprehensive risk management report using Gemini.
        
        Parameters:
        -----------
        report_data : dict
            Data to include in the report
        timeframe : str
            Timeframe of the report ("daily", "weekly", "monthly")
        
        Returns:
        --------
        str: Formatted risk management report
        """
        # Format the complex nested data structures for better prompt readability
        portfolio_metrics = json.dumps(report_data.get("portfolio_metrics", {}), indent=2)
        historical_comparison = json.dumps(report_data.get("historical_comparison", {}), indent=2)
        behavioral_patterns = json.dumps(report_data.get("behavioral_patterns", {}), indent=2)
        
        prompt = self.prompts.RISK_REPORT.format(
            timeframe=timeframe,
            portfolio_metrics=portfolio_metrics,
            historical_comparison=historical_comparison,
            behavioral_patterns=behavioral_patterns
        )
        
        system_instruction = self.prompts.SYSTEM_INSTRUCTIONS["risk_report"]
        
        return self.gemini_client.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2,  # Lower temperature for consistent reporting
            max_output_tokens=4096  # Allow for detailed reports
        )

class ChronosAgent:
    """
    CHRONOS Risk Governance Agent - "The Guardian of Capital"
    Enforces disciplined risk management while adapting to trader psychology and market regime.
    """
    
    def __init__(self, trader_profile, risk_tolerance, market_regime="normal", trading_client=None):
        """
        Initialize the CHRONOS agent with trader profile and risk parameters.
        
        Parameters:
        -----------
        trader_profile : dict
            Profile containing trader's historical behavior and psychological traits
        risk_tolerance : float
            Base risk tolerance level (0-1)
        market_regime : str
            Current market conditions ("normal", "volatile", "trending", "crisis")
        trading_client : TradingClient, optional
            Reference to the trading client for balance and portfolio monitoring
        """
        self.trader_profile = trader_profile
        self.risk_tolerance = risk_tolerance
        self.market_regime = market_regime
        self.trading_client = trading_client
        
        # Initialize components
        self.tools = ChronosTools(trader_profile, risk_tolerance, market_regime, trading_client)
        self.memory = ChronosMemory(trader_profile["user_id"])
        self.llm_engine = ChronosLLMEngine()
        
    def evaluate_risk(self, portfolio, market_data, proposed_trade=None):
        """
        Perform comprehensive risk assessment on current portfolio and proposed trades.
        
        Returns:
        --------
        dict: Risk assessment results and recommendations
        """
        # Current portfolio analysis
        position_risk = self.tools.position_sizer.analyze_current_exposure(portfolio)
        drawdown_risk = self.tools.drawdown_predictor.forecast(portfolio, market_data)
        correlation_risk = self.tools.correlation_analyzer.compute_concentration(portfolio)
        var_metrics = self.tools.var_calculator.calculate(portfolio, confidence=0.95)
        
        # Detect behavioral patterns
        behavioral_flags = self.tools.bias_detector.analyze(
            self.memory.behavioral_patterns, 
            proposed_trade
        )
        
        # Store analysis in memory
        self.memory.update_risk_history(position_risk, drawdown_risk, var_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            position_risk, 
            drawdown_risk, 
            correlation_risk,
            var_metrics,
            behavioral_flags
        )
        
        # Log any risk violations
        if self._check_violations(position_risk, drawdown_risk, var_metrics):
            self.memory.log_violation(portfolio, market_data, proposed_trade)
        
        return {
            "risk_assessment": {
                "position_risk": position_risk,
                "drawdown_risk": drawdown_risk,
                "correlation_risk": correlation_risk,
                "value_at_risk": var_metrics,
                "behavioral_flags": behavioral_flags
            },
            "recommendations": recommendations
        }
    
    def update_market_regime(self, new_regime, volatility_metrics):
        """
        Update the market regime classification and adjust risk parameters.
        """
        self.market_regime = new_regime
        self.tools.update_market_regime(new_regime, volatility_metrics)
        self.memory.update_regime_memory(new_regime, volatility_metrics)
        
    def generate_risk_report(self, timeframe="weekly"):
        """
        Generate comprehensive risk management report.
        """
        report_data = {
            "portfolio_metrics": self.tools.compile_portfolio_metrics(),
            "historical_comparison": self.memory.get_historical_comparison(),
            "behavioral_patterns": self.memory.get_behavioral_patterns(),
            "recommendations": self.tools.generate_recommendations()
        }
        
        return self.llm_engine.generate_report(report_data, timeframe)
    
    def explain_risk_concept(self, concept, complexity_level="intermediate"):
        """
        Provide trader with explanation of risk concept.
        """
        return self.llm_engine.explain_concept(concept, complexity_level)
    
    def _generate_recommendations(self, position_risk, drawdown_risk, correlation_risk, var_metrics, behavioral_flags):
        """
        Generate action recommendations based on risk assessments.
        """
        recommendations = []
        
        # Position sizing recommendations
        if position_risk["largest_position_pct"] > self.risk_tolerance * 2:
            recommendations.append({
                "type": "position_sizing",
                "action": "reduce",
                "target": position_risk["largest_position_name"],
                "reasoning": "Position exceeds optimal size based on Kelly Criterion",
                "urgency": "high"
            })
            
        if correlation_risk["concentration_score"] > 0.7:
            recommendations.append({
                "type": "diversification",
                "action": "reduce_correlation",
                "target": correlation_risk["highest_correlated_cluster"],
                "reasoning": "Portfolio concentration exceeds threshold",
                "urgency": "medium"
            })
            
        for flag in behavioral_flags:
            if flag["severity"] > 0.7:
                recommendations.append({
                    "type": "behavioral",
                    "action": "pause",
                    "target": flag["bias_type"],
                    "reasoning": f"Detected {flag['bias_type']} bias in trading pattern",
                    "urgency": "high"
                })
                
        return recommendations
    
    def _check_violations(self, position_risk, drawdown_risk, var_metrics):
        """
        Check if any risk parameters have been violated.
        """
        violations = []
        
        if position_risk["largest_position_pct"] > self.risk_tolerance * 3:
            violations.append("position_size")
            
        if drawdown_risk["expected_drawdown"] > self.risk_tolerance * 20:
            violations.append("drawdown_risk")
            
        if var_metrics["daily_var_pct"] > self.risk_tolerance * 5:
            violations.append("var_exceeded")
        
        # Check balance-based violations if balance monitor is available
        if hasattr(self.tools, 'balance_monitor'):
            drawdown_metrics = self.tools.balance_monitor.get_drawdown_metrics()
            if drawdown_metrics['current_drawdown'] > self.risk_tolerance * 2:
                violations.append("account_drawdown")
            
        return violations
    
    def monitor_balance(self):
        """
        Monitor current account balance and get balance metrics.
        
        Returns:
        --------
        dict: Balance status report
        """
        if hasattr(self.tools, 'balance_monitor'):
            return self.tools.balance_monitor.get_status_report()
        return {"error": "Balance monitor not available"}
    
    def check_position_sizing(self, proposed_size, max_risk_pct=None):
        """
        Check if a proposed position size is within risk limits.
        
        Parameters:
        -----------
        proposed_size : float
            The proposed position size
        max_risk_pct : float, optional
            Maximum risk percentage (defaults to agent's risk tolerance)
            
        Returns:
        --------
        dict: Position sizing assessment
        """
        if not hasattr(self.tools, 'balance_monitor'):
            return {"error": "Balance monitor not available"}
            
        risk_pct = max_risk_pct if max_risk_pct is not None else self.risk_tolerance
        return self.tools.balance_monitor.check_risk_limits(proposed_size, risk_pct)
    
    def refresh_balance(self, callback=None):
        """
        Force refresh of account balance from trading API
        
        Parameters:
        -----------
        callback : callable, optional
            Function to call after balance is refreshed
        """
        if hasattr(self.tools, 'balance_monitor'):
            self.tools.balance_monitor.force_refresh(callback)
        else:
            if callback:
                callback({"error": "Balance monitor not available"})
            
    def get_capital_efficiency_report(self, portfolio):
        """
        Generate report on how efficiently capital is being used.
        
        Parameters:
        -----------
        portfolio : dict
            Current portfolio data
            
        Returns:
        --------
        dict: Capital efficiency metrics
        """
        if not hasattr(self.tools, 'balance_monitor'):
            return {"error": "Balance monitor not available"}
        
        balance = self.tools.balance_monitor.current_balance
        
        # Calculate capital allocation
        allocated_capital = sum(position.get('value', 0) for position in portfolio.values())
        allocation_pct = allocated_capital / balance if balance > 0 else 0
        
        # Get position sizing recommendations
        max_per_position = self.tools.balance_monitor.estimate_max_position_size()
        
        return {
            "current_balance": balance,
            "allocated_capital": allocated_capital,
            "allocation_percentage": allocation_pct * 100,
            "idle_capital": balance - allocated_capital,
            "idle_percentage": (1 - allocation_pct) * 100 if balance > 0 else 0,
            "max_position_size": max_per_position,
            "drawdown_metrics": self.tools.balance_monitor.get_drawdown_metrics(),
            "balance_trend": self.tools.balance_monitor.get_balance_trend()
        }