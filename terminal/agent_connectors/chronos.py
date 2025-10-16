"""
Chronos Agent Connector
Bridges terminal commands to the Chronos agent instance
Handles agent initialization, lifecycle, and method calls for risk management
"""

import logging
from typing import Dict, Any
from .base import AgentConnector

from src.agents.chronos_workspace.chronos import ChronosAgent
from config.trader_profile import create_ysrastogi_profile


logger = logging.getLogger(__name__)


class ChronosConnector(AgentConnector):
    """Connector for Chronos - Risk Management Agent"""
    
    def __init__(self):
        super().__init__("Chronos")
    
    async def initialize(self):
        """Initialize Chronos agent"""
        try:
            
            
            trader_profile = create_ysrastogi_profile()
            
            self.agent_instance = ChronosAgent(
                trader_profile=trader_profile,
                risk_tolerance=0.02,  # 2% default risk
                market_regime="normal"
            )
            
            # Initialize memory system
            await self.agent_instance.memory.initialize()
            
            self.initialized = True
            logger.info("Chronos agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Chronos: {e}", exc_info=True)
            return False
    
    async def assess_risk(self, metric_choice: str = "5", thresholds: Dict = None) -> Dict[str, Any]:
        """
        Perform risk assessment
        
        Args:
            metric_choice: Which risk metrics to calculate (1-5)
            thresholds: Optional risk thresholds for assessment
            
        Returns:
            Risk assessment results
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Chronos agent not initialized"}
        
        try:
            # Use enhanced trader profile data for realistic portfolio assessment
            trader_profile = self.agent_instance.trader_profile

            # Extract current portfolio from trader profile
            current_portfolio = trader_profile.get("current_portfolio", {})
            positions = current_portfolio.get("positions", [])
            
            # Convert to format expected by agent tools (dict with position values)
            portfolio_data = {f"pos_{i}": pos for i, pos in enumerate(positions)}

            # Use market exposure data from profile
            market_exposure = trader_profile.get("market_exposure", {})
            market_data = {
                "volatility": market_exposure.get("volatility_index", 0.65),
                "regime": market_exposure.get("current_market_regime", "volatile"),
                "correlation_matrix": market_exposure.get("correlation_matrix", {}),
                "beta_to_market": market_exposure.get("beta_to_market", 1.2)
            }

            # Perform risk evaluation with real profile data
            result = await self.agent_instance.evaluate_risk(portfolio_data, market_data)
            
            # Extract key metrics safely
            risk_assessment = result.get('risk_assessment', {})
            position_risk = risk_assessment.get('position_risk', {})
            drawdown_risk = risk_assessment.get('drawdown_risk', {})
            var_metrics = risk_assessment.get('value_at_risk', {})
            
            # Get warnings and recommendations
            warnings = result.get('recommendations', [])
            warning_messages = []
            
            for warning in warnings:
                if isinstance(warning, dict) and 'reasoning' in warning:
                    warning_messages.append(warning['reasoning'])
            
            # Create clean response with safely extracted values
            return {
                "total_exposure": position_risk.get('total_exposure', 0),
                "max_drawdown": drawdown_risk.get('expected_drawdown', 0),
                "var_95": var_metrics.get('var_95', 0),
                "sharpe_ratio": position_risk.get('sharpe_ratio', 0),
                "largest_position": position_risk.get('largest_position_name', ''),
                "largest_position_pct": position_risk.get('largest_position_pct', 0),
                "warnings": warning_messages,
                "full_assessment": risk_assessment
            }
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def calculate_position_size(
        self,
        signal: Dict[str, Any],
        account_balance: float,
        risk_per_trade: float
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size
        
        Args:
            signal: Trading signal dictionary
            account_balance: Current account balance
            risk_per_trade: Risk percentage per trade
            
        Returns:
            Position sizing recommendation
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Chronos agent not initialized"}
        
        try:
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            
            # Calculate risk per unit
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit == 0:
                return {"error": "Invalid stop loss - zero risk"}
            
            # Calculate position size
            risk_amount = account_balance * risk_per_trade
            position_size = risk_amount / risk_per_unit
            position_value = position_size * entry_price
            position_pct = (position_value / account_balance) * 100
            
            # Use Chronos tools for validation
            recommended_size = self.agent_instance.tools.position_sizer.calculate_size(
                account_balance=account_balance,
                entry_price=entry_price,
                stop_loss=stop_loss,
                risk_percentage=risk_per_trade
            )
            
            return {
                "position_size": recommended_size.get('position_size', position_size),
                "position_value": recommended_size.get('position_value', position_value),
                "risk_amount": recommended_size.get('risk_amount', risk_amount),
                "position_pct": recommended_size.get('portfolio_allocation', position_pct),
                "recommendation": recommended_size.get('recommendation', 'Position size calculated')
            }
            
        except Exception as e:
            logger.error(f"Position sizing error: {e}", exc_info=True)
            return {"error": str(e)}

    def _calculate_allocation_efficiency(self, current_portfolio):
        """Calculate how efficiently capital is allocated across positions."""
        positions = current_portfolio.get("positions", [])
        if not positions:
            return 0.0

        # Calculate allocation efficiency based on diversification and concentration
        allocations = [pos.get("allocation", 0.0) for pos in positions]
        max_allocation = max(allocations) if allocations else 0.0

        # Penalize high concentration (efficiency decreases as max allocation increases)
        concentration_penalty = max_allocation * 0.5

        # Reward diversification (more positions = better diversification)
        diversification_bonus = min(len(positions) * 0.1, 0.5)

        # Base efficiency from balanced allocations
        allocation_balance = 1.0 - (sum(abs(alloc - 0.25) for alloc in allocations) / len(allocations))

        return max(0.0, min(1.0, allocation_balance + diversification_bonus - concentration_penalty))

    def _calculate_risk_adjusted_efficiency(self, performance_metrics, risk_metrics):
        """Calculate risk-adjusted efficiency using Sharpe-like ratio."""
        returns = performance_metrics.get("annualized_return", 0.0)
        volatility = risk_metrics.get("portfolio_volatility", 0.0)

        if volatility == 0.0:
            return returns * 100.0  # If no risk, efficiency is pure return

        # Risk-adjusted efficiency (similar to Sharpe ratio but scaled)
        risk_adjusted_return = returns / volatility
        return risk_adjusted_return * 100.0

    async def get_capital_efficiency(self) -> Dict[str, Any]:
        """
        Get capital efficiency analysis using enhanced trader profile data
            
        Returns:
            Capital efficiency report
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Chronos agent not initialized"}
            
        try:
            # Use enhanced trader profile data for capital efficiency analysis
            trader_profile = self.agent_instance.trader_profile
            current_portfolio = trader_profile.get("current_portfolio", {})
            performance_metrics = trader_profile.get("performance_metrics", {})
            risk_metrics = trader_profile.get("risk_metrics", {})

            # Create portfolio data from profile - pass positions as dict values for agent compatibility
            positions = current_portfolio.get("positions", [])
            portfolio_data = {f"pos_{i}": pos for i, pos in enumerate(positions)}

            # Get capital efficiency report using profile data
            result = self.agent_instance.get_capital_efficiency_report(portfolio_data)

            # Enhance result with profile-specific metrics
            if isinstance(result, dict):
                result.update({
                    "profile_performance_metrics": performance_metrics,
                    "profile_risk_metrics": risk_metrics,
                    "allocation_efficiency": self._calculate_allocation_efficiency(current_portfolio),
                    "risk_adjusted_efficiency": self._calculate_risk_adjusted_efficiency(
                        performance_metrics, risk_metrics
                    )
                })

            return result
            
        except Exception as e:
            logger.error(f"Capital efficiency analysis error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def explain_risk_concept(self, concept: str, complexity: str) -> Dict[str, Any]:
        """
        Provide explanation for risk management concept
        
        Args:
            concept: Risk management concept to explain
            complexity: Complexity level (basic, intermediate, advanced)
            
        Returns:
            Explanation of the concept
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Chronos agent not initialized"}
            
        try:
            explanation = self.agent_instance.explain_risk_concept(concept, complexity)
            
            return {
                "concept": concept,
                "complexity": complexity,
                "explanation": explanation
            }
            
        except Exception as e:
            logger.error(f"Risk education error: {e}", exc_info=True)
            return {"error": str(e)}