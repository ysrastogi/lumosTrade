class RecoveryPlanner:
    """
    Plans optimal recovery strategy after drawdowns.
    """
    
    def __init__(self):
        self.recovery_strategies = {
            "minor": {  # < 10% drawdown
                "position_sizing": 0.9,  # Reduce by 10% 
                "timeframe": "immediate"
            },
            "moderate": {  # 10-20% drawdown
                "position_sizing": 0.7,  # Reduce by 30%
                "timeframe": "gradual"
            },
            "severe": {  # > 20% drawdown
                "position_sizing": 0.5,  # Reduce by 50%
                "timeframe": "extended"
            }
        }
    
    def generate_plan(self, current_drawdown, portfolio, risk_tolerance):
        """
        Generate optimal recovery plan after drawdown.
        
        Parameters:
        -----------
        current_drawdown : float
            Current drawdown percentage
        portfolio : list
            Current portfolio state
        risk_tolerance : float
            Trader's risk tolerance
        
        Returns:
        --------
        dict: Recovery plan with steps and timeline
        """
        # Determine severity level
        if current_drawdown < 0.1:
            severity = "minor"
        elif current_drawdown < 0.2:
            severity = "moderate"
        else:
            severity = "severe"
        
        strategy = self.recovery_strategies[severity]
        
        # Generate customized recovery plan
        recovery_plan = {
            "severity": severity,
            "position_sizing_adjustment": strategy["position_sizing"],
            "expected_recovery_time": self._estimate_recovery_time(current_drawdown),
            "steps": self._generate_recovery_steps(severity, portfolio, risk_tolerance),
            "psychological_recommendations": self._psychological_recommendations(severity)
        }
        
        return recovery_plan
    
    def _estimate_recovery_time(self, drawdown):
        """Estimate recovery time based on drawdown severity"""
        # Simple estimation formula: larger drawdowns take exponentially longer
        days = int(100 * (drawdown ** 2))
        return f"{days} trading days"
    
    def _generate_recovery_steps(self, severity, portfolio, risk_tolerance):
        """Generate specific recovery steps based on severity"""
        steps = []
        
        if severity == "minor":
            steps = [
                "Maintain current strategy with 10% reduced position sizing",
                "Review recent trades for patterns",
                "Resume normal position sizing after 5 consecutive profitable trades"
            ]
        elif severity == "moderate":
            steps = [
                "Reduce position sizes by 30% for all new trades",
                "Focus only on highest probability setups",
                "Implement stricter stop-loss rules",
                "Gradually increase position sizing as equity curve recovers"
            ]
        else:  # severe
            steps = [
                "Reduce position sizes by 50%",
                "Consider trading break of 1-2 weeks",
                "Complete strategy review and backtesting",
                "Paper trade for 10 successful trades before resuming",
                "Implement phased recovery of position sizing over 3 months"
            ]
        
        return steps
    
    def _psychological_recommendations(self, severity):
        """Generate psychological recommendations based on severity"""
        if severity == "minor":
            return ["Daily trading journal review", "Maintain routine"]
        elif severity == "moderate":
            return ["Bi-weekly reflection sessions", "Mindfulness practices", "Review trading rules"]
        else:  # severe
            return ["Consider professional coaching", "Comprehensive psychological assessment", 
                    "Develop new mental routines", "Focus on process over outcomes"]