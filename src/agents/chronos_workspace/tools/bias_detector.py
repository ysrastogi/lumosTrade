class BiasDetector:
    """
    Identifies emotional trading patterns and cognitive biases.
    """
    
    def __init__(self, trader_profile):
        self.trader_profile = trader_profile
        self.bias_patterns = {
            "loss_aversion": {
                "pattern": "quick_profit_taking_but_holding_losses",
                "threshold": 0.6
            },
            "overconfidence": {
                "pattern": "increasing_position_sizes_after_wins",
                "threshold": 0.7
            },
            "recency_bias": {
                "pattern": "overweighting_recent_market_action",
                "threshold": 0.65
            },
            "disposition_effect": {
                "pattern": "selling_winners_keeping_losers",
                "threshold": 0.6
            },
            "confirmation_bias": {
                "pattern": "ignoring_contradictory_signals",
                "threshold": 0.7
            }
        }
    
    def analyze(self, behavioral_history, proposed_trade=None):
        """
        Analyze trading history and proposed trade for cognitive biases.
        
        Parameters:
        -----------
        behavioral_history : dict
            Historical trading behaviors
        proposed_trade : dict
            Details of proposed new trade
        
        Returns:
        --------
        list: Detected biases with confidence scores
        """
        # Would analyze patterns in trading history against known bias patterns
        
        # Simplified example
        detected_biases = []
        
        # Example detection logic (simplified)
        if proposed_trade and behavioral_history.get("recent_losses", 0) > 2:
            # Check for revenge trading after losses
            detected_biases.append({
                "bias_type": "revenge_trading",
                "confidence": 0.75,
                "severity": 0.8,
                "evidence": "Multiple trades after consecutive losses",
                "recommendation": "Take a 24-hour trading break"
            })
        
        # Additional bias checks would be implemented here
        
        return detected_biases