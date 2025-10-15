def create_ysrastogi_profile():
    """
    Creates a personalized trader profile for ysrastogi to be used with CHRONOS.
    This profile captures trading tendencies, psychological patterns, and historical behavior.
    """
    return {
        "user_id": "ysrastogi",
        "profile_created": "2025-10-14",
        "profile_version": 1.0,
        
        # Core risk attributes
        "risk_tolerance": 0.65,  # Moderately high risk tolerance (0-1 scale)
        "base_kelly_fraction": 0.6,  # Uses 60% of Kelly optimal bet sizing
        
        # Psychological attributes
        "emotional_reactivity": 0.7,  # Tendency to react emotionally to market moves (0-1)
        "overtrading_tendency": 0.6,  # Tendency to overtrade during certain periods (0-1)
        "drawdown_aversion": 0.45,  # Relatively lower aversion to drawdowns (0-1)
        "risk_seeking_after_losses": 0.75,  # Tendency to increase risk after losses (0-1)
        "risk_aversion_after_wins": 0.4,  # Lower tendency to become risk-averse after wins (0-1)
        
        # Trading style
        "preferred_timeframes": ["daily", "swing"],  # Preferred trading timeframes
        "avg_position_hold_time": {
            "winners": 12.3,  # days
            "losers": 5.2,    # days
        },
        "position_sizing_patterns": {
            "base_size": 0.08,  # Base position size as % of portfolio
            "max_observed": 0.15,  # Maximum observed position size
            "size_variability": 0.7,  # High variability in position sizing (0-1)
        },
        
        # Historical behavior patterns
        "historical_patterns": {
            "win_rate": 0.62,  # Historical win rate
            "profit_factor": 1.85,  # Profit factor (gross profits / gross losses)
            "avg_win_loss_ratio": 1.3,  # Average win / average loss
            "largest_drawdown": 0.22,  # Largest historical drawdown (22%)
            "drawdown_recovery_behavior": "aggressive",  # Tends to be aggressive after drawdowns
            "market_regime_adaptation": 0.45,  # Moderate ability to adapt to changing regimes (0-1)
        },
        
        # Detected biases
        "cognitive_biases": {
            "loss_aversion": 0.4,  # Below average loss aversion (0-1)
            "recency_bias": 0.8,  # High recency bias - overweighting recent events (0-1)
            "overconfidence": 0.75,  # High overconfidence after successful trades (0-1)
            "confirmation_bias": 0.65,  # Moderately high tendency to seek confirming information (0-1)
            "disposition_effect": 0.55,  # Moderate disposition effect - holding losers too long (0-1)
        },
        
        # Trading behaviors requiring monitoring
        "monitoring_priorities": [
            {
                "behavior": "position_sizing_after_wins",
                "risk": "Position sizes tend to increase by 30-40% after 3+ consecutive winning trades",
                "threshold": 3,  # Consecutive wins to trigger alert
                "adjustment_factor": 0.7  # Recommended adjustment to position sizing
            },
            {
                "behavior": "revenge_trading_after_large_loss",
                "risk": "Tendency to enter new positions within 24 hours of a large (>3%) loss",
                "threshold": 0.03,  # Loss size to trigger alert
                "recommended_pause": 48  # Recommended hours to pause before new trades
            },
            {
                "behavior": "correlation_blindness",
                "risk": "Often takes on multiple correlated positions in technology sector",
                "max_sector_allocation": 0.35,  # Maximum recommended sector allocation
                "correlation_threshold": 0.7  # Correlation threshold for alerts
            }
        ],
        
        # Market regime performance
        "regime_performance": {
            "trending": {
                "win_rate": 0.75,
                "avg_return": 0.18,
                "optimal_position_size_factor": 1.2  # Increase position sizing in trending markets
            },
            "volatile": {
                "win_rate": 0.45,
                "avg_return": -0.05,
                "optimal_position_size_factor": 0.6  # Decrease position sizing in volatile markets
            },
            "ranging": {
                "win_rate": 0.58,
                "avg_return": 0.08,
                "optimal_position_size_factor": 0.9  # Slightly decrease position sizing in ranging markets
            }
        }
    }