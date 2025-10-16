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
        },

        # Current Portfolio Metrics (as of 2025-10-16)
        "current_portfolio": {
            "total_value": 25000.0,  # Current portfolio value in USD
            "cash_allocation": 0.15,  # 15% in cash
            "positions": [
                {
                    "symbol": "BTC/USD",
                    "value": 7500.0,
                    "allocation": 0.30,  # 30% allocation
                    "quantity": 0.15,  # 0.15 BTC
                    "entry_price": 50000.0,
                    "current_price": 52000.0,
                    "unrealized_pnl": 450.0,
                    "sector": "crypto",
                    "volatility": 0.85  # High volatility asset
                },
                {
                    "symbol": "ETH/USD",
                    "value": 5000.0,
                    "allocation": 0.20,  # 20% allocation
                    "quantity": 1.2,  # 1.2 ETH
                    "entry_price": 4000.0,
                    "current_price": 4200.0,
                    "unrealized_pnl": 240.0,
                    "sector": "crypto",
                    "volatility": 0.75
                },
                {
                    "symbol": "AAPL",
                    "value": 6250.0,
                    "allocation": 0.25,  # 25% allocation
                    "quantity": 35.0,  # 35 shares
                    "entry_price": 175.0,
                    "current_price": 179.0,
                    "unrealized_pnl": 140.0,
                    "sector": "technology",
                    "volatility": 0.45
                },
                {
                    "symbol": "SPY",
                    "value": 1250.0,
                    "allocation": 0.05,  # 5% allocation
                    "quantity": 3.0,  # 3 shares
                    "entry_price": 410.0,
                    "current_price": 417.0,
                    "unrealized_pnl": 21.0,
                    "sector": "equity_index",
                    "volatility": 0.25
                }
            ],
            "sector_allocation": {
                "crypto": 0.50,  # 50% in crypto
                "technology": 0.25,  # 25% in technology
                "equity_index": 0.05,  # 5% in equity index
                "cash": 0.15,  # 15% in cash
                "other": 0.05  # 5% in other assets
            },
            "geographic_allocation": {
                "us": 0.30,  # 30% US markets
                "crypto_global": 0.50,  # 50% global crypto
                "international": 0.05  # 5% international
            }
        },

        # Market Metrics and Exposure
        "market_exposure": {
            "current_market_regime": "volatile",  # Current market conditions
            "volatility_index": 0.65,  # Overall portfolio volatility (0-1)
            "beta_to_market": 1.2,  # Portfolio beta relative to S&P 500
            "correlation_matrix": {
                "btc_eth": 0.75,  # BTC-ETH correlation
                "btc_aapl": 0.35,  # BTC-AAPL correlation
                "eth_aapl": 0.40,  # ETH-AAPL correlation
                "aapl_spy": 0.85   # AAPL-SPY correlation
            },
            "liquidity_profile": {
                "high_liquidity": 0.30,  # 30% in highly liquid assets
                "medium_liquidity": 0.50,  # 50% in medium liquidity
                "low_liquidity": 0.20    # 20% in low liquidity assets
            },
            "market_cap_exposure": {
                "large_cap": 0.30,  # 30% large cap
                "mid_cap": 0.05,    # 5% mid cap
                "small_cap": 0.00,  # 0% small cap
                "crypto_large": 0.50  # 50% large crypto
            }
        },

        # Performance Metrics (Last 90 days)
        "performance_metrics": {
            "total_return": 0.085,  # 8.5% total return
            "annualized_return": 0.34,  # 34% annualized
            "sharpe_ratio": 1.45,  # Risk-adjusted return
            "sortino_ratio": 1.85,  # Downside risk-adjusted return
            "max_drawdown": 0.12,  # 12% maximum drawdown
            "current_drawdown": 0.03,  # 3% current drawdown
            "win_rate": 0.65,  # 65% win rate
            "profit_factor": 2.1,  # Profit factor
            "avg_win": 0.025,  # Average winning trade 2.5%
            "avg_loss": -0.015,  # Average losing trade -1.5%
            "largest_win": 0.08,  # Largest winning trade 8%
            "largest_loss": -0.045,  # Largest losing trade -4.5%
            "recovery_time_avg": 12.5,  # Average days to recover from drawdown
            "volatility_adjusted_return": 0.28  # Volatility-adjusted return
        },

        # Risk Metrics
        "risk_metrics": {
            "value_at_risk_95": 0.085,  # 95% VaR = 8.5% loss
            "expected_shortfall_95": 0.12,  # Expected shortfall at 95%
            "tail_risk_measure": 0.18,  # Tail risk measure
            "stress_test_results": {
                "covid_crash": -0.25,  # -25% in COVID-like scenario
                "rate_hike": -0.08,    # -8% in rate hike scenario
                "crypto_winter": -0.35  # -35% in crypto winter scenario
            },
            "concentration_risk": {
                "single_asset_max": 0.30,  # Max 30% in single asset
                "sector_max": 0.50,        # Max 50% in single sector
                "correlation_risk_score": 0.65  # Portfolio correlation risk
            },
            "liquidity_risk": {
                "days_to_liquidate_50pct": 2.5,  # Days to liquidate 50% of portfolio
                "days_to_liquidate_100pct": 5.0,  # Days to liquidate entire portfolio
                "liquidity_coverage_ratio": 1.8   # Liquidity coverage ratio
            }
        },

        # Asset Allocation Preferences
        "allocation_preferences": {
            "target_allocations": {
                "crypto": {"min": 0.30, "target": 0.40, "max": 0.50},
                "technology": {"min": 0.15, "target": 0.25, "max": 0.35},
                "equity_index": {"min": 0.05, "target": 0.10, "max": 0.15},
                "cash": {"min": 0.10, "target": 0.15, "max": 0.25},
                "international": {"min": 0.05, "target": 0.10, "max": 0.15}
            },
            "rebalancing_threshold": 0.05,  # Rebalance when allocation deviates by 5%
            "tactical_tilts": {
                "momentum_tilt": 0.1,  # Slight momentum bias
                "quality_tilt": 0.05,  # Slight quality bias
                "volatility_tilt": -0.1  # Slight volatility avoidance
            },
            "blacklist_sectors": ["real_estate", "energy"],  # Avoid these sectors
            "preferred_sectors": ["technology", "crypto", "healthcare"]  # Preferred sectors
        },

        # Market Timing and Regime Detection
        "market_timing": {
            "regime_detection_confidence": 0.75,  # Confidence in regime detection
            "regime_transition_probability": {
                "normal_to_volatile": 0.25,
                "volatile_to_crisis": 0.15,
                "crisis_to_normal": 0.20
            },
            "volatility_regimes": {
                "low_vol": {"threshold": 0.15, "position_size_factor": 1.3},
                "normal_vol": {"threshold": 0.25, "position_size_factor": 1.0},
                "high_vol": {"threshold": 0.35, "position_size_factor": 0.7},
                "extreme_vol": {"threshold": 0.50, "position_size_factor": 0.4}
            },
            "trend_strength": {
                "weak_trend": {"factor": 0.8},
                "moderate_trend": {"factor": 1.0},
                "strong_trend": {"factor": 1.2}
            }
        }
    }