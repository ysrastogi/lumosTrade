"""
Daedalus Agent Connector
Bridges terminal commands to the Daedalus agent instance
Handles agent initialization, lifecycle, and method calls for strategy simulation
"""

import logging
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta
from .base import AgentConnector


logger = logging.getLogger(__name__)


class DaedalusConnector(AgentConnector):
    """Connector for Daedalus - Strategy Simulation Agent"""
    
    def __init__(self):
        super().__init__("Daedalus")
        self.cache = None
    
    async def initialize(self):
        """Initialize Daedalus agent"""
        try:
            from src.agents.daedalus_workspace.daedalus import DaedalusAgent
            from src.data_layer.aggregator import InMemoryCache
            
            self.agent_instance = DaedalusAgent()
            self.cache = InMemoryCache.get_instance()
            self.initialized = True
            logger.info("Daedalus agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Daedalus: {e}", exc_info=True)
            return False
    
    def _fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: int
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from cache
        
        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe (1h, 4h, 1d)
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.cache:
            raise ValueError("Cache not initialized")
        
        # Get OHLC data from cache
        ohlc_data = self.cache.get_all_ohlc()
        
        if not ohlc_data:
            raise ValueError(f"No historical data available for {symbol} at {timeframe}")
        
        candles = ohlc_data['candles']
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
        
        # Rename columns to match backtesting.py requirements
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Filter by lookback period
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        df = df[df.index >= cutoff_date]
        
        if df.empty:
            raise ValueError(f"No data available for the specified lookback period")
        
        logger.info(f"Fetched {len(df)} candles for {symbol} at {timeframe}")
        
        return df
    
    def _create_strategy_config(self, strategy_type: str) -> Dict[str, Any]:
        """
        Create strategy configuration based on user selection
        
        Args:
            strategy_type: Strategy type (1-4 from user input)
            
        Returns:
            Strategy configuration dict
        """
        from src.agents.daedalus_workspace.models import StrategyConfig
        
        strategy_map = {
            "1": {
                "name": "Mean Reversion",
                "type": "mean_reversion",
                "parameters": {
                    "rsi_period": 14,
                    "upper_threshold": 70,
                    "lower_threshold": 30
                },
                "entry_rules": ["rsi_oversold"],
                "exit_rules": ["rsi_overbought"],
                "risk_params": {
                    "stop_loss": 0.02,
                    "take_profit": 0.04
                }
            },
            "2": {
                "name": "Trend Following",
                "type": "momentum",
                "parameters": {
                    "fast": 10,
                    "slow": 50
                },
                "entry_rules": ["ma_cross_up"],
                "exit_rules": ["ma_cross_down"],
                "risk_params": {
                    "stop_loss": 0.03,
                    "take_profit": 0.06
                }
            },
            "3": {
                "name": "Breakout",
                "type": "breakout",
                "parameters": {
                    "lookback": 20,
                    "atr_period": 14
                },
                "entry_rules": ["price_breaks_high"],
                "exit_rules": ["price_breaks_low"],
                "risk_params": {
                    "stop_loss": 0.025,
                    "take_profit": 0.05
                }
            },
            "4": {
                "name": "Custom Strategy",
                "type": "custom",
                "parameters": {
                    "param1": 10,
                    "param2": 20
                },
                "entry_rules": ["custom_entry"],
                "exit_rules": ["custom_exit"],
                "risk_params": {
                    "stop_loss": 0.02,
                    "take_profit": 0.04
                }
            }
        }
        
        config = strategy_map.get(strategy_type, strategy_map["2"])
        
        return StrategyConfig(
            name=config["name"],
            strategy_type=config["type"],
            parameters=config["parameters"],
            entry_rules=config["entry_rules"],
            exit_rules=config["exit_rules"],
            risk_params=config["risk_params"]
        )
    
    async def run_backtest(
        self,
        strategy: str,
        symbol: str,
        timeframe: str,
        lookback_days: int,
        capital: float
    ) -> Dict[str, Any]:
        """
        Run strategy backtest
        
        Args:
            strategy: Strategy type (1-4 from user selection)
            symbol: Trading symbol
            timeframe: Candle timeframe
            lookback_days: Days to backtest
            capital: Initial capital
            
        Returns:
            Backtest results
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Daedalus agent not initialized"}
        
        try:
            # Fetch historical data from cache
            logger.info(f"Fetching historical data for {symbol} at {timeframe}")
            data = self._fetch_historical_data(symbol, timeframe, lookback_days)
            
            # Create strategy configuration
            logger.info(f"Creating strategy configuration for type {strategy}")
            strategy_config = self._create_strategy_config(strategy)
            
            # Run simulation using DaedalusAgent
            logger.info("Running backtest simulation...")
            result = self.agent_instance.run_simulation(
                strategy=strategy_config,
                data=data,
                initial_capital=capital
            )
            
            # Convert SimulationResult to dict for response
            return {
                "strategy": result.strategy_name,
                "strategy_id": result.strategy_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback_days": lookback_days,
                "initial_capital": capital,
                "total_trades": result.total_trades,
                "winning_trades": int(result.total_trades * result.win_rate),
                "losing_trades": int(result.total_trades * (1 - result.win_rate)),
                "win_rate": result.win_rate,
                "total_return": result.total_return,
                "annual_return": result.annual_return,
                "annual_volatility": result.annual_volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "max_drawdown": result.max_drawdown,
                "calmar_ratio": result.calmar_ratio,
                "profit_factor": result.profit_factor,
                "avg_trade_duration": result.avg_trade_duration,
                "consistency_score": result.consistency_score,
                "equity_curve": result.equity_curve,
                "trades": result.trades
            }
            
        except Exception as e:
            logger.error(f"Backtest error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def optimize_strategy(
        self,
        strategy: str,
        symbol: str,
        timeframe: str,
        lookback_days: int,
        param_space: Dict[str, tuple],
        method: str = "genetic"
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters
        
        Args:
            strategy: Strategy type
            symbol: Trading symbol
            timeframe: Candle timeframe
            lookback_days: Days to backtest
            param_space: Parameter space for optimization
            method: Optimization method (grid, random, genetic)
            
        Returns:
            Optimization results
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Daedalus agent not initialized"}
        
        try:
            # Fetch historical data
            data = self._fetch_historical_data(symbol, timeframe, lookback_days)
            
            # Create strategy configuration
            strategy_config = self._create_strategy_config(strategy)
            
            # Run optimization
            logger.info(f"Running {method} optimization...")
            task = self.agent_instance.optimize_strategy(
                strategy=strategy_config,
                data=data,
                param_space=param_space,
                method=method,
                metric="sharpe_ratio"
            )
            
            return {
                "task_id": task.task_id,
                "status": task.status.value,
                "best_params": task.best_params,
                "best_score": task.best_score,
                "method": method,
                "metric": "sharpe_ratio"
            }
            
        except Exception as e:
            logger.error(f"Optimization error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def monte_carlo_forecast(
        self,
        strategy_result_id: str,
        n_days: int = 252,
        n_paths: int = 10000
    ) -> Dict[str, Any]:
        """
        Generate Monte Carlo forecast for a strategy
        
        Args:
            strategy_result_id: ID of previous simulation result
            n_days: Number of days to forecast
            n_paths: Number of Monte Carlo paths
            
        Returns:
            Monte Carlo forecast results
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Daedalus agent not initialized"}
        
        try:
            # Get strategy result from memory
            result = self.agent_instance.memory.get_result(strategy_result_id)
            
            if not result:
                return {"error": f"Strategy result {strategy_result_id} not found"}
            
            # Run Monte Carlo forecast
            logger.info(f"Running Monte Carlo forecast with {n_paths} paths for {n_days} days")
            forecast = self.agent_instance.monte_carlo_forecast(
                strategy_result=result,
                n_days=n_days,
                n_paths=n_paths
            )
            
            # Serialize numpy arrays to lists for JSON response
            forecast_response = {
                "expected_value": float(forecast["expected_value"]),
                "median_value": float(forecast["median_value"]),
                "percentile_5": float(forecast["5th_percentile"]),
                "percentile_95": float(forecast["95th_percentile"]),
                "var_95": float(forecast["var_95"]),
                "cvar_95": float(forecast["cvar_95"]),
                "prob_profit": float(forecast["prob_profit"]),
                "n_days": n_days,
                "n_paths": n_paths
            }
            
            return forecast_response
            
        except Exception as e:
            logger.error(f"Monte Carlo forecast error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def get_past_simulations(
        self,
        limit: int = 10,
        sort_by: str = "sharpe"
    ) -> Dict[str, Any]:
        """
        Get past simulation results
        
        Args:
            limit: Number of results to return
            sort_by: Sort criterion (sharpe, profit, win_rate)
            
        Returns:
            List of past simulation results
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Daedalus agent not initialized"}
        
        try:
            # Get top strategies from memory
            if sort_by == "profit":
                top_results = self.agent_instance.memory.get_top_strategies("profit", limit)
            else:
                top_results = self.agent_instance.memory.get_top_strategies("sharpe", limit)
            
            # Convert to dict format
            results = []
            for result in top_results:
                results.append({
                    "strategy_id": result.strategy_id,
                    "strategy_name": result.strategy_name,
                    "total_return": result.total_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "total_trades": result.total_trades,
                    "profit_factor": result.profit_factor
                })
            
            return {
                "count": len(results),
                "sort_by": sort_by,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error getting past simulations: {e}", exc_info=True)
            return {"error": str(e)}