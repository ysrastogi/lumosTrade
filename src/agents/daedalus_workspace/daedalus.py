import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import random
from dataclasses import dataclass, asdict
from scipy import stats
from datetime import datetime, timedelta
import pandas as pd
import logging

from src.llm.client import GeminiClient, gemini
from src.agents.daedalus_workspace.models import (
    StrategyConfig,
    SimulationResult,
    OptimizationTask,
    SimulationStatus
)

from src.agents.daedalus_workspace.prompts import (
    SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_PROMPT, 
    INTENT_KEYWORDS,
    VALID_INTENTS,
    STRATEGY_EXTRACTION_PROMPT,
    DEFAULT_STRATEGY,
    EXPLANATION_PROMPT,
    NEXT_ACTIONS_PROMPT,
    STRATEGY_IDEAS_PROMPT,
    MARKET_PARAMS_PROMPT,
    FORECAST_PARAMS_PROMPT
)
from src.agents.daedalus_workspace.prompts.base import (
    format_prompt, 
    TEMPERATURE_PRESETS,
    TOKEN_LIMITS
)

from src.agents.daedalus_workspace.tools.monte_carlo_engine import MonteCarloEngine
from src.agents.daedalus_workspace.tools.strategy_evolver import MLStrategyEvolver
from src.agents.daedalus_workspace.tools.parameter_optimizer import ParameterOptimizer
from src.agents.daedalus_workspace.tools.walk_forward_analyzer import WalkForwardAnalyzer
from src.agents.daedalus_workspace.tools.portfolio_rebalancer import PortfolioRebalancer
from src.agents.daedalus_workspace.tools.scenario_generator import ScenarioGenerator
from src.agents.daedalus_workspace.tools.execution_simulator import ExecutionSimulator
from src.agents.daedalus_workspace.memory_manager import DaedalusMemory

logger = logging.getLogger(__name__)



class DaedalusAgent:
    """Main DAEDALUS agent orchestrator"""
    
    def __init__(self):
        self.memory = DaedalusMemory()
        self.monte_carlo = MonteCarloEngine(n_simulations=10000)
        self.scenario_gen = ScenarioGenerator()
        self.execution_sim = ExecutionSimulator()
        
        logger.info("DAEDALUS Agent initialized - The Architect of Possibilities")
    
    def run_simulation(
        self,
        strategy: StrategyConfig,
        data: pd.DataFrame,
        initial_capital: float = 100000
    ) -> SimulationResult:
        """Run a complete strategy simulation"""
        strategy_id = strategy.get_id()
        
        # Placeholder for actual backtest - would integrate with Backtesting.py
        # This demonstrates the structure
        returns = data['close'].pct_change().dropna()
        
        # Calculate metrics (simplified for demonstration)
        total_return = (1 + returns).prod() - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        sortino = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Consistency score based on rolling Sharpe stability
        rolling_sharpe = returns.rolling(63).mean() / returns.rolling(63).std() * np.sqrt(252)
        consistency = 1 / (1 + rolling_sharpe.std()) if not rolling_sharpe.isna().all() else 0
        
        result = SimulationResult(
            strategy_id=strategy_id,
            strategy_name=strategy.name,
            parameters=strategy.parameters,
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=0.55,  # Placeholder
            profit_factor=1.5,  # Placeholder
            total_trades=100,  # Placeholder
            avg_trade_duration=5.0,  # Placeholder
            calmar_ratio=total_return / abs(max_dd) if max_dd != 0 else 0,
            omega_ratio=1.2,  # Placeholder
            tail_ratio=0.8,  # Placeholder
            consistency_score=consistency,
            annual_return=total_return * (252 / len(returns)),
            annual_volatility=returns.std() * np.sqrt(252),
            avg_slippage=0.0005,
            avg_commission=0.001,
            equity_curve=cumulative.tolist(),
            trades=[]
        )
        
        self.memory.add_result(result)
        return result
    
    def optimize_strategy(
        self,
        strategy: StrategyConfig,
        data: pd.DataFrame,
        param_space: Dict[str, Tuple],
        method: str = "genetic",
        metric: str = "sharpe_ratio"
    ) -> OptimizationTask:
        """Optimize strategy parameters"""
        task_id = f"opt_{strategy.get_id()}_{datetime.now().timestamp()}"
        
        task = OptimizationTask(
            task_id=task_id,
            strategy_config=strategy,
            param_space=param_space,
            optimization_metric=metric,
            method=method
        )
        
        self.memory.optimization_tasks[task_id] = task
        
        def evaluator(params):
            # Update strategy with new params
            test_strategy = StrategyConfig(
                name=strategy.name,
                strategy_type=strategy.strategy_type,
                parameters=params,
                entry_rules=strategy.entry_rules,
                exit_rules=strategy.exit_rules,
                risk_params=strategy.risk_params,
                timeframe=strategy.timeframe,
                universe=strategy.universe
            )
            
            result = self.run_simulation(test_strategy, data)
            return getattr(result, metric)
        
        optimizer = ParameterOptimizer(evaluator)
        
        task.status = SimulationStatus.RUNNING
        
        if method == "grid":
            best_params, best_score, _ = optimizer.grid_search(param_space, max_iterations=200)
        elif method == "random":
            best_params, best_score, _ = optimizer.random_search(param_space, n_iterations=100)
        elif method == "genetic":
            best_params, best_score, _ = optimizer.genetic_algorithm(param_space)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        task.best_params = best_params
        task.best_score = best_score
        task.status = SimulationStatus.COMPLETED
        
        logger.info(f"Optimization complete: Best {metric} = {best_score:.4f}")
        
        return task
    
    def walk_forward_analysis(
        self,
        strategy: StrategyConfig,
        data: pd.DataFrame,
        param_space: Dict[str, Tuple]
    ) -> Dict:
        """Perform walk-forward analysis"""
        wfa = WalkForwardAnalyzer()
        
        def strategy_func(data_slice, params):
            test_strategy = StrategyConfig(
                name=strategy.name,
                strategy_type=strategy.strategy_type,
                parameters=params,
                entry_rules=strategy.entry_rules,
                exit_rules=strategy.exit_rules,
                risk_params=strategy.risk_params
            )
            result = self.run_simulation(test_strategy, data_slice)
            return result.sharpe_ratio
        
        return wfa.analyze(data, strategy_func, param_space)
    
    def stress_test_strategy(
        self,
        strategy: StrategyConfig,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Stress test strategy across scenarios"""
        def strategy_func(data_slice, params):
            result = self.run_simulation(strategy, data_slice)
            return result.sharpe_ratio
        
        return self.scenario_gen.stress_test(strategy_func, data, strategy.parameters)
    
    def monte_carlo_forecast(
        self,
        strategy_result: SimulationResult,
        n_days: int = 252,
        n_paths: int = 10000
    ) -> Dict[str, Any]:
        """Generate Monte Carlo forecast for strategy"""
        paths = self.monte_carlo.simulate_paths(
            initial_capital=100000,
            expected_return=strategy_result.annual_return,
            volatility=strategy_result.annual_volatility,
            n_days=n_days
        )
        
        final_values = paths[:, -1]
        
        return {
            "paths": paths,
            "expected_value": np.mean(final_values),
            "median_value": np.median(final_values),
            "5th_percentile": np.percentile(final_values, 5),
            "95th_percentile": np.percentile(final_values, 95),
            "var_95": self.monte_carlo.calculate_var(paths, 0.95),
            "cvar_95": self.monte_carlo.calculate_cvar(paths, 0.95),
            "prob_profit": (final_values > 100000).mean()
        }
    
    def get_state_report(self) -> Dict:
        """Get comprehensive state report"""
        state = self.memory.export_state()
        
        # Add top strategies details
        top_sharpe = self.memory.get_top_strategies("sharpe", 5)
        top_profit = self.memory.get_top_strategies("profit", 5)
        
        state["top_strategies"] = {
            "by_sharpe": [
                {
                    "name": r.strategy_name,
                    "sharpe": r.sharpe_ratio,
                    "return": r.total_return,
                    "max_dd": r.max_drawdown
                }
                for r in top_sharpe
            ],
            "by_profit": [
                {
                    "name": r.strategy_name,
                    "return": r.total_return,
                    "sharpe": r.sharpe_ratio,
                    "max_dd": r.max_drawdown
                }
                for r in top_profit
            ]
        }
        
        return state
    
    def batch_simulate(
        self,
        strategies: List[StrategyConfig],
        data: pd.DataFrame,
        n_workers: int = 4
    ) -> List[SimulationResult]:
        """Run multiple strategies in parallel"""
        logger.info(f"Running batch simulation for {len(strategies)} strategies")
        
        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(self.run_simulation, strategy, data)
                for strategy in strategies
            ]
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Simulation failed: {e}")
        
        logger.info(f"Batch simulation complete: {len(results)} successful")
        return results
    

"""
DAEDALUS LLM Integration Layer
Natural language interface for strategy simulation
"""




class DaedalusLLM:
    """LLM-powered natural language interface for DAEDALUS"""
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.agent = DaedalusAgent()
        self.client = gemini  # Use the pre-initialized GeminiClient
        self.conversation_history: List[Dict] = []
        
        # System prompt - loaded from the prompts module
        self.system_prompt = SYSTEM_PROMPT
    
    def process_query(self, user_query: str, data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Process natural language query and execute appropriate actions
        
        Args:
            user_query: Natural language query from user
            data: Market data (pandas DataFrame)
        
        Returns:
            Response with results and explanations
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_query
        })
        
        # Intent classification
        intent = self._classify_intent(user_query)
        
        response = {
            "intent": intent,
            "timestamp": datetime.now().isoformat(),
            "results": None,
            "explanation": "",
            "next_actions": []
        }
        
        try:
            # Execute the appropriate handler based on intent
            if intent == "create_strategy":
                results = self._handle_create_strategy(user_query)
                response["results"] = results
            
            elif intent == "simulate":
                results = self._handle_simulate(user_query, data)
                response["results"] = results
            
            elif intent == "optimize":
                results = self._handle_optimize(user_query, data)
                response["results"] = results
            
            elif intent == "analyze":
                results = self._handle_analyze(user_query, data)
                response["results"] = results
            
            elif intent == "forecast":
                results = self._handle_forecast(user_query)
                response["results"] = results
            
            elif intent == "compare":
                results = self._handle_compare(user_query, data)
                response["results"] = results
            
            elif intent == "status":
                results = self.agent.get_state_report()
                response["results"] = results
                
            elif intent == "generate_ideas":
                # Extract parameters from query using Gemini
                try:
                    prompt = format_prompt(MARKET_PARAMS_PROMPT, query=user_query)
                    
                    response_text = self.client.generate(
                        prompt=prompt,
                        temperature=TEMPERATURE_PRESETS["extraction"],
                        max_output_tokens=TOKEN_LIMITS["extraction"]
                    )
                    
                    import json
                    try:
                        params = json.loads(response_text)
                        market_conditions = params.get("market_conditions", "current volatile market")
                        asset_class = params.get("asset_class", "equities")
                        risk_appetite = params.get("risk_appetite", "medium")
                    except:
                        market_conditions = "current volatile market"
                        asset_class = "equities"
                        risk_appetite = "medium"
                        
                except Exception as e:
                    logger.error(f"Error extracting parameters with Gemini: {e}")
                    market_conditions = "current volatile market"
                    asset_class = "equities"
                    risk_appetite = "medium"
                
                # Generate strategy ideas
                results = self.generate_strategy_ideas(
                    market_conditions=market_conditions,
                    asset_class=asset_class,
                    risk_appetite=risk_appetite
                )
                response["results"] = {
                    "ideas": results,
                    "market_conditions": market_conditions,
                    "asset_class": asset_class,
                    "risk_appetite": risk_appetite
                }
            
            else:
                results = None
                response["explanation"] = "I'm not sure how to help with that. Try asking about creating, simulating, or optimizing trading strategies."
                
            # Use Gemini to generate a more detailed explanation if we have results
            if results and not response["explanation"]:
                try:
                    prompt = format_prompt(EXPLANATION_PROMPT, 
                                           query=user_query,
                                           intent=intent,
                                           results=str(results))
                    
                    gemini_explanation = self.client.generate(
                        prompt=prompt,
                        temperature=TEMPERATURE_PRESETS["explanation"],
                        max_output_tokens=TOKEN_LIMITS["explanation"]
                    )
                    
                    response["explanation"] = gemini_explanation.strip()
                    
                    # Also suggest next actions using Gemini
                    next_actions_prompt = format_prompt(NEXT_ACTIONS_PROMPT,
                                                        query=user_query,
                                                        intent=intent,
                                                        results=str(results))
                    
                    next_actions = self.client.generate(
                        prompt=next_actions_prompt,
                        temperature=0.4,
                        max_output_tokens=TOKEN_LIMITS["explanation"]
                    )
                    
                    # Try to parse as JSON array
                    import json
                    try:
                        actions_list = json.loads(next_actions)
                        if isinstance(actions_list, list):
                            response["next_actions"] = actions_list
                    except:
                        # If parsing fails, try simple splitting
                        response["next_actions"] = [a.strip() for a in next_actions.split(",")]
                        
                except Exception as e:
                    logger.error(f"Error generating explanation with Gemini: {e}")
                    # Fall back to simple explanations
                    if intent == "create_strategy":
                        response["explanation"] = f"Created strategy '{results['strategy']['name']}' of type {results['strategy']['strategy_type']}."
                    elif intent == "simulate":
                        response["explanation"] = f"Simulated strategy '{results['strategy_name']}' with Sharpe ratio {results['metrics']['sharpe_ratio']:.2f} and return {results['metrics']['total_return']*100:.1f}%."
                    elif intent == "optimize":
                        response["explanation"] = f"Optimized strategy with {results['method']} method. Best parameters: {results['best_parameters']} with score {results['best_score']:.2f}."
                    else:
                        response["explanation"] = f"Successfully processed your {intent} request."
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            response["explanation"] = f"Error: {str(e)}"
            response["results"] = None
    
    def _classify_intent(self, query: str) -> str:
        """Classify user intent from query using Gemini"""
        query_lower = query.lower()
        
        # First try rule-based approach using imported keywords
        for intent, keywords in INTENT_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return intent
        
        # If rule-based approach fails, use Gemini for more advanced intent detection
        prompt = format_prompt(INTENT_CLASSIFICATION_PROMPT, query=query)
        
        try:
            response = self.client.generate(
                prompt=prompt, 
                temperature=TEMPERATURE_PRESETS["classification"],
                max_output_tokens=TOKEN_LIMITS["classification"]
            ).strip().lower()
            
            # Validate the response
            if response in VALID_INTENTS:
                return response
            else:
                logger.warning(f"Gemini returned invalid intent: {response}")
        except Exception as e:
            logger.error(f"Error using Gemini for intent classification: {e}")
            
        return "unknown"
    
    def _handle_create_strategy(self, query: str) -> Dict:
        """Parse query and create strategy configuration using Gemini"""
        # Use Gemini to extract strategy parameters from the query
        prompt = format_prompt(STRATEGY_EXTRACTION_PROMPT, query=query)
        
        try:
            # Get strategy parameters from Gemini
            response = self.client.generate(
                prompt=prompt,
                temperature=TEMPERATURE_PRESETS["extraction"],
                max_output_tokens=TOKEN_LIMITS["extraction"]
            )
            
            # Try to parse the response as JSON
            import json
            try:
                strategy_data = json.loads(response)
                logger.info(f"Successfully extracted strategy parameters using Gemini")
            except json.JSONDecodeError:
                # If parsing fails, fall back to simple extraction
                logger.warning("Failed to parse Gemini response as JSON, using fallback extraction")
                strategy_data = {
                    "name": self._extract_strategy_name(query) or DEFAULT_STRATEGY["name"],
                    "strategy_type": self._extract_strategy_type(query) or DEFAULT_STRATEGY["strategy_type"],
                    "parameters": self._extract_parameters(query) or DEFAULT_STRATEGY["parameters"],
                    "entry_rules": DEFAULT_STRATEGY["entry_rules"],
                    "exit_rules": DEFAULT_STRATEGY["exit_rules"],
                    "risk_params": DEFAULT_STRATEGY["risk_params"]
                }
                
            strategy = StrategyConfig(
                name=strategy_data["name"],
                strategy_type=strategy_data["strategy_type"],
                parameters=strategy_data["parameters"],
                entry_rules=strategy_data["entry_rules"],
                exit_rules=strategy_data["exit_rules"],
                risk_params=strategy_data["risk_params"]
            )
            
        except Exception as e:
            logger.error(f"Error using Gemini for strategy creation: {e}")
            # Fallback to default strategy
            strategy = StrategyConfig(
                name=DEFAULT_STRATEGY["name"],
                strategy_type=DEFAULT_STRATEGY["strategy_type"],
                parameters=DEFAULT_STRATEGY["parameters"],
                entry_rules=DEFAULT_STRATEGY["entry_rules"],
                exit_rules=DEFAULT_STRATEGY["exit_rules"],
                risk_params=DEFAULT_STRATEGY["risk_params"]
            )
        
        strategy_id = self.agent.memory.add_strategy(strategy)
        
        return {
            "strategy_id": strategy_id,
            "strategy": asdict(strategy),
            "status": "created"
        }
    
    def _extract_strategy_name(self, query: str) -> Optional[str]:
        """Extract strategy name from query"""
        # Simple rule-based extraction
        query_lower = query.lower()
        if "name" in query_lower:
            parts = query.split("name")
            if len(parts) > 1:
                name_part = parts[1].strip()
                if ":" in name_part:
                    name_part = name_part.split(":", 1)[1]
                if '"' in name_part:
                    return name_part.split('"')[1]
                elif "'" in name_part:
                    return name_part.split("'")[1]
                else:
                    # Take first few words as name
                    words = name_part.split()
                    return " ".join(words[:3])
        return None
    
    def _extract_strategy_type(self, query: str) -> Optional[str]:
        """Extract strategy type from query"""
        query_lower = query.lower()
        types = {
            "momentum": ["momentum", "trend", "moving average", "ma"],
            "mean_reversion": ["mean reversion", "reversion", "rsi", "overbought", "oversold"],
            "trend_following": ["trend following", "trend", "following", "adx"],
            "breakout": ["breakout", "break out", "range break", "channel"],
            "statistical_arbitrage": ["statistical arbitrage", "stat arb", "pair", "correlation"],
            "options_strategy": ["option", "call", "put", "strike", "expiry"]
        }
        
        for type_name, keywords in types.items():
            if any(kw in query_lower for kw in keywords):
                return type_name
                
        return None
        
    def _extract_parameters(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract parameters from query"""
        params = {}
        query_lower = query.lower()
        
        # Common parameters with regex
        import re
        
        # Period parameters
        period_match = re.search(r'period[s]?[\s:]?(\d+)', query_lower)
        if period_match:
            params["period"] = int(period_match.group(1))
            
        # Moving average parameters
        fast_ma_match = re.search(r'fast[\s:]?(\d+)', query_lower)
        if fast_ma_match:
            params["fast"] = int(fast_ma_match.group(1))
            
        slow_ma_match = re.search(r'slow[\s:]?(\d+)', query_lower)
        if slow_ma_match:
            params["slow"] = int(slow_ma_match.group(1))
            
        # RSI parameters
        rsi_match = re.search(r'rsi[\s:]?(\d+)', query_lower)
        if rsi_match:
            params["rsi_period"] = int(rsi_match.group(1))
            
        # Threshold parameters
        upper_match = re.search(r'upper[\s:]?(\d+)', query_lower)
        if upper_match:
            params["upper_threshold"] = int(upper_match.group(1))
            
        lower_match = re.search(r'lower[\s:]?(\d+)', query_lower)
        if lower_match:
            params["lower_threshold"] = int(lower_match.group(1))
            
        return params if params else None
    
    def _handle_simulate(self, query: str, data) -> Dict:
        """Handle simulation request"""
        # Get latest strategy or create default
        strategies = list(self.agent.memory.strategy_library.values())
        
        if not strategies:
            # Create default strategy
            default_strategy = StrategyConfig(
                name="Default_MA_Strategy",
                strategy_type="momentum",
                parameters={"fast": 20, "slow": 50},
                entry_rules=["ma_cross_up"],
                exit_rules=["ma_cross_down"],
                risk_params={"position_size": 0.1, "stop_loss": 0.02}
            )
            self.agent.memory.add_strategy(default_strategy)
            strategy = default_strategy
        else:
            strategy = strategies[-1]
        
        result = self.agent.run_simulation(strategy, data)
        
        return {
            "strategy_id": result.strategy_id,
            "strategy_name": result.strategy_name,
            "metrics": {
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades
            },
            "status": "completed"
        }
    
    def _handle_optimize(self, query: str, data) -> Dict:
        """Handle optimization request"""
        strategies = list(self.agent.memory.strategy_library.values())
        if not strategies:
            return {"error": "No strategy available. Please create a strategy first."}
        
        strategy = strategies[-1]
        
        # Define parameter space (could be extracted from query)
        param_space = {
            "fast": (10, 50, 5),
            "slow": (30, 100, 10)
        }
        
        # Extract optimization method from query
        method = "genetic"
        if "grid" in query.lower():
            method = "grid"
        elif "random" in query.lower():
            method = "random"
        
        task = self.agent.optimize_strategy(strategy, data, param_space, method=method)
        
        return {
            "task_id": task.task_id,
            "method": method,
            "best_parameters": task.best_params,
            "best_score": task.best_score,
            "status": task.status.value
        }
    
    def _handle_analyze(self, query: str, data) -> Dict:
        """Handle analysis request"""
        strategies = list(self.agent.memory.strategy_library.values())
        if not strategies:
            return {"error": "No strategy available."}
        
        strategy = strategies[-1]
        
        if "walk forward" in query.lower():
            param_space = {"fast": (10, 50, 5), "slow": (30, 100, 10)}
            results = self.agent.walk_forward_analysis(strategy, data, param_space)
            return {
                "analysis_type": "walk_forward",
                "results": results
            }
        
        elif "stress" in query.lower():
            results = self.agent.stress_test_strategy(strategy, data)
            return {
                "analysis_type": "stress_test",
                "results": results
            }
        
        return {"error": "Unknown analysis type"}
    
    def _handle_forecast(self, query: str) -> Dict:
        """Handle Monte Carlo forecast"""
        results = list(self.agent.memory.simulation_results.values())
        if not results:
            return {"error": "No simulation results available."}
            
        # Use the most recent simulation result
        result = results[-1]
        
        # Extract forecast parameters from query using Gemini
        try:
            prompt = format_prompt(FORECAST_PARAMS_PROMPT, query=query)
            
            response = self.client.generate(
                prompt=prompt,
                temperature=TEMPERATURE_PRESETS["extraction"],
                max_output_tokens=TOKEN_LIMITS["extraction"]
            )
            
            import json
            try:
                params = json.loads(response)
                n_days = params.get("n_days", 252)
                n_paths = params.get("n_paths", 10000)
            except:
                n_days = 252
                n_paths = 10000
                
        except Exception as e:
            logger.error(f"Error using Gemini for forecast parameters: {e}")
            n_days = 252
            n_paths = 10000
        
        forecast = self.agent.monte_carlo_forecast(result, n_days=n_days, n_paths=n_paths)
        
        return {
            "forecast": forecast,
            "strategy_name": result.strategy_name,
            "n_days": n_days,
            "n_paths": n_paths
        }
        
    def _handle_compare(self, query: str, data) -> Dict:
        """Handle strategy comparison"""
        strategies = list(self.agent.memory.strategy_library.values())
        
        if not strategies:
            return {"error": "No strategies available to compare."}
            
        if len(strategies) == 1:
            # Create a variation of the existing strategy for comparison
            orig_strategy = strategies[0]
            
            # Create a modified version with slightly different parameters
            modified_params = dict(orig_strategy.parameters)
            for key in modified_params:
                if isinstance(modified_params[key], (int, float)):
                    modified_params[key] *= 1.2  # Increase by 20%
            
            new_strategy = StrategyConfig(
                name=f"{orig_strategy.name}_Variant",
                strategy_type=orig_strategy.strategy_type,
                parameters=modified_params,
                entry_rules=orig_strategy.entry_rules,
                exit_rules=orig_strategy.exit_rules,
                risk_params=orig_strategy.risk_params
            )
            
            strategies = [orig_strategy, new_strategy]
            
        # Run simulations in parallel
        results = self.agent.batch_simulate(strategies, data)
        
        # Find the best strategy based on Sharpe ratio
        best_strategy = max(results, key=lambda x: x.sharpe_ratio)
        
        comparison_data = []
        for result in results:
            comparison_data.append({
                "name": result.strategy_name,
                "sharpe": result.sharpe_ratio,
                "return": result.total_return,
                "max_dd": result.max_drawdown,
                "win_rate": result.win_rate,
                "consistency": result.consistency_score
            })
            
        return {
            "strategies": comparison_data,
            "best_strategy": best_strategy.strategy_name,
            "metrics": ["sharpe", "return", "max_dd", "win_rate", "consistency"]
        }
    
    def generate_strategy_ideas(self, market_conditions: str, asset_class: str, risk_appetite: str) -> List[Dict]:
        """
        Generate trading strategy ideas based on market conditions using Gemini
        
        Args:
            market_conditions: Description of current market conditions
            asset_class: Asset class to focus on (equities, forex, crypto, etc.)
            risk_appetite: Risk tolerance level (low, medium, high)
            
        Returns:
            List of strategy ideas with parameters
        """
        prompt = format_prompt(STRATEGY_IDEAS_PROMPT, 
                               market_conditions=market_conditions,
                               asset_class=asset_class,
                               risk_appetite=risk_appetite)
        
        try:
            response = self.client.generate(
                prompt=prompt,
                temperature=TEMPERATURE_PRESETS["creative"],
                max_output_tokens=TOKEN_LIMITS["strategy_ideas"]
            )
            
            import json
            try:
                strategies = json.loads(response)
                if isinstance(strategies, list):
                    return strategies
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini strategy ideas response: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating strategy ideas with Gemini: {e}")
            return []
        
        return []