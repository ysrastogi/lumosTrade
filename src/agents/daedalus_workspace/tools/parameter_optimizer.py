import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import random
from dataclasses import dataclass
from scipy import stats
from datetime import datetime, timedelta



class ParameterOptimizer:
    """Grid and random search for optimal strategy parameters"""
    
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
    
    async def grid_search(
        self,
        strategy_function: callable,
        parameter_space: Dict[str, List[Any]],
        objective_function: callable,
        max_iterations: Optional[int] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Exhaustive grid search over parameter space
        
        Args:
            strategy_function: Function that runs strategy with given parameters
            parameter_space: Dict of parameter names to list of values to try
            objective_function: Function to evaluate strategy performance
            max_iterations: Maximum combinations to try (for large spaces)
        
        Returns:
            Tuple of (best_parameters, best_score)
        """
        
        # Generate all parameter combinations
        param_combinations = self._generate_grid_combinations(parameter_space)
        
        if max_iterations and len(param_combinations) > max_iterations:
            param_combinations = random.sample(param_combinations, max_iterations)
        
        # Run evaluations in parallel
        tasks = []
        for params in param_combinations:
            task = self._evaluate_parameters(
                strategy_function,
                params,
                objective_function
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Find best result
        best_idx = np.argmax([r[1] for r in results])
        best_params, best_score = results[best_idx]
        
        return best_params, best_score
    
    async def random_search(
        self,
        strategy_function: callable,
        parameter_space: Dict[str, Tuple[Any, Any]],  # (min, max) for each param
        objective_function: callable,
        num_iterations: int = 100
    ) -> Tuple[Dict[str, Any], float]:
        """
        Random search over continuous parameter space
        
        More efficient than grid search for high-dimensional spaces
        """
        
        best_params = None
        best_score = float('-inf')
        
        tasks = []
        for _ in range(num_iterations):
            # Generate random parameters
            params = self._generate_random_parameters(parameter_space)
            task = self._evaluate_parameters(
                strategy_function,
                params,
                objective_function
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Find best result
        best_idx = np.argmax([r[1] for r in results])
        best_params, best_score = results[best_idx]
        
        return best_params, best_score
    
    async def bayesian_optimization(
        self,
        strategy_function: callable,
        parameter_space: Dict[str, Tuple[float, float]],
        objective_function: callable,
        num_iterations: int = 50,
        exploration_weight: float = 2.0
    ) -> Tuple[Dict[str, Any], float]:
        """
        Bayesian optimization using Gaussian Process
        More sample-efficient than random or grid search
        """
        
        # This is a simplified version; full implementation would use sklearn or GPyOpt
        from collections import defaultdict
        
        tested_params = []
        tested_scores = []
        
        # Initial random samples
        for _ in range(min(10, num_iterations // 5)):
            params = self._generate_random_parameters(parameter_space)
            params_tuple, score = await self._evaluate_parameters(
                strategy_function, params, objective_function
            )
            tested_params.append(params_tuple)
            tested_scores.append(score)
        
        # Iterative refinement using Upper Confidence Bound
        for _ in range(num_iterations - len(tested_params)):
            # Select next point using acquisition function (UCB)
            next_params = self._select_next_bayesian_point(
                tested_params,
                tested_scores,
                parameter_space,
                exploration_weight
            )
            
            params_tuple, score = await self._evaluate_parameters(
                strategy_function, next_params, objective_function
            )
            
            tested_params.append(params_tuple)
            tested_scores.append(score)
        
        # Return best found
        best_idx = np.argmax(tested_scores)
        return tested_params[best_idx], tested_scores[best_idx]
    
    def _generate_grid_combinations(self, parameter_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search"""
        import itertools
        
        keys = list(parameter_space.keys())
        values = list(parameter_space.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _generate_random_parameters(self, parameter_space: Dict[str, Tuple[Any, Any]]) -> Dict[str, Any]:
        """Generate random parameters within bounds"""
        params = {}
        for key, (min_val, max_val) in parameter_space.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[key] = random.randint(min_val, max_val)
            elif isinstance(min_val, float) or isinstance(max_val, float):
                params[key] = random.uniform(min_val, max_val)
            else:
                params[key] = random.choice([min_val, max_val])
        
        return params
    
    def _select_next_bayesian_point(
        self,
        tested_params: List[Dict],
        tested_scores: List[float],
        parameter_space: Dict[str, Tuple[float, float]],
        exploration_weight: float
    ) -> Dict[str, Any]:
        """Select next point to sample using UCB acquisition"""
        
        # Simplified: Generate candidates and pick best UCB score
        candidates = [self._generate_random_parameters(parameter_space) for _ in range(100)]
        
        best_ucb = float('-inf')
        best_candidate = candidates[0]
        
        for candidate in candidates:
            ucb = self._calculate_ucb(candidate, tested_params, tested_scores, exploration_weight)
            if ucb > best_ucb:
                best_ucb = ucb
                best_candidate = candidate
        
        return best_candidate
    
    def _calculate_ucb(
        self,
        candidate: Dict,
        tested_params: List[Dict],
        tested_scores: List[float],
        exploration_weight: float
    ) -> float:
        """Calculate Upper Confidence Bound for a candidate"""
        
        # Simplified UCB: mean of nearby points + exploration bonus
        distances = []
        nearby_scores = []
        
        for tested, score in zip(tested_params, tested_scores):
            dist = sum((candidate.get(k, 0) - tested.get(k, 0)) ** 2 for k in candidate.keys()) ** 0.5
            distances.append(dist)
            if dist < 1.0:  # Nearby threshold
                nearby_scores.append(score)
        
        if nearby_scores:
            mean_score = np.mean(nearby_scores)
            std_score = np.std(nearby_scores) if len(nearby_scores) > 1 else 1.0
        else:
            mean_score = 0.0
            std_score = 1.0
        
        min_distance = min(distances) if distances else 1.0
        exploration_bonus = exploration_weight * std_score / (1 + min_distance)
        
        return mean_score + exploration_bonus
    
    async def _evaluate_parameters(
        self,
        strategy_function: callable,
        parameters: Dict[str, Any],
        objective_function: callable
    ) -> Tuple[Dict[str, Any], float]:
        """Evaluate a single parameter configuration"""
        
        loop = asyncio.get_event_loop()
        
        # Run strategy with parameters
        strategy_result = await loop.run_in_executor(
            self.executor,
            strategy_function,
            parameters
        )
        
        # Calculate objective score
        score = await loop.run_in_executor(
            self.executor,
            objective_function,
            strategy_result
        )
        
        return parameters, float(score)
    
    def shutdown(self):
        """Cleanup executor"""
        self.executor.shutdown(wait=True)