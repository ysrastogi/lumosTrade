import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from enum import Enum
import logging

from src.agents.dedalus_workspace.models import StrategyConfig, SimulationResult


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLStrategyEvolver:
    """Genetic algorithm for strategy evolution and mutation"""
    
    def __init__(self):
        self.generation_history: List[Dict] = []
    
    def evolve_strategies(
        self,
        base_strategies: List[StrategyConfig],
        data: pd.DataFrame,
        evaluator: Callable,
        n_generations: int = 10,
        population_size: int = 50,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        elitism: int = 5
    ) -> List[StrategyConfig]:
        """
        Evolve trading strategies using genetic algorithms
        
        Args:
            base_strategies: Initial population of strategies
            data: Market data for evaluation
            evaluator: Function to evaluate strategy fitness
            n_generations: Number of generations to evolve
            population_size: Size of population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism: Number of top strategies to preserve
        
        Returns:
            Evolved population of strategies
        """
        logger.info(f"Starting strategy evolution: {n_generations} generations")
        
        # Initialize population
        population = base_strategies.copy()
        
        # Expand to population size
        while len(population) < population_size:
            population.append(self._mutate_strategy(np.random.choice(base_strategies)))
        
        for generation in range(n_generations):
            # Evaluate fitness
            fitness_scores = []
            for strategy in population:
                try:
                    score = evaluator(strategy, data)
                    fitness_scores.append(score)
                except Exception as e:
                    logger.warning(f"Strategy evaluation failed: {e}")
                    fitness_scores.append(-np.inf)
            
            # Sort by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]
            
            best_fitness = fitness_scores[0]
            avg_fitness = np.mean(fitness_scores)
            
            logger.info(f"Generation {generation + 1}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
            
            self.generation_history.append({
                "generation": generation + 1,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "best_strategy": population[0].name
            })
            
            # Create next generation
            next_generation = []
            
            # Elitism: preserve top strategies
            next_generation.extend(population[:elitism])
            
            # Generate offspring
            while len(next_generation) < population_size:
                # Selection (tournament)
                parent1 = self._tournament_selection(population, fitness_scores, k=3)
                parent2 = self._tournament_selection(population, fitness_scores, k=3)
                
                # Crossover
                if np.random.random() < crossover_rate:
                    offspring = self._crossover_strategies(parent1, parent2)
                else:
                    offspring = parent1 if np.random.random() < 0.5 else parent2
                
                # Mutation
                if np.random.random() < mutation_rate:
                    offspring = self._mutate_strategy(offspring)
                
                next_generation.append(offspring)
            
            population = next_generation[:population_size]
        
        # Final evaluation
        final_scores = [evaluator(s, data) for s in population]
        best_idx = np.argmax(final_scores)
        
        logger.info(f"Evolution complete. Best strategy: {population[best_idx].name}")
        
        return population
    
    def _tournament_selection(
        self,
        population: List[StrategyConfig],
        fitness: List[float],
        k: int = 3
    ) -> StrategyConfig:
        """Tournament selection"""
        tournament_idx = np.random.choice(len(population), k, replace=False)
        tournament_fitness = [fitness[i] for i in tournament_idx]
        winner_idx = tournament_idx[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover_strategies(
        self,
        parent1: StrategyConfig,
        parent2: StrategyConfig
    ) -> StrategyConfig:
        """Crossover two strategies"""
        # Combine parameters
        child_params = {}
        for key in parent1.parameters.keys():
            if key in parent2.parameters:
                # Average numeric parameters
                if isinstance(parent1.parameters[key], (int, float)):
                    child_params[key] = (parent1.parameters[key] + parent2.parameters[key]) / 2
                else:
                    # Random choice for non-numeric
                    child_params[key] = np.random.choice([
                        parent1.parameters[key],
                        parent2.parameters[key]
                    ])
            else:
                child_params[key] = parent1.parameters[key]
        
        # Combine rules
        child_entry = list(set(parent1.entry_rules + parent2.entry_rules))
        child_exit = list(set(parent1.exit_rules + parent2.exit_rules))
        
        # Combine risk params
        child_risk = {}
        for key in parent1.risk_params.keys():
            if key in parent2.risk_params:
                child_risk[key] = (parent1.risk_params[key] + parent2.risk_params[key]) / 2
            else:
                child_risk[key] = parent1.risk_params[key]
        
        return StrategyConfig(
            name=f"Evolved_{datetime.now().timestamp()}",
            strategy_type=parent1.strategy_type,
            parameters=child_params,
            entry_rules=child_entry,
            exit_rules=child_exit,
            risk_params=child_risk,
            timeframe=parent1.timeframe,
            universe=parent1.universe
        )
    
    def _mutate_strategy(self, strategy: StrategyConfig) -> StrategyConfig:
        """Mutate strategy parameters"""
        mutated_params = strategy.parameters.copy()
        
        # Mutate random parameter
        if mutated_params:
            param_key = np.random.choice(list(mutated_params.keys()))
            if isinstance(mutated_params[param_key], (int, float)):
                # Add random noise
                noise = np.random.normal(0, 0.1)
                mutated_params[param_key] *= (1 + noise)
        
        # Mutate risk parameters
        mutated_risk = strategy.risk_params.copy()
        if mutated_risk:
            risk_key = np.random.choice(list(mutated_risk.keys()))
            noise = np.random.normal(0, 0.05)
            mutated_risk[risk_key] *= (1 + noise)
        
        return StrategyConfig(
            name=f"{strategy.name}_mutated",
            strategy_type=strategy.strategy_type,
            parameters=mutated_params,
            entry_rules=strategy.entry_rules.copy(),
            exit_rules=strategy.exit_rules.copy(),
            risk_params=mutated_risk,
            timeframe=strategy.timeframe,
            universe=strategy.universe.copy()
        )