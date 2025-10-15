import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import hashlib
from enum import Enum

class SimulationStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    name: str
    strategy_type: str  # momentum, mean_reversion, breakout, ml_based, etc.
    parameters: Dict[str, Any]
    entry_rules: List[str]
    exit_rules: List[str]
    risk_params: Dict[str, float]
    timeframe: str = "1D"
    universe: List[str] = field(default_factory=list)
    
    def get_id(self) -> str:
        """Generate unique ID based on strategy configuration"""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


@dataclass
class SimulationResult:
    """Results from a strategy simulation"""
    strategy_id: str
    strategy_name: str
    parameters: Dict[str, Any]
    
    # Performance metrics
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    
    # Advanced metrics
    calmar_ratio: float
    omega_ratio: float
    tail_ratio: float
    consistency_score: float
    
    # Time-based metrics
    annual_return: float
    annual_volatility: float
    
    # Execution metrics
    avg_slippage: float
    avg_commission: float
    
    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)
    status: SimulationStatus = SimulationStatus.COMPLETED


@dataclass
class OptimizationTask:
    """Task for parameter optimization"""
    task_id: str
    strategy_config: StrategyConfig
    param_space: Dict[str, Tuple]  # param_name: (min, max, step)
    optimization_metric: str = "sharpe_ratio"
    method: str = "grid"  # grid, random, genetic, bayesian
    max_iterations: int = 1000
    status: SimulationStatus = SimulationStatus.QUEUED
    best_params: Optional[Dict] = None
    best_score: float = -np.inf