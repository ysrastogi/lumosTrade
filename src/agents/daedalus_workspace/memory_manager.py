import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import uuid
from dataclasses import dataclass, field, asdict

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    name: str
    parameters: Dict[str, Any]
    description: str = ""
    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def get_id(self) -> str:
        """Return unique ID for this strategy"""
        return self.strategy_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class SimulationResult:
    """Results from a strategy simulation"""
    strategy_id: str
    performance_metrics: Dict[str, float]
    trades: List[Dict[str, Any]]
    start_time: datetime
    end_time: datetime
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert datetime objects to strings
        data["start_time"] = self.start_time.isoformat()
        data["end_time"] = self.end_time.isoformat()
        return data

@dataclass
class OptimizationTask:
    """Strategy optimization task"""
    strategy_id: str
    parameters_range: Dict[str, List[Any]]
    objective: str
    status: str = "pending"  # pending, running, completed, failed
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    results: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

# Temporal Memory Events
@dataclass
class MemoryEvent:
    """Base class for temporal memory events"""
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0  # 0-10 scale
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
@dataclass
class TradeEvent(MemoryEvent):
    """Memory event for trades"""
    def __init__(self, trade_data: Dict[str, Any], importance: float = 5.0):
        super().__init__(
            timestamp=datetime.now(),
            event_type="trade",
            content=trade_data,
            importance=importance
        )

@dataclass
class InsightEvent(MemoryEvent):
    """Memory event for market insights"""
    def __init__(self, insight: str, source: str, confidence: float, importance: float = 3.0):
        super().__init__(
            timestamp=datetime.now(),
            event_type="insight",
            content={"insight": insight, "source": source, "confidence": confidence},
            importance=importance
        )

@dataclass
class DecisionEvent(MemoryEvent):
    """Memory event for strategy decisions"""
    def __init__(self, decision: str, rationale: str, strategy_id: str, importance: float = 7.0):
        super().__init__(
            timestamp=datetime.now(),
            event_type="decision",
            content={
                "decision": decision,
                "rationale": rationale,
                "strategy_id": strategy_id
            },
            importance=importance
        )

# Memory system for Daedalus with Chronos integration
class ChronosIntegratedMemory:
    """Enhanced memory storage for Daedalus Agent with Chronos temporal capabilities"""
    
    def __init__(self, memory_path: str = None, retention_days: int = 30):
        # Strategy memory
        self.strategy_library: Dict[str, StrategyConfig] = {}
        self.simulation_results: Dict[str, SimulationResult] = {}
        self.optimization_tasks: Dict[str, OptimizationTask] = {}
        
        # Chronos temporal memory
        self.temporal_events: List[MemoryEvent] = []
        self.memory_path = memory_path
        self.retention_days = retention_days
        
        # Indexing for efficient retrieval
        self.event_type_index: Dict[str, List[str]] = {
            "trade": [],
            "insight": [],
            "decision": [],
            "market": [],
            "system": []
        }
        
        # Memory statistics
        self.stats = {
            "total_events": 0,
            "last_pruned": datetime.now(),
            "last_saved": None
        }
        
        # Load memory if path provided
        if memory_path:
            self.load_memory()
    
    def add_strategy(self, strategy: StrategyConfig) -> str:
        """Add strategy to memory and return its ID"""
        strategy_id = strategy.get_id()
        self.strategy_library[strategy_id] = strategy
        
        # Also record this as a system event in temporal memory
        self.add_event(MemoryEvent(
            event_type="system",
            content={
                "action": "add_strategy",
                "strategy_id": strategy_id,
                "strategy_name": strategy.name
            },
            importance=2.0
        ))
        return strategy_id
        
    def add_result(self, result: SimulationResult) -> str:
        """Add simulation result to memory"""
        result_id = f"{result.strategy_id}_{datetime.now().timestamp()}"
        self.simulation_results[result_id] = result
        
        # Record in temporal memory
        self.add_event(MemoryEvent(
            event_type="system",
            content={
                "action": "add_result",
                "result_id": result_id,
                "strategy_id": result.strategy_id,
                "performance": {
                    "sharpe": result.sharpe_ratio,
                    "sortino": result.sortino_ratio,
                    "return": result.total_return
                }
            },
            importance=4.0
        ))
        return result_id
    
    def add_event(self, event: MemoryEvent) -> str:
        """Add temporal event to memory"""
        self.temporal_events.append(event)
        
        # Update index
        if event.event_type in self.event_type_index:
            self.event_type_index[event.event_type].append(event.event_id)
        else:
            self.event_type_index[event.event_type] = [event.event_id]
        
        self.stats["total_events"] += 1
        return event.event_id
    
    def add_trade_event(self, trade_data: Dict[str, Any], importance: float = 5.0) -> str:
        """Add a trade event to temporal memory"""
        event = TradeEvent(trade_data, importance)
        return self.add_event(event)
    
    def add_insight(self, insight: str, source: str, confidence: float, importance: float = 3.0) -> str:
        """Add a market insight to temporal memory"""
        event = InsightEvent(insight, source, confidence, importance)
        return self.add_event(event)
    
    def add_decision(self, decision: str, rationale: str, strategy_id: str, importance: float = 7.0) -> str:
        """Add a strategic decision to temporal memory"""
        event = DecisionEvent(decision, rationale, strategy_id, importance)
        return self.add_event(event)
        
    def get_top_strategies(self, metric: str = "sharpe", count: int = 5) -> List[SimulationResult]:
        """Get top strategies by metric"""
        results = list(self.simulation_results.values())
        if not results:
            return []
            
        return sorted(results, key=lambda x: getattr(x, f"{metric}_ratio", 0), reverse=True)[:count]
    
    def get_recent_events(self, event_type: Optional[str] = None, hours: int = 24, 
                          min_importance: float = 0.0) -> List[MemoryEvent]:
        """Get recent memory events filtered by type and importance"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        if event_type:
            event_ids = self.event_type_index.get(event_type, [])
            events = [e for e in self.temporal_events 
                     if e.event_id in event_ids 
                     and e.timestamp.timestamp() >= cutoff_time
                     and e.importance >= min_importance]
        else:
            events = [e for e in self.temporal_events 
                     if e.timestamp.timestamp() >= cutoff_time
                     and e.importance >= min_importance]
            
        # Sort by time (recent first) and importance
        return sorted(events, key=lambda x: (x.timestamp.timestamp(), x.importance), reverse=True)
    
    def get_events_by_timeframe(self, start_time: datetime, end_time: datetime,
                               event_type: Optional[str] = None) -> List[MemoryEvent]:
        """Get events within a specific timeframe"""
        start_ts = start_time.timestamp()
        end_ts = end_time.timestamp()
        
        if event_type:
            event_ids = self.event_type_index.get(event_type, [])
            events = [e for e in self.temporal_events 
                     if e.event_id in event_ids 
                     and start_ts <= e.timestamp.timestamp() <= end_ts]
        else:
            events = [e for e in self.temporal_events 
                     if start_ts <= e.timestamp.timestamp() <= end_ts]
                     
        return sorted(events, key=lambda x: x.timestamp.timestamp())
    
    def prune_memory(self, force: bool = False) -> int:
        """Remove old events based on retention policy"""
        if not force:
            # Only prune once per day
            last_prune = self.stats["last_pruned"]
            if (datetime.now() - last_prune).days < 1:
                return 0
        
        retention_cutoff = datetime.now().timestamp() - (self.retention_days * 24 * 3600)
        original_count = len(self.temporal_events)
        
        # Keep only events within retention period
        self.temporal_events = [e for e in self.temporal_events 
                               if e.timestamp.timestamp() >= retention_cutoff]
        
        # Rebuild indices
        self.rebuild_indices()
        
        pruned_count = original_count - len(self.temporal_events)
        if pruned_count > 0:
            logger.info(f"Pruned {pruned_count} events from memory")
            self.stats["last_pruned"] = datetime.now()
            
        return pruned_count
    
    def rebuild_indices(self):
        """Rebuild event indices after pruning"""
        self.event_type_index = {event_type: [] for event_type in self.event_type_index.keys()}
        
        for event in self.temporal_events:
            if event.event_type in self.event_type_index:
                self.event_type_index[event.event_type].append(event.event_id)
            else:
                self.event_type_index[event.event_type] = [event.event_id]
    
    def save_memory(self, path: Optional[str] = None) -> bool:
        """Save memory state to disk"""
        save_path = path or self.memory_path
        if not save_path:
            logger.warning("No memory path provided for saving")
            return False
            
        try:
            # Convert memory to serializable format
            memory_data = {
                "strategies": {k: v.to_dict() for k, v in self.strategy_library.items()},
                "results": {k: v.to_dict() for k, v in self.simulation_results.items()},
                "tasks": {k: v.to_dict() for k, v in self.optimization_tasks.items()},
                "events": [e.to_dict() for e in self.temporal_events],
                "stats": self.stats,
                "metadata": {
                    "version": "1.0",
                    "saved_at": datetime.now().isoformat(),
                    "retention_days": self.retention_days
                }
            }
            
            # Update stats["last_saved"]
            self.stats["last_saved"] = datetime.now()
            memory_data["stats"] = self.stats
            
            with open(save_path, 'w') as f:
                json.dump(memory_data, f, indent=2)
                
            logger.info(f"Memory saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save memory: {str(e)}")
            return False
    
    def load_memory(self, path: Optional[str] = None) -> bool:
        """Load memory state from disk"""
        load_path = path or self.memory_path
        if not load_path:
            logger.warning("No memory path provided for loading")
            return False
            
        try:
            with open(load_path, 'r') as f:
                memory_data = json.load(f)
                
            # Load strategies
            for strategy_id, strategy_dict in memory_data.get("strategies", {}).items():
                self.strategy_library[strategy_id] = StrategyConfig(**strategy_dict)
                
            # Load results
            for result_id, result_dict in memory_data.get("results", {}).items():
                # Convert string dates back to datetime
                result_dict["start_time"] = datetime.fromisoformat(result_dict["start_time"])
                result_dict["end_time"] = datetime.fromisoformat(result_dict["end_time"])
                self.simulation_results[result_id] = SimulationResult(**result_dict)
                
            # Load tasks
            for task_id, task_dict in memory_data.get("tasks", {}).items():
                self.optimization_tasks[task_id] = OptimizationTask(**task_dict)
                
            # Load events
            self.temporal_events = []
            for event_dict in memory_data.get("events", []):
                event = MemoryEvent(
                    timestamp=datetime.fromisoformat(event_dict["timestamp"]),
                    event_type=event_dict["event_type"],
                    content=event_dict["content"],
                    importance=event_dict["importance"],
                    event_id=event_dict["event_id"]
                )
                self.temporal_events.append(event)
                
            # Load stats
            self.stats = memory_data.get("stats", self.stats)
            
            # Rebuild indices
            self.rebuild_indices()
            
            logger.info(f"Memory loaded from {load_path} with {len(self.temporal_events)} events")
            return True
        except Exception as e:
            logger.error(f"Failed to load memory: {str(e)}")
            return False
    
    def export_state(self) -> Dict:
        """Export memory state"""
        return {
            "strategy_count": len(self.strategy_library),
            "simulation_count": len(self.simulation_results),
            "optimization_count": len(self.optimization_tasks),
            "event_count": len(self.temporal_events),
            "event_types": {k: len(v) for k, v in self.event_type_index.items()},
            "memory_size_kb": self._estimate_memory_size() / 1024,
            "retention_days": self.retention_days,
            "last_pruned": self.stats["last_pruned"].isoformat(),
            "last_saved": self.stats["last_saved"].isoformat() if self.stats["last_saved"] else None
        }
    
    def _estimate_memory_size(self) -> int:
        """Estimate memory size in bytes"""
        import sys
        
        # Sample a subset of events to estimate size
        sample_size = min(100, len(self.temporal_events))
        if sample_size == 0:
            return 0
            
        sample_events = self.temporal_events[:sample_size]
        sample_size_bytes = sum(sys.getsizeof(str(e.to_dict())) for e in sample_events)
        
        # Extrapolate to full dataset
        estimated_events_size = (sample_size_bytes / sample_size) * len(self.temporal_events)
        
        # Add size of strategies and results (rough estimate)
        strategies_size = sum(sys.getsizeof(str(v.to_dict())) for v in self.strategy_library.values())
        results_size = sum(sys.getsizeof(str(v.to_dict())) for v in self.simulation_results.values())
        
        return int(estimated_events_size + strategies_size + results_size)


# For backward compatibility
class ChronosMemory:
    """Specialized temporal memory system for Chronos Agent"""
    
    def __init__(self, base_memory: ChronosIntegratedMemory):
        self.memory = base_memory
        self.time_windows = {
            "recent": 24,  # last 24 hours
            "day": 24,     # 1 day window
            "week": 168,   # 7 days window
            "month": 720   # 30 days window
        }
    
    def remember_event(self, event_type: str, content: Dict[str, Any], 
                      importance: float = 1.0) -> str:
        """Add new memory event"""
        event = MemoryEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            content=content,
            importance=importance
        )
        return self.memory.add_event(event)
    
    def remember_market_event(self, symbol: str, event_type: str, 
                             data: Dict[str, Any], importance: float = 3.0) -> str:
        """Add market event memory"""
        content = {
            "symbol": symbol,
            "type": event_type,
            **data
        }
        return self.remember_event("market", content, importance)
    
    def get_timeline(self, start_time: datetime, end_time: datetime, 
                    event_types: List[str] = None) -> List[Dict[str, Any]]:
        """Get timeline of events in chronological order"""
        if event_types:
            all_events = []
            for event_type in event_types:
                all_events.extend(self.memory.get_events_by_timeframe(start_time, end_time, event_type))
        else:
            all_events = self.memory.get_events_by_timeframe(start_time, end_time)
            
        # Sort by timestamp
        timeline = sorted(all_events, key=lambda x: x.timestamp)
        return [e.to_dict() for e in timeline]
    
    def get_significant_events(self, hours: int = 24, threshold: float = 6.0) -> List[Dict[str, Any]]:
        """Get high-importance events from recent memory"""
        events = self.memory.get_recent_events(hours=hours, min_importance=threshold)
        return [e.to_dict() for e in events]
    
    def analyze_trends(self, event_type: str, hours: int = 168, key_field: str = None) -> Dict[str, Any]:
        """Analyze trends in temporal memory
        
        Args:
            event_type: Type of event to analyze
            hours: Time window to analyze
            key_field: Field within content to group by (e.g., "symbol")
            
        Returns:
            Dictionary with trend analysis
        """
        events = self.memory.get_recent_events(event_type=event_type, hours=hours)
        
        if not events:
            return {"status": "no_data"}
        
        # Group by time periods
        hourly_buckets = {}
        daily_buckets = {}
        
        # Group by key_field if provided
        key_field_stats = {}
        
        for event in events:
            # Get hour and day buckets
            hour_bucket = event.timestamp.replace(minute=0, second=0, microsecond=0)
            day_bucket = event.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Count by hour
            hour_key = hour_bucket.isoformat()
            hourly_buckets[hour_key] = hourly_buckets.get(hour_key, 0) + 1
            
            # Count by day
            day_key = day_bucket.isoformat()
            daily_buckets[day_key] = daily_buckets.get(day_key, 0) + 1
            
            # Group by key_field if provided
            if key_field and key_field in event.content:
                field_value = str(event.content[key_field])
                if field_value not in key_field_stats:
                    key_field_stats[field_value] = {"count": 0, "importance_sum": 0}
                    
                key_field_stats[field_value]["count"] += 1
                key_field_stats[field_value]["importance_sum"] += event.importance
        
        # Calculate averages for key_field
        if key_field_stats:
            for k in key_field_stats:
                stats = key_field_stats[k]
                stats["importance_avg"] = stats["importance_sum"] / stats["count"]
            
            # Sort by count
            top_by_count = sorted(
                key_field_stats.items(), 
                key=lambda x: x[1]["count"], 
                reverse=True
            )[:10]
            
            # Sort by importance
            top_by_importance = sorted(
                key_field_stats.items(), 
                key=lambda x: x[1]["importance_avg"], 
                reverse=True
            )[:10]
        else:
            top_by_count = []
            top_by_importance = []
        
        return {
            "status": "success",
            "total_events": len(events),
            "hourly_distribution": hourly_buckets,
            "daily_distribution": daily_buckets,
            "top_by_count": top_by_count if key_field else [],
            "top_by_importance": top_by_importance if key_field else [],
            "time_window_hours": hours
        }
    
    def get_memory_digest(self) -> Dict[str, Any]:
        """Get digest of important memory contents"""
        # Get the last day's significant events
        significant_events = self.get_significant_events(hours=24, threshold=5.0)
        
        # Get latest events by type
        latest_trades = self.memory.get_recent_events(event_type="trade", hours=24)
        latest_insights = self.memory.get_recent_events(event_type="insight", hours=24)
        latest_decisions = self.memory.get_recent_events(event_type="decision", hours=24)
        
        return {
            "significant_events_count": len(significant_events),
            "significant_events": [e.to_dict() for e in significant_events[:5]],  # Top 5 only
            "recent_trades_count": len(latest_trades),
            "recent_insights_count": len(latest_insights),
            "recent_decisions_count": len(latest_decisions),
            "memory_stats": self.memory.stats
        }


class DaedalusMemory(ChronosIntegratedMemory):
    """Legacy class for compatibility, now using ChronosIntegratedMemory under the hood"""
    
    def __init__(self, memory_path: str = None, retention_days: int = 30):
        super().__init__(memory_path, retention_days)
        logger.info("Using enhanced ChronosIntegratedMemory system")


# Example usage
if __name__ == "__main__":
    # Initialize the integrated memory system
    memory_path = "chronos_memory.json"
    memory = ChronosIntegratedMemory(memory_path=memory_path, retention_days=60)
    
    # Create Chronos temporal memory interface
    chronos = ChronosMemory(memory)
    
    # Add sample strategy
    strategy = StrategyConfig(
        name="MACD Crossover",
        parameters={
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "symbols": ["BTC/USD", "ETH/USD"]
        },
        description="MACD crossover strategy for crypto pairs"
    )
    strategy_id = memory.add_strategy(strategy)
    
    # Record market events
    chronos.remember_market_event(
        symbol="BTC/USD",
        event_type="price_spike",
        data={
            "price": 65000,
            "change_percent": 3.5,
            "volume": 1500000000
        },
        importance=6.5
    )
    
    # Record trading decision
    chronos.remember_event(
        event_type="decision",
        content={
            "action": "buy",
            "symbol": "BTC/USD",
            "quantity": 0.5,
            "price": 65000,
            "strategy_id": strategy_id,
            "reason": "MACD bullish crossover detected with increasing volume"
        },
        importance=7.0
    )
    
    # Get memory digest
    digest = chronos.get_memory_digest()
    print(f"Memory digest: {digest}")
    
    # Save memory state
    memory.save_memory()
    
    print("Chronos memory system initialized and tested successfully")