# Setup logging
logger = logging.getLogger(__name__)

# Memory system for Daedalus
class DaedalusMemory:
    """Memory storage for Daedalus Agent"""
    
    def __init__(self):
        self.strategy_library: Dict[str, StrategyConfig] = {}
        self.simulation_results: Dict[str, SimulationResult] = {}
        self.optimization_tasks: Dict[str, OptimizationTask] = {}
        
    def add_strategy(self, strategy: StrategyConfig) -> str:
        """Add strategy to memory and return its ID"""
        strategy_id = strategy.get_id()
        self.strategy_library[strategy_id] = strategy
        return strategy_id
        
    def add_result(self, result: SimulationResult) -> str:
        """Add simulation result to memory"""
        result_id = f"{result.strategy_id}_{datetime.now().timestamp()}"
        self.simulation_results[result_id] = result
        return result_id
        
    def get_top_strategies(self, metric: str = "sharpe", count: int = 5) -> List[SimulationResult]:
        """Get top strategies by metric"""
        results = list(self.simulation_results.values())
        if not results:
            return []
            
        return sorted(results, key=lambda x: getattr(x, f"{metric}_ratio", 0), reverse=True)[:count]
        
    def export_state(self) -> Dict:
        """Export memory state"""
        return {
            "strategy_count": len(self.strategy_library),
            "simulation_count": len(self.simulation_results),
            "optimization_count": len(self.optimization_tasks)
        }