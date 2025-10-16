from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime
from src.agents.swift_striker_workspace.models import Order, VenueType

@dataclass
class ExecutionMemory:
    """Persistent execution history and learned patterns"""
    
    # Execution History
    trade_log: List[Dict] = field(default_factory=list)
    
    # Venue Performance
    venue_fill_rates: Dict[VenueType, float] = field(default_factory=dict)
    venue_avg_latency: Dict[VenueType, float] = field(default_factory=dict)
    
    # Slippage Patterns (by hour of day)
    hourly_slippage: Dict[int, List[float]] = field(default_factory=lambda: {h: [] for h in range(24)})
    
    # Failed Orders
    failed_orders: List[Dict] = field(default_factory=list)
    failure_reasons: Dict[str, int] = field(default_factory=dict)
    
    def log_execution(self, order: Order, venue: VenueType, latency_us: float, slippage_bps: float):
        """Record completed execution"""
        execution_record = {
            'timestamp': datetime.now(),
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'filled_qty': order.filled_qty,
            'avg_price': order.avg_fill_price,
            'venue': venue.value,
            'latency_us': latency_us,
            'slippage_bps': slippage_bps,
            'algo': order.algo.value if order.algo else None
        }
        self.trade_log.append(execution_record)
        
        # Update hourly slippage pattern
        hour = datetime.now().hour
        self.hourly_slippage[hour].append(slippage_bps)
    
    def log_failure(self, order: Order, reason: str):
        """Record failed execution"""
        failure_record = {
            'timestamp': datetime.now(),
            'order_id': order.order_id,
            'symbol': order.symbol,
            'reason': reason
        }
        self.failed_orders.append(failure_record)
        self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1
    
    def get_best_venue(self, order: Order) -> VenueType:
        """Select venue based on historical performance"""
        if not self.venue_fill_rates:
            return VenueType.PRIMARY
        
        # Score venues: 70% fill rate, 30% latency
        scores = {}
        for venue in VenueType:
            fill_rate = self.venue_fill_rates.get(venue, 0.95)
            latency = self.venue_avg_latency.get(venue, 1.0)
            scores[venue] = (fill_rate * 0.7) + ((1.0 / latency) * 0.3)
        
        return max(scores, key=scores.get)
    
    def predict_best_execution_time(self, symbol: str) -> int:
        """Suggest optimal hour for execution based on historical slippage"""
        avg_slippage_by_hour = {
            hour: sum(slippages) / len(slippages) if slippages else 999
            for hour, slippages in self.hourly_slippage.items()
        }
        return min(avg_slippage_by_hour, key=avg_slippage_by_hour.get)
    
    def get_execution_stats(self) -> Dict:
        """Calculate overall execution metrics"""
        if not self.trade_log:
            return {}
        
        total_trades = len(self.trade_log)
        avg_latency = sum(t['latency_us'] for t in self.trade_log) / total_trades
        avg_slippage = sum(t['slippage_bps'] for t in self.trade_log) / total_trades
        
        fill_rate = sum(1 for t in self.trade_log if t['filled_qty'] == t['quantity']) / total_trades
        
        return {
            'total_executions': total_trades,
            'avg_latency_us': round(avg_latency, 2),
            'avg_slippage_bps': round(avg_slippage, 2),
            'fill_rate_pct': round(fill_rate * 100, 2),
            'total_failures': len(self.failed_orders),
            'top_failure_reason': max(self.failure_reasons, key=self.failure_reasons.get) if self.failure_reasons else None
        }