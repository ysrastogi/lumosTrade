from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
from collections import deque
from datetime import datetime, timedelta

class SignalQuality(Enum):
    EXCEPTIONAL = "exceptional"  # >80% confidence
    STRONG = "strong"            # 65-80%
    MODERATE = "moderate"        # 50-65%
    WEAK = "weak"                # <50%

class SignalStatus(Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    STOPPED = "stopped"
    TARGET_HIT = "target_hit"
    EXPIRED = "expired"

@dataclass
class Signal:
    symbol: str
    direction: str
    pattern: str
    entry: float
    stop_loss: float
    confidence: float
    timeframe: str = "1h"
    id: str = None
    timestamp: datetime = None
    time_horizon: str = "short_term"
    description: str = ""
    target: float = None
    risk_reward: float = 0.0
    take_profit: float = None
    reasoning: str = ""
    invalidation_criteria: List[str] = None
    supporting_factors: List[str] = None
    similar_historical_count: int = 0
    historical_win_rate: float = 0.5
    status: SignalStatus = SignalStatus.ACTIVE
    outcome: Optional[str] = None
    pnl: Optional[float] = None
    
    def __post_init__(self):
        # Generate ID if not provided
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())[:8]
            
        # Set timestamp if not provided
        if not self.timestamp:
            self.timestamp = datetime.now()
            
        # Initialize lists if None
        if self.invalidation_criteria is None:
            self.invalidation_criteria = []
            
        if self.supporting_factors is None:
            self.supporting_factors = []
            
        # Calculate risk/reward if target is provided and not already set
        if self.risk_reward == 0.0 and self.target is not None and self.stop_loss is not None:
            risk = abs(self.entry - self.stop_loss)
            if risk > 0:
                reward = abs(self.target - self.entry)
                self.risk_reward = reward / risk
                
        # Set take_profit to target if not set
        if self.take_profit is None and self.target is not None:
            self.take_profit = self.target
    
    def to_dict(self):
        """Convert signal to dictionary for serialization"""
        result = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else datetime.now().isoformat(),
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit if self.take_profit is not None else self.target,
            "confidence": self.confidence * 100 if self.confidence <= 1.0 else self.confidence,
            "pattern": self.pattern,
            "timeframe": self.timeframe
        }
        
        if self.reasoning:
            result["reasoning"] = self.reasoning
            
        if self.invalidation_criteria:
            result["invalidation_criteria"] = self.invalidation_criteria
            
        return result