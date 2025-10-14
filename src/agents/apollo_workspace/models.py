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
    id: str
    timestamp: datetime
    symbol: str
    pattern: str
    direction: str
    confidence: float
    entry: float
    stop_loss: float
    target: float
    risk_reward: float
    reasoning: str
    invalidation_criteria: List[str]
    supporting_factors: List[str]
    similar_historical_count: int
    historical_win_rate: float
    status: SignalStatus = SignalStatus.ACTIVE
    outcome: Optional[str] = None
    pnl: Optional[float] = None