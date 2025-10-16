from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import statistics
from collections import defaultdict

class AgentSignal(Enum):
    """Possible agent signals/decisions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    WAIT = "wait"
    REDUCE = "reduce"
    INCREASE = "increase"
    EMERGENCY = "emergency"


class ConflictType(Enum):
    """Types of agent conflicts"""
    BINARY_SPLIT = "binary_split"  # 50/50 split
    MAJORITY_MINORITY = "majority_minority"  # Clear majority
    THREE_WAY = "three_way"  # No clear majority
    CONFIDENCE_MISMATCH = "confidence_mismatch"  # Same signal, different confidence
    CRITICAL_OVERRIDE = "critical_override"  # Emergency situation


@dataclass
class AgentVote:
    """Individual agent's vote/signal"""
    agent_name: str
    signal: AgentSignal
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """Final consensus decision"""
    decision: AgentSignal
    confidence: float
    method: str  # weighted_vote, unanimous, override, etc.
    participating_agents: List[str]
    dissenting_agents: List[str]
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    vote_breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictResolution:
    """Record of how a conflict was resolved"""
    conflict_type: ConflictType
    resolution_method: str
    original_votes: List[AgentVote]
    final_decision: ConsensusResult
    resolution_reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)