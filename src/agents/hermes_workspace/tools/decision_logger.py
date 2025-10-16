from typing import List, Dict
from collections import defaultdict
from src.agents.hermes_workspace.models import AgentVote, ConsensusResult, ConflictResolution

class DecisionLogger:
    """Audit trail of all consensus decisions"""
    
    def __init__(self, max_history: int = 1000):
        self.decisions = []
        self.conflicts = []
        self.max_history = max_history
    
    def log_decision(self, consensus: ConsensusResult, votes: List[AgentVote]):
        """Log a consensus decision"""
        self.decisions.append({
            "timestamp": consensus.timestamp,
            "decision": consensus.decision.value,
            "confidence": consensus.confidence,
            "method": consensus.method,
            "votes": [{"agent": v.agent_name, "signal": v.signal.value, "confidence": v.confidence} for v in votes]
        })
        
        # Maintain max history
        if len(self.decisions) > self.max_history:
            self.decisions = self.decisions[-self.max_history:]
    
    def log_conflict(self, resolution: ConflictResolution):
        """Log a conflict resolution"""
        self.conflicts.append({
            "timestamp": resolution.timestamp,
            "type": resolution.conflict_type.value,
            "method": resolution.resolution_method,
            "decision": resolution.final_decision.decision.value
        })
        
        if len(self.conflicts) > self.max_history:
            self.conflicts = self.conflicts[-self.max_history:]
    
    def get_decision_history(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent decisions"""
        return self.decisions[-limit:]
    
    def get_conflict_patterns(self) -> Dict[str, int]:
        """Analyze common conflict types"""
        patterns = defaultdict(int)
        for conflict in self.conflicts:
            patterns[conflict["type"]] += 1
        return dict(patterns)