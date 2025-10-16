from typing import List, Optional
from datetime import datetime
import statistics
from src.agents.hermes_workspace.models import AgentVote, AgentSignal, ConsensusResult


class EmergencyOverride:
    """Human intervention system for critical situations"""
    
    def __init__(self):
        self.override_active = False
        self.override_reason = None
        self.override_timestamp = None
        self.override_history = []
    
    def activate_override(self, reason: str, decision: AgentSignal) -> ConsensusResult:
        """Activate human override"""
        self.override_active = True
        self.override_reason = reason
        self.override_timestamp = datetime.now()
        
        result = ConsensusResult(
            decision=decision,
            confidence=1.0,
            method="human_override",
            participating_agents=["human"],
            dissenting_agents=[],
            reasoning=f"HUMAN OVERRIDE: {reason}"
        )
        
        self.override_history.append({
            "timestamp": self.override_timestamp,
            "reason": reason,
            "decision": decision.value
        })
        
        return result
    
    def deactivate_override(self):
        """Return to normal consensus operation"""
        self.override_active = False
        self.override_reason = None
    
    def check_override_conditions(self, votes: List[AgentVote]) -> Optional[str]:
        """Check if conditions warrant human intervention"""
        # Check for extremely low confidence across all agents
        avg_confidence = statistics.mean([v.confidence for v in votes])
        if avg_confidence < 0.3:
            return f"Extremely low average confidence: {avg_confidence:.2f}"
        
        # Check for critical signal disagreement
        signals = set(v.signal for v in votes)
        if AgentSignal.EMERGENCY in signals and len(signals) > 1:
            return "Emergency signal conflict with other agents"
        
        return None