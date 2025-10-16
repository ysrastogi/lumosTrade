from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from src.agents.hermes_workspace.models import AgentVote, AgentSignal

class ConsensusEngine:
    """Weighted voting mechanism for agent consensus"""
    
    def __init__(self, trust_weights: Dict[str, float]):
        self.trust_weights = trust_weights
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Ensure weights sum to 1.0"""
        total = sum(self.trust_weights.values())
        if total > 0:
            self.trust_weights = {k: v/total for k, v in self.trust_weights.items()}
    
    def calculate_weighted_vote(self, votes: List[AgentVote]) -> Tuple[AgentSignal, float]:
        """Calculate consensus using weighted voting"""
        signal_scores = defaultdict(float)
        confidence_sum = defaultdict(float)
        
        for vote in votes:
            weight = self.trust_weights.get(vote.agent_name, 0.0)
            signal_scores[vote.signal] += weight * vote.confidence
            confidence_sum[vote.signal] += weight
        
        # Find winning signal
        winning_signal = max(signal_scores.items(), key=lambda x: x[1])
        
        # Calculate aggregate confidence
        aggregate_confidence = winning_signal[1] / confidence_sum[winning_signal[0]] if confidence_sum[winning_signal[0]] > 0 else 0.0
        
        return winning_signal[0], min(aggregate_confidence, 1.0)
    
    def detect_unanimity(self, votes: List[AgentVote]) -> bool:
        """Check if all agents agree"""
        if not votes:
            return False
        signals = set(vote.signal for vote in votes)
        return len(signals) == 1
    
    def calculate_majority_threshold(self, votes: List[AgentVote], threshold: float = 0.66) -> Optional[AgentSignal]:
        """Check if any signal has supermajority"""
        signal_weights = defaultdict(float)
        
        for vote in votes:
            weight = self.trust_weights.get(vote.agent_name, 0.0)
            signal_weights[vote.signal] += weight
        
        for signal, weight in signal_weights.items():
            if weight >= threshold:
                return signal
        return None