from typing import List
import statistics

from src.agents.hermes_workspace.tools.consensus_engine import ConsensusEngine
from src.agents.hermes_workspace.models import AgentVote, AgentSignal, ConflictType, ConsensusResult, ConflictResolution

class ConflictResolver:
    """Handles agent disagreements"""
    
    def __init__(self, consensus_engine: ConsensusEngine):
        self.consensus_engine = consensus_engine
        self.resolution_strategies = {
            ConflictType.BINARY_SPLIT: self._resolve_binary_split,
            ConflictType.MAJORITY_MINORITY: self._resolve_majority,
            ConflictType.THREE_WAY: self._resolve_three_way,
            ConflictType.CONFIDENCE_MISMATCH: self._resolve_confidence_mismatch,
            ConflictType.CRITICAL_OVERRIDE: self._resolve_critical_override,
        }
    
    def identify_conflict_type(self, votes: List[AgentVote]) -> ConflictType:
        """Classify the type of conflict"""
        signals = [vote.signal for vote in votes]
        unique_signals = set(signals)
        
        # Check for emergency signals
        if any(vote.signal == AgentSignal.EMERGENCY for vote in votes):
            return ConflictType.CRITICAL_OVERRIDE
        
        # Check if all same signal but different confidence
        if len(unique_signals) == 1:
            confidences = [vote.confidence for vote in votes]
            if max(confidences) - min(confidences) > 0.3:
                return ConflictType.CONFIDENCE_MISMATCH
        
        # Check distribution
        if len(unique_signals) == 2:
            counts = {s: signals.count(s) for s in unique_signals}
            values = list(counts.values())
            if abs(values[0] - values[1]) <= 1:
                return ConflictType.BINARY_SPLIT
            return ConflictType.MAJORITY_MINORITY
        
        if len(unique_signals) >= 3:
            return ConflictType.THREE_WAY
        
        return ConflictType.MAJORITY_MINORITY
    
    def resolve(self, votes: List[AgentVote]) -> ConflictResolution:
        """Main conflict resolution method"""
        conflict_type = self.identify_conflict_type(votes)
        strategy = self.resolution_strategies[conflict_type]
        return strategy(votes, conflict_type)
    
    def _resolve_binary_split(self, votes: List[AgentVote], conflict_type: ConflictType) -> ConflictResolution:
        """Resolve 50/50 splits using trust weights and confidence"""
        signal, confidence = self.consensus_engine.calculate_weighted_vote(votes)
        
        dissenting = [v.agent_name for v in votes if v.signal != signal]
        
        result = ConsensusResult(
            decision=signal,
            confidence=confidence,
            method="weighted_vote_binary_split",
            participating_agents=[v.agent_name for v in votes],
            dissenting_agents=dissenting,
            reasoning=f"Binary split resolved by weighted voting. Trust-weighted decision favors {signal.value}.",
            vote_breakdown={v.agent_name: (v.signal.value, v.confidence) for v in votes}
        )
        
        return ConflictResolution(
            conflict_type=conflict_type,
            resolution_method="weighted_trust_voting",
            original_votes=votes,
            final_decision=result,
            resolution_reasoning="Applied trust weights to break tie between equally split agents"
        )
    
    def _resolve_majority(self, votes: List[AgentVote], conflict_type: ConflictType) -> ConflictResolution:
        """Resolve clear majority situations"""
        signal, confidence = self.consensus_engine.calculate_weighted_vote(votes)
        
        dissenting = [v.agent_name for v in votes if v.signal != signal]
        
        result = ConsensusResult(
            decision=signal,
            confidence=confidence,
            method="weighted_majority",
            participating_agents=[v.agent_name for v in votes],
            dissenting_agents=dissenting,
            reasoning=f"Clear majority consensus on {signal.value}.",
            vote_breakdown={v.agent_name: (v.signal.value, v.confidence) for v in votes}
        )
        
        return ConflictResolution(
            conflict_type=conflict_type,
            resolution_method="majority_consensus",
            original_votes=votes,
            final_decision=result,
            resolution_reasoning="Majority of agents agree on signal"
        )
    
    def _resolve_three_way(self, votes: List[AgentVote], conflict_type: ConflictType) -> ConflictResolution:
        """Resolve three-way splits"""
        signal, confidence = self.consensus_engine.calculate_weighted_vote(votes)
        
        dissenting = [v.agent_name for v in votes if v.signal != signal]
        
        result = ConsensusResult(
            decision=signal,
            confidence=confidence * 0.85,  # Reduce confidence for three-way split
            method="weighted_vote_three_way",
            participating_agents=[v.agent_name for v in votes],
            dissenting_agents=dissenting,
            reasoning=f"Three-way split resolved via weighted voting. Decision: {signal.value} (confidence adjusted down).",
            vote_breakdown={v.agent_name: (v.signal.value, v.confidence) for v in votes}
        )
        
        return ConflictResolution(
            conflict_type=conflict_type,
            resolution_method="weighted_three_way_resolution",
            original_votes=votes,
            final_decision=result,
            resolution_reasoning="No clear majority - applied trust weights with confidence penalty"
        )
    
    def _resolve_confidence_mismatch(self, votes: List[AgentVote], conflict_type: ConflictType) -> ConflictResolution:
        """Resolve same signal but different confidence levels"""
        signal = votes[0].signal  # All same signal
        avg_confidence = statistics.mean([v.confidence for v in votes])
        weighted_confidence = sum(
            v.confidence * self.consensus_engine.trust_weights.get(v.agent_name, 0.0) 
            for v in votes
        )
        
        result = ConsensusResult(
            decision=signal,
            confidence=weighted_confidence,
            method="confidence_weighted_consensus",
            participating_agents=[v.agent_name for v in votes],
            dissenting_agents=[],
            reasoning=f"Unanimous signal ({signal.value}) with varying confidence. Weighted confidence: {weighted_confidence:.2f}",
            vote_breakdown={v.agent_name: (v.signal.value, v.confidence) for v in votes}
        )
        
        return ConflictResolution(
            conflict_type=conflict_type,
            resolution_method="confidence_aggregation",
            original_votes=votes,
            final_decision=result,
            resolution_reasoning="All agents agree on signal but differ on confidence - using weighted average"
        )
    
    def _resolve_critical_override(self, votes: List[AgentVote], conflict_type: ConflictType) -> ConflictResolution:
        """Handle emergency override situations"""
        emergency_votes = [v for v in votes if v.signal == AgentSignal.EMERGENCY]
        
        if emergency_votes:
            # Take the highest confidence emergency signal
            emergency_vote = max(emergency_votes, key=lambda v: v.confidence)
            
            result = ConsensusResult(
                decision=AgentSignal.EMERGENCY,
                confidence=emergency_vote.confidence,
                method="emergency_override",
                participating_agents=[emergency_vote.agent_name],
                dissenting_agents=[v.agent_name for v in votes if v.agent_name != emergency_vote.agent_name],
                reasoning=f"EMERGENCY OVERRIDE by {emergency_vote.agent_name}: {emergency_vote.reasoning}",
                vote_breakdown={v.agent_name: (v.signal.value, v.confidence) for v in votes}
            )
            
            return ConflictResolution(
                conflict_type=conflict_type,
                resolution_method="emergency_override",
                original_votes=votes,
                final_decision=result,
                resolution_reasoning=f"Emergency signal from {emergency_vote.agent_name} overrides normal consensus"
            )
        
        # Fallback to weighted vote
        return self._resolve_three_way(votes, conflict_type)