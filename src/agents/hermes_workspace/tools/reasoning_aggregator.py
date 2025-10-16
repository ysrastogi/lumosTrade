from typing import List
from src.agents.hermes_workspace.models import AgentVote, ConsensusResult

class ReasoningAggregator:
    """Combines multiple agent explanations into unified reasoning"""
    
    def aggregate_reasoning(self, votes: List[AgentVote], consensus: ConsensusResult) -> str:
        """Combine reasoning from all agents"""
        supporting = [v for v in votes if v.signal == consensus.decision]
        dissenting = [v for v in votes if v.signal != consensus.decision]
        
        summary = []
        
        # Majority reasoning
        if supporting:
            summary.append("SUPPORTING ARGUMENTS:")
            for vote in supporting:
                summary.append(f"  • {vote.agent_name} (confidence: {vote.confidence:.2f}): {vote.reasoning}")
        
        # Dissenting opinions
        if dissenting:
            summary.append("\nDISSENTING VIEWS:")
            for vote in dissenting:
                summary.append(f"  • {vote.agent_name} (confidence: {vote.confidence:.2f}): {vote.reasoning}")
        
        # Final synthesis
        summary.append(f"\nCONSENSUS: {consensus.reasoning}")
        
        return "\n".join(summary)
    
    def generate_executive_summary(self, consensus: ConsensusResult, votes: List[AgentVote]) -> str:
        """Create brief executive summary"""
        total_agents = len(votes)
        supporting = sum(1 for v in votes if v.signal == consensus.decision)
        
        return (
            f"Decision: {consensus.decision.value.upper()} "
            f"(Confidence: {consensus.confidence:.0%}) | "
            f"Support: {supporting}/{total_agents} agents | "
            f"Method: {consensus.method}"
        )