from typing import Dict, List
from src.agents.hermes_workspace.models import AgentVote

class PriorityRouter:
    """Determines which agent should lead in different scenarios"""
    
    def __init__(self, agent_specializations: Dict[str, List[str]]):
        """
        Args:
            agent_specializations: Dict mapping agent names to their specializations
            Example: {"apollo": ["technical", "risk"], "chronos": ["timing", "trends"]}
        """
        self.specializations = agent_specializations
    
    def determine_lead_agent(self, context: str, votes: List[AgentVote]) -> str:
        """Determine which agent should have priority based on context"""
        # Check if context matches any specialization
        for agent, specs in self.specializations.items():
            if any(spec.lower() in context.lower() for spec in specs):
                return agent
        
        # Default to highest confidence vote
        if votes:
            return max(votes, key=lambda v: v.confidence).agent_name
        
        return "hermes"  # Default to self