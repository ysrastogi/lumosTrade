from typing import Dict, Any
from collections import defaultdict

class TrustScoreTracker:
    """Track and update agent reliability metrics"""
    
    def __init__(self, initial_weights: Dict[str, float]):
        self.trust_scores = initial_weights.copy()
        self.performance_history = defaultdict(list)
        self.decision_outcomes = []
    
    def update_trust(self, agent_name: str, outcome_success: bool, impact: float = 0.05):
        """Update trust score based on decision outcome"""
        if agent_name not in self.trust_scores:
            self.trust_scores[agent_name] = 0.25
        
        # Record performance
        self.performance_history[agent_name].append(outcome_success)
        
        # Update trust score
        adjustment = impact if outcome_success else -impact
        self.trust_scores[agent_name] = max(0.05, min(0.50, self.trust_scores[agent_name] + adjustment))
        
        # Renormalize
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Ensure all weights sum to 1.0"""
        total = sum(self.trust_scores.values())
        if total > 0:
            self.trust_scores = {k: v/total for k, v in self.trust_scores.items()}
    
    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get performance statistics for an agent"""
        history = self.performance_history[agent_name]
        if not history:
            return {"trust_score": self.trust_scores.get(agent_name, 0.25), "decisions": 0}
        
        return {
            "trust_score": self.trust_scores.get(agent_name, 0.25),
            "decisions": len(history),
            "success_rate": sum(history) / len(history),
            "recent_performance": sum(history[-10:]) / min(len(history), 10)
        }