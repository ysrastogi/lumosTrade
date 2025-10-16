"""
Consensus report prompts for the Hermes Consensus Mediator Agent.
"""

from typing import List
from datetime import datetime
from src.agents.hermes_workspace.models import ConsensusResult, AgentVote

def get_consensus_report_template(
    consensus: ConsensusResult, 
    votes: List[AgentVote], 
    aggregated_reasoning: str
) -> str:
    """
    Generate a template for consensus reports.
    
    Args:
        consensus: The consensus result.
        votes: The list of agent votes.
        aggregated_reasoning: The aggregated reasoning from the agents.
        
    Returns:
        A formatted string template for the consensus report.
    """
    report_template = f"""
═══════════════════════════════════════════════════════════════
                    CONSENSUS DECISION REPORT
═══════════════════════════════════════════════════════════════

DECISION: {consensus.decision.value.upper()}
CONFIDENCE: {consensus.confidence:.0%}
METHOD: {consensus.method}
TIMESTAMP: {consensus.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

───────────────────────────────────────────────────────────────
AGENT VOTES ({len(votes)} participating)
───────────────────────────────────────────────────────────────
"""
    for vote in votes:
        status = "✓ SUPPORTING" if vote.signal == consensus.decision else "✗ DISSENTING"
        report_template += f"\n{vote.agent_name.upper()}: {vote.signal.value} ({vote.confidence:.0%}) - {status}"
    
    report_template += f"""

───────────────────────────────────────────────────────────────
REASONING SYNTHESIS
───────────────────────────────────────────────────────────────
{aggregated_reasoning}

───────────────────────────────────────────────────────────────
DISSENTING AGENTS: {', '.join(consensus.dissenting_agents) if consensus.dissenting_agents else 'None'}
═══════════════════════════════════════════════════════════════
"""
    return report_template

def get_vote_details_summary(votes: List[AgentVote]) -> str:
    """
    Generate a summary of vote details.
    
    Args:
        votes: The list of agent votes.
        
    Returns:
        A formatted string summarizing the vote details.
    """
    return "\n".join([
        f"- {v.agent_name}: {v.signal.value} (confidence: {v.confidence:.0%}) - Reasoning: {v.reasoning[:200]}..."
        for v in votes
    ])

def get_consensus_report_prompt(
    consensus: ConsensusResult,
    vote_details: str
) -> str:
    """
    Generate a prompt for enhancing consensus reports with LLM analysis.
    
    Args:
        consensus: The consensus result.
        vote_details: A string containing the vote details summary.
        
    Returns:
        The consensus report enhancement prompt as a string.
    """
    return f"""As HERMES, the consensus mediator, analyze this decision with dissenting votes:

Decision: {consensus.decision.value}
Method: {consensus.method}
Confidence: {consensus.confidence:.0%}
Dissenting agents: {', '.join(consensus.dissenting_agents)}

Vote details:
{vote_details}

Provide a nuanced analysis of:
1. Why the consensus was reached despite dissent
2. The key factors that influenced the final decision
3. The most compelling points from both supporting and dissenting agents
4. A balanced conclusion that acknowledges all perspectives

Use a professional, objective tone that acknowledges trade-offs.
"""