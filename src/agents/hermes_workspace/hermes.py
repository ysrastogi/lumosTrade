from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
import statistics

from src.agents.hermes_workspace.models import AgentVote, ConsensusResult, ConflictType, AgentSignal
from src.agents.hermes_workspace.tools.consensus_engine import ConsensusEngine
from src.agents.hermes_workspace.tools.conflict_resolver import ConflictResolver
from src.agents.hermes_workspace.tools.reasoning_aggregator import ReasoningAggregator
from src.agents.hermes_workspace.tools.trust_socre_tracker import TrustScoreTracker
from src.agents.hermes_workspace.tools.decision_logger import DecisionLogger
from src.agents.hermes_workspace.tools.emergency_override import EmergencyOverride
from src.agents.hermes_workspace.tools.priority_router import PriorityRouter
from src.agents.hermes_workspace.memory_manager import HermesMemory
from src.agents.hermes_workspace.prompts import (
    get_system_prompt,
    get_debate_summary_prompt,
    get_override_explanation_prompt, 
    get_override_context_summary, 
    get_fallback_override_explanation,
    get_consensus_report_template,
    get_vote_details_summary,
    get_consensus_report_prompt
)
from src.llm.client import GeminiClient

class HermesLLMEngine:
    """LLM integration for natural language processing and generation using Gemini"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.system_prompt = get_system_prompt()
        self.llm_client = GeminiClient(model=model_name)
    
    def summarize_debate(self, votes: List[AgentVote]) -> str:
        """Generate 3-bullet summary of agent debate using Gemini LLM"""
        # First create a simple structured summary
        signals = defaultdict(list)
        for vote in votes:
            signals[vote.signal.value].append((vote.agent_name, vote.confidence))
        
        summary_points = []
        for signal, agents in signals.items():
            agent_list = ", ".join([f"{name} ({conf:.0%})" for name, conf in agents])
            summary_points.append(f"• {signal.upper()}: {agent_list}")
        
        structured_summary = "\n".join(summary_points)
        
        # Then use LLM to create a more insightful summary if there are multiple signals
        if len(signals) > 1:
            prompt = get_debate_summary_prompt(structured_summary)
            
            try:
                llm_summary = self.llm_client.generate(
                    prompt=prompt,
                    system_instruction=self.system_prompt,
                    temperature=0.3,
                    max_output_tokens=300
                )
                return llm_summary
            except Exception as e:
                print(f"Error generating debate summary with LLM: {e}")
                return structured_summary
        
        return structured_summary
    
    def explain_override(self, overriding_agent: str, overridden_agent: str, context: Dict[str, Any]) -> str:
        """Explain why one agent overrode another using Gemini LLM"""
        # Generate context summary from the override details
        context_summary = get_override_context_summary(
            overriding_agent=overriding_agent,
            overridden_agent=overridden_agent,
            context=context
        )
        
        # Get the prompt for override explanation
        prompt = get_override_explanation_prompt(context_summary)

        try:
            # Generate explanation using the LLM
            llm_explanation = self.llm_client.generate(
                prompt=prompt,
                system_instruction=self.system_prompt,
                temperature=0.2,
                max_output_tokens=400
            )
            return llm_explanation
        except Exception as e:
            print(f"Error generating override explanation with LLM: {e}")
            
            # Fallback to template-based explanation when LLM fails
            return get_fallback_override_explanation(
                overriding_agent=overriding_agent,
                overridden_agent=overridden_agent,
                context=context
            )
    
    def generate_consensus_report(self, consensus: ConsensusResult, votes: List[AgentVote], aggregated_reasoning: str) -> str:
        """Generate comprehensive consensus report using Gemini LLM"""
        # First create a structured report with the essential information
        report_template = get_consensus_report_template(
            consensus=consensus,
            votes=votes,
            aggregated_reasoning=aggregated_reasoning
        )

        # For complex decisions with dissent, use LLM to enhance the reasoning section
        if consensus.dissenting_agents:
            # Get a formatted summary of vote details
            vote_details = get_vote_details_summary(votes)
            
            # Get the prompt for consensus report enhancement
            prompt = get_consensus_report_prompt(
                consensus=consensus,
                vote_details=vote_details
            )

            try:
                enhanced_reasoning = self.llm_client.generate(
                    prompt=prompt,
                    system_instruction=self.system_prompt,
                    temperature=0.3,
                    max_output_tokens=600
                )
                
                # Replace the reasoning section with the enhanced LLM version
                report_sections = report_template.split("REASONING SYNTHESIS")
                updated_report = report_sections[0] + "REASONING SYNTHESIS\n───────────────────────────────────────────────────────────────\n" + enhanced_reasoning
                updated_report += "\n\n───────────────────────────────────────────────────────────────"
                updated_report += "\nDISSENTING AGENTS: " + (', '.join(consensus.dissenting_agents) if consensus.dissenting_agents else 'None')
                updated_report += "\n═══════════════════════════════════════════════════════════════"
                
                return updated_report
            except Exception as e:
                print(f"Error generating enhanced consensus report with LLM: {e}")
                return report_template
                
        return report_template


# ============================================================================
# MAIN HERMES AGENT CLASS
# ============================================================================

class HermesAgent:
    """Main HERMES Consensus Mediator Agent"""
    
    def __init__(self, 
                 initial_trust_weights: Dict[str, float],
                 agent_specializations: Optional[Dict[str, List[str]]] = None,
                 llm_model: str = "gemini-2.5-flash",
                 llm_temperature: float = 0.2):
        
        # Initialize trust weights
        self.trust_weights = initial_trust_weights
        
        # Initialize all components
        self.consensus_engine = ConsensusEngine(self.trust_weights)
        self.conflict_resolver = ConflictResolver(self.consensus_engine)
        self.reasoning_aggregator = ReasoningAggregator()
        self.trust_tracker = TrustScoreTracker(self.trust_weights)
        self.decision_logger = DecisionLogger()
        self.emergency_override = EmergencyOverride()
        self.priority_router = PriorityRouter(agent_specializations or {})
        
        # Memory systems
        self.memory = HermesMemory()
        
        # LLM Engine configuration
        self.llm = HermesLLMEngine(model_name=llm_model)
        self.llm_temperature = llm_temperature
        
        # State
        self.current_state = {
            "agent_votes": {},
            "trust_weights": self.trust_weights,
            "consensus_result": None,
            "last_decision_time": None,
            "llm_config": {
                "model": llm_model,
                "temperature": llm_temperature
            }
        }
    
    def receive_votes(self, votes: List[AgentVote]) -> ConsensusResult:
        """Main entry point: receive agent votes and produce consensus"""
        
        # Check for emergency override
        override_reason = self.emergency_override.check_override_conditions(votes)
        if override_reason:
            print(f"⚠️  WARNING: Emergency conditions detected: {override_reason}")
        
        # Check if human override is active
        if self.emergency_override.override_active:
            return self.emergency_override.activate_override(
                self.emergency_override.override_reason,
                AgentSignal.HOLD
            )
        
        # Update state with votes
        self.current_state["agent_votes"] = {v.agent_name: v for v in votes}
        
        # Check for unanimity
        if self.consensus_engine.detect_unanimity(votes):
            consensus = self._handle_unanimous_vote(votes)
        else:
            # Handle conflict
            conflict_resolution = self.conflict_resolver.resolve(votes)
            consensus = conflict_resolution.final_decision
            
            # Log conflict
            self.decision_logger.log_conflict(conflict_resolution)
            self.memory.record_conflict_pattern(conflict_resolution.conflict_type)
        
        # Aggregate reasoning
        aggregated_reasoning = self.reasoning_aggregator.aggregate_reasoning(votes, consensus)
        
        # Log decision
        self.decision_logger.log_decision(consensus, votes)
        self.memory.record_decision(consensus, votes)
        
        # Update state
        self.current_state["consensus_result"] = consensus
        self.current_state["last_decision_time"] = datetime.now()
        
        # Snapshot trust scores
        self.memory.snapshot_trust_scores(self.trust_tracker.trust_scores)
        
        return consensus
    
    def _handle_unanimous_vote(self, votes: List[AgentVote]) -> ConsensusResult:
        """Handle unanimous agreement"""
        signal = votes[0].signal
        avg_confidence = statistics.mean([v.confidence for v in votes])
        
        return ConsensusResult(
            decision=signal,
            confidence=min(avg_confidence * 1.1, 1.0),  # Boost confidence for unanimity
            method="unanimous_consensus",
            participating_agents=[v.agent_name for v in votes],
            dissenting_agents=[],
            reasoning=f"Unanimous agreement on {signal.value}. All agents align.",
            vote_breakdown={v.agent_name: (v.signal.value, v.confidence) for v in votes}
        )
    
    def update_agent_performance(self, agent_name: str, outcome_success: bool):
        """Update trust scores based on outcome"""
        self.trust_tracker.update_trust(agent_name, outcome_success)
        
        # Update consensus engine with new weights
        self.consensus_engine.trust_weights = self.trust_tracker.trust_scores
        self.current_state["trust_weights"] = self.trust_tracker.trust_scores
        
        # Log the trust update
        print(f"📊 Trust updated for {agent_name}: {self.trust_tracker.trust_scores[agent_name]:.3f} ({'✓' if outcome_success else '✗'})")
        
    def query_llm(self, 
                 prompt: str, 
                 system_instruction: str = None, 
                 temperature: float = None,
                 max_output_tokens: int = None,
                 top_p: float = None,
                 top_k: int = None) -> str:
        """
        Direct interface to query the LLM with custom prompts
        
        Args:
            prompt: The prompt to send to the LLM
            system_instruction: Optional override for the system instruction
            temperature: Optional temperature override (defaults to instance setting)
            max_output_tokens: Optional maximum output tokens limit
            top_p: Optional top_p parameter for sampling
            top_k: Optional top_k parameter for sampling
            
        Returns:
            The LLM response as a string
        """
        # Use instance temperature if not specified
        temp = temperature if temperature is not None else self.llm_temperature
        
        # Use default system prompt if not specified
        sys_instruction = system_instruction if system_instruction is not None else self.llm.system_prompt
        
        # Build LLM parameters dictionary
        llm_params = {
            "prompt": prompt,
            "system_instruction": sys_instruction,
            "temperature": temp
        }
        
        # Add optional parameters if specified
        if max_output_tokens is not None:
            llm_params["max_output_tokens"] = max_output_tokens
        if top_p is not None:
            llm_params["top_p"] = top_p
        if top_k is not None:
            llm_params["top_k"] = top_k
        
        try:
            response = self.llm.llm_client.generate(**llm_params)
            return response
        except Exception as e:
            error_msg = f"Error querying LLM: {str(e)}"
            print(f"⚠️ {error_msg}")
            return error_msg