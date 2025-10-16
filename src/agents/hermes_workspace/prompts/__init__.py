"""
Prompt templates for the Hermes Consensus Mediator Agent.

This package contains prompt templates used by the Hermes agent for various 
LLM operations such as debate summarization, override explanation, and 
consensus report generation.
"""

from src.agents.hermes_workspace.prompts.system_prompts import (
    get_system_prompt
)
from src.agents.hermes_workspace.prompts.debate_prompts import (
    get_debate_summary_prompt
)
from src.agents.hermes_workspace.prompts.override_prompts import (
    get_override_explanation_prompt
)
from src.agents.hermes_workspace.prompts.report_prompts import (
    get_consensus_report_prompt
)