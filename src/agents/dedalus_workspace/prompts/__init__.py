"""
Daedalus Prompt Collection Module
Contains organized prompt templates for the Daedalus trading agent
"""

from src.agents.dedalus_workspace.prompts.system import SYSTEM_PROMPT
from src.agents.dedalus_workspace.prompts.intent_classification import (
    INTENT_CLASSIFICATION_PROMPT
)
from src.agents.dedalus_workspace.prompts.strategy_creation import (
    STRATEGY_EXTRACTION_PROMPT
)
from src.agents.dedalus_workspace.prompts.explanations import (
    EXPLANATION_PROMPT,
    NEXT_ACTIONS_PROMPT
)
from src.agents.dedalus_workspace.prompts.strategy_ideas import (
    STRATEGY_IDEAS_PROMPT
)
from src.agents.dedalus_workspace.prompts.market_params_prompt import (
    MARKET_PARAMS_PROMPT
)
from src.agents.dedalus_workspace.prompts.forecast import (
    FORECAST_PARAMS_PROMPT
)