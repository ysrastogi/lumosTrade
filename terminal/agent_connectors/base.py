"""
Base Agent Connector
Defines the base class for all agent connectors
"""

import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class AgentConnector:
    """Base class for agent connectors"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.agent_instance = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the agent instance"""
        raise NotImplementedError("Subclasses must implement initialize()")
    
    async def cleanup(self):
        """Cleanup agent resources"""
        self.initialized = False
        self.agent_instance = None