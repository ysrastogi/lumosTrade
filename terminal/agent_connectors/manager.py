"""
Agent Connector Manager
Manages all agent connectors and provides a unified interface
"""

import asyncio
import logging
from typing import Optional, Dict
from .base import AgentConnector
from .athena import AthenaConnector
from .apollo import ApolloConnector
from .chronos import ChronosConnector
from .daedalus import DaedalusConnector
from .hermes import HermesConnector


logger = logging.getLogger(__name__)


class AgentConnectorManager:
    """Manages all agent connectors"""
    
    def __init__(self):
        # Initialize Athena first as it's needed by Apollo
        athena_connector = AthenaConnector()
        
        self.connectors = {
            'athena': athena_connector,
            'apollo': ApolloConnector(athena_connector=athena_connector),  # Link to Athena!
            'chronos': ChronosConnector(),
            'daedalus': DaedalusConnector(),
            'hermes': HermesConnector()
        }
    
    def get_connector(self, agent_name: str) -> Optional[AgentConnector]:
        """Get connector for an agent"""
        return self.connectors.get(agent_name.lower())
    
    async def initialize_all(self):
        """Initialize all agent connectors"""
        results = await asyncio.gather(
            *[connector.initialize() for connector in self.connectors.values()],
            return_exceptions=True
        )
        
        for agent_name, result in zip(self.connectors.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to initialize {agent_name}: {result}")
            else:
                logger.info(f"{agent_name} initialization: {'Success' if result else 'Failed'}")
    
    async def cleanup_all(self):
        """Cleanup all agent connectors"""
        await asyncio.gather(
            *[connector.cleanup() for connector in self.connectors.values()],
            return_exceptions=True
        )