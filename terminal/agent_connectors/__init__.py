"""
Agent Connectors Package
Bridges terminal commands to actual agent instances
Handles agent initialization, lifecycle, and method calls
"""

from .base import AgentConnector
from .athena import AthenaConnector
from .apollo import ApolloConnector
from .chronos import ChronosConnector
from .daedalus import DaedalusConnector
from .hermes import HermesConnector
from .manager import AgentConnectorManager

__all__ = [
    'AgentConnector',
    'AthenaConnector',
    'ApolloConnector',
    'ChronosConnector',
    'DaedalusConnector',
    'HermesConnector',
    'AgentConnectorManager',
]