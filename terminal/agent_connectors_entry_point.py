"""
Agent Connectors Entry Point
Re-exports all agent connector classes and the manager class
"""

# Re-export all agent connectors for backward compatibility
from terminal.agent_connectors.base import AgentConnector
from terminal.agent_connectors.athena import AthenaConnector
from terminal.agent_connectors.apollo import ApolloConnector
from terminal.agent_connectors.chronos import ChronosConnector
from terminal.agent_connectors.daedalus import DaedalusConnector
from terminal.agent_connectors.hermes import HermesConnector
from terminal.agent_connectors.manager import AgentConnectorManager

# Provide a convenience instance of the manager
connector_manager = AgentConnectorManager()

# Define __all__ to control what gets imported with wildcard imports
__all__ = [
    'AgentConnector',
    'AthenaConnector',
    'ApolloConnector',
    'ChronosConnector',
    'DaedalusConnector',
    'HermesConnector',
    'AgentConnectorManager',
    'connector_manager',
]