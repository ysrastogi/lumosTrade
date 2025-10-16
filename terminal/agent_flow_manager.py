"""
Enhanced AgentFlowManager for all trading agents
"""

from typing import Dict, List, Any, Optional
from .agent_flow import AgentFlow, AthenaFlow, ApolloFlow, DaedalusFlow, HermesFlow

# Import the enhanced ChronosFlow implementation
from .chronos_flow_implementation import ChronosFlow

class AgentFlowManager:
    """Manages all agent flows"""
    
    def __init__(self, connector_manager=None):
        """
        Initialize flow manager
        
        Args:
            connector_manager: Optional AgentConnectorManager instance
        """
        self.flows = {
            'athena': AthenaFlow(),
            'apollo': ApolloFlow(),
            'chronos': ChronosFlow(),  # Using the enhanced ChronosFlow implementation
            'daedalus': DaedalusFlow(),
            'hermes': HermesFlow()
        }
        
        # Set connectors if provided
        if connector_manager:
            self.set_connectors(connector_manager)
    
    def set_connectors(self, connector_manager):
        """
        Set agent connectors for all flows
        
        Args:
            connector_manager: AgentConnectorManager instance
        """
        self.flows['athena'].agent_connector = connector_manager.get_connector('athena')
        self.flows['apollo'].agent_connector = connector_manager.get_connector('apollo')
        self.flows['chronos'].agent_connector = connector_manager.get_connector('chronos')
        self.flows['daedalus'].agent_connector = connector_manager.get_connector('daedalus')
        self.flows['hermes'].agent_connector = connector_manager.get_connector('hermes')
    
    def get_flow(self, agent_name: str) -> Optional[AgentFlow]:
        """Get flow handler for an agent"""
        return self.flows.get(agent_name.lower())
    
    async def run_flow(self, agent_name: str, flow_type: str, **kwargs) -> Dict[str, Any]:
        """Run a specific flow for an agent"""
        flow = self.get_flow(agent_name)
        if not flow:
            return {"error": f"No flow found for agent: {agent_name}"}
        
        # Map flow types to method names (strings)
        flow_method_names = {
            'athena': {
                'market_analysis': 'run_market_analysis_flow',
                'regime_detection': 'run_regime_detection_flow',
                'pattern_detection': 'run_pattern_detection_flow',
                'memory_context': 'run_memory_context_flow',
                'multi_symbol': 'run_multi_symbol_flow',
                'insights': 'run_insights_flow',
                'athena_apollo_integrated': 'run_athena_apollo_integrated_flow'
            },
            'apollo': {
                'signal_generation': 'run_signal_generation_flow',
                'athena_based_signal': 'run_athena_based_signal_flow',
                'multi_signal': 'run_multi_signal_flow',
                'signal_validation': 'run_signal_validation_flow',
                'memory_signals': 'run_memory_signals_flow'
            },
            'chronos': {
                'risk_assessment': 'run_risk_assessment_flow',
                'position_sizing': 'run_position_sizing_flow',
                'capital_efficiency': 'run_capital_efficiency_flow',  # New flow added
                'risk_education': 'run_risk_education_flow'  # New flow added
            },
            'daedalus': {
                'backtest': 'run_backtest_flow'
            },
            'hermes': {
                'consensus': 'run_consensus_flow'
            }
        }
        
        agent_methods = flow_method_names.get(agent_name.lower(), {})
        method_name = agent_methods.get(flow_type)
        
        if method_name and hasattr(flow, method_name):
            method = getattr(flow, method_name)
            return await method(**kwargs)
        else:
            return {"error": f"Flow type '{flow_type}' not found for {agent_name}"}