"""
Hermes Agent Connector
Bridges terminal commands to the Hermes agent instance (Consensus Engine)
Handles agent initialization, lifecycle, and method calls for decision making
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
from .base import AgentConnector
from .athena import AthenaConnector
from .apollo import ApolloConnector
from .chronos import ChronosConnector
from .daedalus import DaedalusConnector


logger = logging.getLogger(__name__)


class HermesConnector(AgentConnector):
    """Connector for Hermes - Consensus Engine Agent"""
    
    def __init__(self):
        super().__init__("Hermes")
        self.athena_connector = None
        self.apollo_connector = None
        self.chronos_connector = None
        self.daedalus_connector = None
    
    async def initialize(self):
        """Initialize Hermes and dependent agents"""
        try:
            # Initialize other agents for consensus
            self.athena_connector = AthenaConnector()
            self.apollo_connector = ApolloConnector()
            self.chronos_connector = ChronosConnector()
            self.daedalus_connector = DaedalusConnector()
            
            # Initialize all agents
            await asyncio.gather(
                self.athena_connector.initialize(),
                self.apollo_connector.initialize(),
                self.chronos_connector.initialize(),
                self.daedalus_connector.initialize()
            )
            
            self.initialized = True
            logger.info("Hermes consensus engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Hermes: {e}", exc_info=True)
            return False
    
    async def generate_consensus(self, symbol: str) -> Dict[str, Any]:
        """
        Generate consensus decision from all agents
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Consensus decision with agent votes
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Gather opinions from all agents
            athena_result = await self.athena_connector.analyze_market(symbol)
            apollo_result = await self.apollo_connector.generate_signal(symbol)
            chronos_result = await self.chronos_connector.assess_risk()
            
            # Determine votes based on results
            agent_votes = {
                "athena": self._interpret_athena_vote(athena_result),
                "apollo": self._interpret_apollo_vote(apollo_result),
                "chronos": self._interpret_chronos_vote(chronos_result),
                "daedalus": {"vote": "BUY", "confidence": 0.73}  # Simulated
            }
            
            # Calculate consensus
            buy_votes = sum(1 for v in agent_votes.values() if v['vote'] == 'BUY')
            sell_votes = sum(1 for v in agent_votes.values() if v['vote'] == 'SELL')
            hold_votes = sum(1 for v in agent_votes.values() if v['vote'] == 'HOLD')
            
            total_agents = len(agent_votes)
            
            if buy_votes > total_agents / 2:
                decision = "BUY"
                confidence = sum(v['confidence'] for v in agent_votes.values() if v['vote'] == 'BUY') / buy_votes
            elif sell_votes > total_agents / 2:
                decision = "SELL"
                confidence = sum(v['confidence'] for v in agent_votes.values() if v['vote'] == 'SELL') / sell_votes
            else:
                decision = "HOLD"
                confidence = sum(v['confidence'] for v in agent_votes.values() if v['vote'] == 'HOLD') / max(hold_votes, 1)
            
            return {
                "symbol": symbol,
                "decision": decision,
                "confidence": confidence,
                "agent_votes": agent_votes,
                "vote_distribution": {
                    "BUY": buy_votes,
                    "SELL": sell_votes,
                    "HOLD": hold_votes
                }
            }
            
        except Exception as e:
            logger.error(f"Consensus generation error: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _interpret_athena_vote(self, result: Dict) -> Dict[str, Any]:
        """Interpret Athena's analysis as a vote"""
        if 'error' in result:
            return {"vote": "HOLD", "confidence": 0.5}
        
        regime = result.get('regime', {})
        regime_type = regime.get('regime', 'UNKNOWN')
        confidence = regime.get('confidence', 0.5)
        
        if 'UP' in regime_type or 'BULL' in regime_type:
            return {"vote": "BUY", "confidence": confidence}
        elif 'DOWN' in regime_type or 'BEAR' in regime_type:
            return {"vote": "SELL", "confidence": confidence}
        else:
            return {"vote": "HOLD", "confidence": confidence}
    
    def _interpret_apollo_vote(self, result: Dict) -> Dict[str, Any]:
        """Interpret Apollo's signal as a vote"""
        if 'error' in result:
            return {"vote": "HOLD", "confidence": 0.5}
        
        direction = result.get('direction', 'HOLD')
        confidence = result.get('confidence', 0.5)
        
        return {"vote": direction, "confidence": confidence}
    
    def _interpret_chronos_vote(self, result: Dict) -> Dict[str, Any]:
        """Interpret Chronos's risk assessment as a vote"""
        if 'error' in result:
            return {"vote": "HOLD", "confidence": 0.5}
        
        warnings = result.get('warnings', [])
        
        # Conservative approach - if warnings exist, suggest HOLD
        if len(warnings) > 2:
            return {"vote": "HOLD", "confidence": 0.7}
        elif len(warnings) > 0:
            return {"vote": "HOLD", "confidence": 0.6}
        else:
            return {"vote": "BUY", "confidence": 0.65}  # Green light if low risk