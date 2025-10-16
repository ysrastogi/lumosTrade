import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from src.memory.assistant import MemoryAssistant
from src.memory.memory_core import MemoryCore

logger = logging.getLogger(__name__)

class AthenaMemoryManager:
    
    def __init__(self, agent_id: str = "athena_agent", use_redis: bool = True, memory_core: Optional[MemoryCore] = None):
        self.agent_id = agent_id
        self.use_redis = use_redis
        
        # Use provided memory core (global) or create a new one
        if memory_core is not None:
            self.memory_core = memory_core
            logger.info(f"Using provided global memory core for agent {agent_id}")
        else:
            self.memory_core = MemoryCore(use_redis=use_redis)
            self.memory_core.initialize_components()
            logger.info(f"Created new memory core for agent {agent_id}")
            
        self.memory_assistant = MemoryAssistant(memory_core=self.memory_core)
        self._initialized = False
        
        logger.info(f"Athena Memory Manager created for agent {agent_id}")
        
    async def initialize(self):
        if self._initialized:
            logger.debug("Memory manager already initialized")
            return True
        
        agent_metadata = {
            "name": "Athena Market Intelligence",
            "role": "Market analysis and intelligence generation",
            "description": "Analyzes market data to identify patterns, regimes, and generate insights",
            "version": "1.0.0",
            "capabilities": ["market_analysis", "pattern_recognition", "regime_detection", "insight_generation"]
        }
    
        success = await self.memory_assistant.initialize_for_agent(
            agent_id=self.agent_id,
            agent_metadata=agent_metadata
        )
    
        if success:
            await self.memory_core.registry.subscribe_agent_to_topic(
                agent_id=self.agent_id,
                topic="market_data"
            )
            await self.memory_core.registry.subscribe_agent_to_topic(
                agent_id=self.agent_id,
                topic="market_analysis"
            )
            await self.memory_core.registry.subscribe_agent_to_topic(
                agent_id=self.agent_id,
                topic="trading_signals"
            )
            self._initialized = True
            logger.info(f"Athena agent {self.agent_id} successfully registered with memory system")
            return True
        
        logger.error("Failed to initialize Athena memory manager")
        return False
        
    async def store_observation(self, symbol: str, data: Dict[str, Any]) -> str:
        if not self._initialized:
            await self.initialize()
            
        content = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Store in memory
        memory_id = await self.memory_core.store_memory(
            agent_id=self.agent_id,
            content=content,
            memory_type="market_observation",
            tags=[symbol, "market_data"]
        )
        
        return memory_id
        
    async def store_analysis(self, symbol: str, analysis_type: str, analysis_data: Dict[str, Any]) -> str:
        """
        Store a market analysis result in memory.
        
        Args:
            symbol: Market symbol analyzed
            analysis_type: Type of analysis (pattern, regime, etc.)
            analysis_data: Analysis results
            
        Returns:
            str: Memory ID
        """
        if not self._initialized:
            await self.initialize()
            
        # Prepare analysis content
        content = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": analysis_type,
            "results": analysis_data
        }
        
        # Store in memory
        memory_id = await self.memory_core.store_memory(
            agent_id=self.agent_id,
            content=content,
            memory_type="market_analysis",
            tags=[symbol, analysis_type]
        )
        
        return memory_id
        
    async def store_insight(self, symbol: str, insight_type: str, insight: str, confidence: float = 0.0) -> str:

        if not self._initialized:
            await self.initialize()
            
        content = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "insight_type": insight_type,
            "insight": insight,
            "confidence": confidence
        }

        memory_id = await self.memory_core.store_memory(
            agent_id=self.agent_id,
            content=content,
            memory_type="market_insight",
            tags=[symbol, insight_type]
        )
        
        await self.memory_core.publish_message(
            sender_id=self.agent_id,
            topic="market_analysis",
            content=content
        )
        
        return memory_id
        
    async def recall_recent_observations(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:

        if not self._initialized:
            await self.initialize()
            
        memories = await self.memory_core.get_agent_context(
            agent_id=self.agent_id,
            context_window=limit
        )
        
        if symbol:
            memories = [m for m in memories if m["data"].get("symbol") == symbol]
            
        return memories
        
    async def recall_recent_insights(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:

        if not self._initialized:
            await self.initialize()
            
        memories = await self.memory_core.get_agent_context(
            agent_id=self.agent_id,
            context_window=limit
        )
        
        if symbol:
            memories = [m for m in memories if m["data"].get("symbol") == symbol]
            
        return memories
        
    async def get_context(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive memory context including observations, insights, and other data.
        
        Args:
            symbol: Optional symbol to filter memories by
            
        Returns:
            Dictionary with memory context
        """
        if not self._initialized:
            await self.initialize()

        # Get agent-specific context
        agent_memories = await self.memory_core.get_agent_context(
            agent_id=self.agent_id,
            context_window=20
        )
        
        # Get market observations from episodic memory store
        observations = await self.memory_core.episodic_store.get_by_type(
            agent_id=self.agent_id,  # Initially search only this agent's memories
            memory_type="market_observation",
            limit=10
        )
        
        # Get market insights from episodic memory store
        insights = await self.memory_core.episodic_store.get_by_type(
            agent_id=self.agent_id,
            memory_type="market_insight",
            limit=10
        )
        
        # If symbol is specified, filter the results
        if symbol:
            # Filter observations and insights by symbol, checking different possible memory formats
            filtered_observations = []
            for obs in observations:
                # Check different possible locations of the symbol in the memory structure
                content = obs.get('content', {})
                if content.get('symbol') == symbol:
                    filtered_observations.append(obs)
                elif content.get('data', {}).get('symbol') == symbol:
                    filtered_observations.append(obs)
                # Add more patterns if needed
            observations = filtered_observations
            
            filtered_insights = []
            for insight in insights:
                content = insight.get('content', {})
                if content.get('symbol') == symbol:
                    filtered_insights.append(insight)
                elif content.get('data', {}).get('symbol') == symbol:
                    filtered_insights.append(insight)
            insights = filtered_insights
        # Now try to get cross-agent observations if available
        # We'll check for known agents directly
        cross_agent_observations = []
        cross_agent_insights = []
        
        try:
            # Known agent IDs (hardcoded for now)
            known_agents = ["apollo_agent", "chronos_agent", "daedalus_agent", "hermes_agent"]
            agent_ids = [agent_id for agent_id in known_agents if agent_id != self.agent_id]
            
            # Get memories from other agents
            for other_agent_id in agent_ids:
                try:
                    # Get observations from other agent
                    other_obs = await self.memory_core.episodic_store.get_by_type(
                        agent_id=other_agent_id,
                        memory_type="market_observation",
                        limit=5
                    )
                    
                    # Get insights from other agent
                    other_insights = await self.memory_core.episodic_store.get_by_type(
                        agent_id=other_agent_id,
                        memory_type="market_insight",
                        limit=5
                    )
                    
                    # Filter by symbol if needed
                    if symbol:
                        other_obs = [
                            obs for obs in other_obs 
                            if obs.get('content', {}).get('symbol') == symbol
                        ]
                        other_insights = [
                            insight for insight in other_insights 
                            if insight.get('content', {}).get('symbol') == symbol
                        ]
                    
                    # Add to cross-agent collections
                    cross_agent_observations.extend(other_obs)
                    cross_agent_insights.extend(other_insights)
                except Exception as e:
                    # Skip if this agent has no memories
                    logger.debug(f"Error retrieving memories from {other_agent_id}: {e}")
                
            # Add cross-agent memories to main collections
            observations.extend(cross_agent_observations)
            insights.extend(cross_agent_insights)
            
        except Exception as e:
            logger.warning(f"Error retrieving cross-agent memories: {e}")
                
        return {
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat(),
            "symbol_filter": symbol,
            "memories": agent_memories,
            "observations": observations,
            "insights": insights,
            "intelligence_summary": {
                "observation_count": len(observations),
                "insight_count": len(insights),
                "cross_agent_observations": len(cross_agent_observations),
                "cross_agent_insights": len(cross_agent_insights),
            }
        }


# Global memory core instance for cross-agent sharing
_global_memory_core = None

def get_global_memory_core(use_redis: bool = True) -> MemoryCore:
    """
    Get or create a global memory core instance for cross-agent memory sharing.
    This ensures all agents (Athena, Apollo, etc.) use the same memory core.

    Args:
        use_redis: Whether to use Redis for persistence

    Returns:
        Global MemoryCore instance
    """
    global _global_memory_core

    if _global_memory_core is None:
        _global_memory_core = MemoryCore(use_redis=use_redis)
        _global_memory_core.initialize_components()
        logger.info("Created global memory core for cross-agent sharing")

    return _global_memory_core