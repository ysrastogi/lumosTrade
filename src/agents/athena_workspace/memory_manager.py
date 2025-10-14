import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from src.memory.assistant import MemoryAssistant
from src.memory.memory_core import MemoryCore

logger = logging.getLogger(__name__)

class AthenaMemoryManager:
    
    def __init__(self, agent_id: str = "athena_agent", use_redis: bool = True):
        self.agent_id = agent_id
        self.use_redis = use_redis
        self.memory_core = MemoryCore(use_redis=use_redis)
        self.memory_assistant = MemoryAssistant(memory_core=self.memory_core)
        self.memory_core.initialize_components()
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
        memory_id = await self.memory_assistant.remember(
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
        memory_id = await self.memory_assistant.remember(
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
        
        memory_id = await self.memory_assistant.remember(
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
            
        memories = await self.memory_assistant.recall_recent(
            limit=limit,
            memory_types=["market_observation"]
        )
        
        if symbol:
            memories = [m for m in memories if m["data"].get("symbol") == symbol]
            
        return memories
        
    async def recall_recent_insights(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:

        if not self._initialized:
            await self.initialize()
            
        memories = await self.memory_assistant.recall_recent(
            limit=limit,
            memory_types=["market_insight"]
        )
        
        if symbol:
            memories = [m for m in memories if m["data"].get("symbol") == symbol]
            
        return memories
        
    async def get_context(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()
            
        context = await self.memory_assistant.get_context(window_size=20)
    
        if symbol:
            if "recent_memories" in context:
                context["recent_memories"] = [
                    m for m in context["recent_memories"] 
                    if m.get("data", {}).get("symbol") == symbol
                ]
                
        return context