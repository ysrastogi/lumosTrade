"""
Memory Management Assistant Interface for LumosTrade

This module provides a high-level assistant interface for LumosTrade agents
to interact with the memory system in a simplified manner.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from src.memory.memory_core import MemoryCore

logger = logging.getLogger(__name__)

class MemoryAssistant:
    """
    Memory Management Assistant for LumosTrade agents.
    
    This assistant provides a simplified interface to the memory system,
    with higher-level functions designed for easier agent integration.
    It abstracts many of the complexities of direct memory system interactions.
    """
    
    def __init__(self, memory_core: Optional[MemoryCore] = None):
        """
        Initialize the Memory Assistant.
        
        Args:
            memory_core: Reference to a MemoryCore instance (creates one if None)
        """
        self.memory_core = memory_core or MemoryCore()
        self.agent_id = None
        self._init_complete = False
    
    async def initialize_for_agent(self, agent_id: str, agent_metadata: Dict[str, Any]) -> bool:
        """
        Initialize the memory assistant for a specific agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_metadata: Agent metadata including capabilities, description, etc.
            
        Returns:
            bool: Success status
        """
        self.agent_id = agent_id
        
        # Extract role and output schema from metadata
        role = agent_metadata.get("role", "Agent")
        output_schema = agent_metadata.get("output_schema", ["text"])
        
        # Check if agent is already registered
        registry = self.memory_core.registry
        agent_info = await registry.get_agent_info(agent_id)
        
        # If agent is already registered, consider it a success
        if agent_info is not None:
            success = True
        else:
            # Otherwise, register the agent
            success = await self.memory_core.register_agent(
                agent_id, role, output_schema
            )
        
        # Subscribe to relevant topics based on agent metadata
        if success and "interests" in agent_metadata:
            topics = agent_metadata["interests"]
            # Subscribe to each topic
            if self.memory_core.message_bus is not None:
                for topic in topics:
                    # Use message_bus directly instead of the non-existent memory_core.subscribe
                    await self.memory_core.registry.subscribe_agent_to_topic(
                        agent_id, topic
                    )
        
        self._init_complete = success
        return success
    
    def _check_initialized(self):
        """Verify the assistant has been initialized for an agent"""
        if not self._init_complete or not self.agent_id:
            raise RuntimeError("Memory assistant must be initialized with agent_id first")
    
    async def remember(self, content: Dict[str, Any], memory_type: str = "observation",
                     tags: Optional[List[str]] = None) -> str:
        """
        Store a memory for the agent.
        
        Args:
            content: Memory content to store
            memory_type: Type of memory (observation, decision, analysis, etc.)
            tags: Optional tags for easier retrieval
            
        Returns:
            str: Memory ID
        """
        self._check_initialized()
        return await self.memory_core.store_memory(
            agent_id=self.agent_id,
            memory_type=memory_type,
            content=content,
            tags=tags
        )
    
    async def recall(self, query: str, limit: int = 10,
                   memory_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Recall memories based on a query.
        
        Args:
            query: Natural language or keyword query
            limit: Maximum number of memories to retrieve
            memory_types: Optional list of memory types to filter
            
        Returns:
            List[Dict]: List of memory objects
        """
        self._check_initialized()
        return await self.memory_core.retrieve_memories(
            query=query,
            agent_id=self.agent_id,
            limit=limit,
            memory_types=memory_types
        )
    
    async def recall_recent(self, limit: int = 10,
                          memory_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Recall recent memories.
        
        Args:
            limit: Maximum number of memories to retrieve
            memory_types: Optional list of memory types to filter
            
        Returns:
            List[Dict]: List of recent memory objects
        """
        self._check_initialized()
        
        # Use the episodic_store component directly for time-based retrieval
        return await self.memory_core.episodic_store.get_recent(
            agent_id=self.agent_id,
            limit=limit,
            memory_types=memory_types
        )
    
    async def get_context(self, window_size: int = 10) -> Dict[str, Any]:
        """
        Get the current agent context.
        
        Args:
            window_size: Size of the context window (number of recent items)
            
        Returns:
            Dict: Agent context including recent memories, messages, events
        """
        self._check_initialized()
        return await self.memory_core.get_agent_context(
            agent_id=self.agent_id,
            context_window=window_size
        )
    
    async def send_message(self, topic: str, content: Dict[str, Any]) -> str:
        """
        Send a message to other agents via the message bus.
        
        Args:
            topic: Message topic
            content: Message content
            
        Returns:
            str: Message ID
        """
        self._check_initialized()
        return await self.memory_core.publish_message(
            sender_id=self.agent_id,
            topic=topic,
            content=content
        )
    
    async def analyze_history(self, query: str = "", 
                            timeframe: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Analyze agent history to extract insights.
        
        Args:
            query: Optional query to focus analysis
            timeframe: Optional time range {"start": ISO8601, "end": ISO8601}
            
        Returns:
            Dict: Analysis results
        """
        self._check_initialized()
        
        # Retrieve relevant memories
        memories = await self.memory_core.retrieve_memories(
            query=query,
            agent_id=self.agent_id,
            time_range=timeframe,
            limit=100  # Higher limit for analysis
        )
        
        # Use summarizer to generate insights
        if memories:
            memory_texts = [json.dumps(m["content"]) for m in memories]
            combined_text = "\n".join(memory_texts)
            
            summary = await self.memory_core.summarizer.generate_summary(
                content=combined_text,
                prompt_template="Analyze the agent history and extract key insights: {content}"
            )
            
            return {
                "agent_id": self.agent_id,
                "query": query,
                "timeframe": timeframe,
                "memory_count": len(memories),
                "insights": summary
            }
        
        return {
            "agent_id": self.agent_id,
            "query": query,
            "timeframe": timeframe,
            "memory_count": 0,
            "insights": "No memories found for analysis."
        }
    
    async def subscribe_to_topics(self, topics: List[str]) -> bool:
        """
        Subscribe to additional message topics.
        
        Args:
            topics: List of topics to subscribe to
            
        Returns:
            bool: Success status
        """
        self._check_initialized()
        return await self.memory_core.subscribe(self.agent_id, topics)