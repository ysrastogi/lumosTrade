"""
Memory Core for LumosTrade

This module implements the central orchestrator for the memory system,
providing a unified interface for agent memory management.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union

import redis

from .registry import AgentRegistry
from .episodic_store import EpisodicMemoryStore
from .semantic_store import SemanticMemoryStore
from .temporal_log import TemporalEventLog
from .message_bus import MessageBus
from .retriever import QueryEngine
from .summarizer import SummarizationEngine

logger = logging.getLogger(__name__)

class MemoryCore:
    """
    Central orchestrator for the LumosTrade memory system.
    
    This class integrates all memory components and provides a unified interface
    for agents to store and retrieve information, manage context, and facilitate
    inter-agent communication.
    """
    
    def __init__(self, 
                config: Optional[Dict[str, Any]] = None,
                use_redis: bool = True,
                redis_url: str = "redis://localhost:6379/0",
                vector_db_uri: Optional[str] = None):
        """
        Initialize the Memory Core.
        
        Args:
            config: Configuration dictionary for memory components
            use_redis: Whether to use Redis for memory persistence
            redis_url: Redis connection URL
            vector_db_uri: Connection URI for vector database (optional)
        """
        # Configuration
        self.config = config or {}
        self.config.update({
            "use_redis": use_redis,
            "redis_url": redis_url,
            "vector_db_uri": vector_db_uri
        })
        
        # Component initialization - will be implemented in full version
        # These are placeholders to avoid circular imports during stub creation
        self.registry = None
        self.episodic_store = None
        self.semantic_store = None
        self.temporal_log = None
        self.message_bus = None
        self.query_engine = None
        self.summarizer = None
        
        # Session management
        self.current_session_id = None
        self._components_initialized = False
        
        logger.info("🧠 Memory Core initialized - stub implementation")
        
    @property
    def is_initialized(self) -> bool:
        """Check if memory components are initialized"""
        return self._components_initialized
    
    def initialize_components(self):
        """
        Initialize all memory components.
        
        This method must be called after instantiation to set up
        all component relationships correctly.
        """
        # Initialize components
        self.registry = AgentRegistry(self)
        self.episodic_store = EpisodicMemoryStore(self, 
                                                 use_redis=self.config.get("use_redis", True),
                                                 redis_url=self.config.get("redis_url"))
        self.semantic_store = SemanticMemoryStore(self, 
                                                vector_db_uri=self.config.get("vector_db_uri"))
        self.temporal_log = TemporalEventLog(self)
        self.message_bus = MessageBus(self, 
                                     use_redis=self.config.get("use_redis", True),
                                     redis_url=self.config.get("redis_url"))
        self.query_engine = QueryEngine(self)
        self.summarizer = SummarizationEngine(self)
        
        # Start a new session
        self.start_new_session()
        
        # Mark components as initialized
        self._components_initialized = True
        
        logger.info("🧠 Memory Core components initialized")
    
    def start_new_session(self) -> str:
        """
        Start a new memory session.
        
        Returns:
            str: The new session ID
        """
        self.current_session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        logger.info(f"New memory session started: {self.current_session_id}")
        return self.current_session_id
    
    def get_current_session(self) -> str:
        """Get the current active session ID"""
        return self.current_session_id
        
    async def get_agent_context(self, agent_id: str, context_window: int = 10) -> Dict[str, Any]:
        """
        Get the current agent context including recent memories, messages, and events.
        
        Args:
            agent_id: ID of the agent
            context_window: Size of the context window (number of recent items)
            
        Returns:
            Dict: Agent context data
        """
        if not self.is_initialized:
            raise RuntimeError("Memory components not initialized")
            
        # Get recent memories
        recent_memories = await self.episodic_store.get_recent(
            agent_id=agent_id, 
            limit=context_window
        )
        
        # Get recent messages for agent
        recent_messages = await self.message_bus.get_messages_for_agent(
            agent_id=agent_id,
            limit=context_window
        )
        
        # Get recent events
        recent_events = await self.temporal_log.get_recent_events(
            limit=context_window
        )
        
        return {
            "agent_id": agent_id,
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "recent_memories": recent_memories,
            "recent_messages": recent_messages,
            "recent_events": recent_events
        }
    
    # === Agent Registration Methods ===
    
    async def store_memory(self, agent_id: str, memory_type: str, 
                      content: Dict[str, Any], tags: List[str] = None) -> str:
        """
        Store a memory in the episodic memory store.
        
        Args:
            agent_id: ID of the agent storing the memory
            memory_type: Type of memory (observation, decision, analysis, etc.)
            content: Memory content to store
            tags: Optional tags for easier retrieval
            
        Returns:
            str: Memory ID
        """
        if self.episodic_store is None:
            raise RuntimeError("Memory components not initialized")
            
        return await self.episodic_store.store(
            agent_id=agent_id,
            memory_type=memory_type,
            content=content,
            tags=tags
        )
        
    async def publish_message(self, sender_id: str, topic: str, content: Dict[str, Any]) -> str:
        """
        Publish a message from an agent to a topic.
        
        Args:
            sender_id: ID of the sending agent
            topic: Message topic
            content: Message content
            
        Returns:
            str: Message ID
        """
        if not self.is_initialized:
            raise RuntimeError("Memory components not initialized")
            
        return await self.message_bus.publish(
            topic=topic,
            sender_id=sender_id,
            content=content
        )
    
    async def register_agent(self, agent_id: str, role: str, 
                      output_schema: List[str], 
                      dependencies: Optional[List[str]] = None) -> bool:
        """
        Register a new agent in the memory system.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Role description for the agent
            output_schema: List of fields in the agent's output
            dependencies: List of agent IDs this agent depends on
            
        Returns:
            bool: Success status
        """
        if self.registry:
            return await self.registry.register_agent(agent_id, role, output_schema, dependencies)
        return False
    
    # Additional method stubs will be implemented in the full version