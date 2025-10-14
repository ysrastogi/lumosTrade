"""
Episodic Memory Store for LumosTrade Memory System

This module implements the short-term memory store for agent outputs within a session.
"""

import asyncio
import json
import logging
import os
import pickle
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

import redis

logger = logging.getLogger(__name__)

class EpisodicMemoryStore:
    """
    Short-term memory store for agent outputs within a session.
    
    This component handles the storage and retrieval of recent agent outputs,
    providing quick access to the current context of each agent.
    """
    
    def __init__(self, memory_core, use_redis: bool = True, 
                redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize the Episodic Memory Store.
        
        Args:
            memory_core: Reference to the parent MemoryCore instance
            use_redis: Whether to use Redis for persistence
            redis_url: Redis connection URL
        """
        self.memory_core = memory_core
        self.use_redis = use_redis
        self.redis_url = redis_url
        
        # In-memory storage for fast access
        self.memory_sessions = {}  # session_id -> {agent_id -> [memory entries]}
        self.memory_index = {}  # memory_id -> {session_id, agent_id, data}
        
        # Redis connection for persistence
        self.redis = None
        if use_redis:
            self._connect_redis()
    
    def _connect_redis(self):
        """Connect to Redis for persistence"""
        try:
            self.redis = redis.from_url(self.redis_url)
            logger.info("Connected to Redis for episodic memory store")
        except Exception as e:
            logger.error(f"Failed to connect to Redis for episodic memory: {e}")
            self.redis = None
    
    def _get_memory_key(self, session_id, agent_id):
        """Generate Redis key for memory entries"""
        return f"lumos:memory:episodic:{session_id}:{agent_id}"
    
    async def store(self, agent_id: str, memory_type: str = None, 
                   content: Dict[str, Any] = None, tags: List[str] = None, 
                   data: Dict[str, Any] = None, session_id: str = None) -> str:
        """
        Store an agent output in episodic memory.
        
        Args:
            agent_id: ID of the agent storing data
            memory_type: Type of memory being stored (observation, decision, etc.)
            content: Content to store
            tags: Tags for categorization
            data: Legacy parameter for output data to store
            session_id: Session ID (uses active session if None)
            
        Returns:
            str: Memory entry ID
        """
        # Use current session if not specified
        if session_id is None:
            session_id = self.memory_core.current_session_id
            
        # Generate unique memory ID
        timestamp = datetime.now()
        memory_id = f"mem_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Process data from either content or data parameter
        memory_data = {}
        if data is not None:
            memory_data = data
        elif content is not None:
            memory_data = content
        
        # Add metadata to the memory entry
        entry = {
            "memory_id": memory_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "timestamp": timestamp.isoformat(),
            "memory_type": memory_type,
            "tags": tags or [],
            "data": memory_data
        }
        
        # Store in memory index
        self.memory_index[memory_id] = entry
        
        # Initialize session and agent structures if needed
        if session_id not in self.memory_sessions:
            self.memory_sessions[session_id] = {}
        
        if agent_id not in self.memory_sessions[session_id]:
            self.memory_sessions[session_id][agent_id] = []
        
        # Add to in-memory storage
        self.memory_sessions[session_id][agent_id].append(entry)
        
        return memory_id
    
    async def get_recent(self, agent_id: str, limit: int = 10, 
                     session_id: str = None, memory_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve recent memories for an agent.
        
        Args:
            agent_id: ID of the agent
            limit: Maximum number of memories to retrieve
            session_id: Session ID (uses active session if None)
            memory_types: Optional list of memory types to filter
            
        Returns:
            List[Dict[str, Any]]: List of memory entries
        """
        # Use current session if not specified
        if session_id is None:
            session_id = self.memory_core.current_session_id
            
        if session_id not in self.memory_sessions or agent_id not in self.memory_sessions[session_id]:
            return []
            
        # Get agent memories and sort by timestamp (newest first)
        memories = self.memory_sessions[session_id][agent_id]
        
        # Filter by memory types if specified
        if memory_types:
            memories = [m for m in memories if m.get("memory_type") in memory_types]
        
        sorted_memories = sorted(
            memories, 
            key=lambda x: datetime.fromisoformat(x["timestamp"]),
            reverse=True
        )
        
        # Return limited number of entries
        return sorted_memories[:limit]
        
    async def get_by_type(self, agent_id: str, memory_type: str, 
                       session_id: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve memories of a specific type for an agent.
        
        Args:
            agent_id: ID of the agent
            memory_type: Type of memories to retrieve
            session_id: Session ID (uses active session if None)
            
        Returns:
            List[Dict[str, Any]]: List of memory entries of the specified type
        """
        # Use current session if not specified
        if session_id is None:
            session_id = self.memory_core.current_session_id
            
        if session_id not in self.memory_sessions or agent_id not in self.memory_sessions[session_id]:
            return []
            
        # Filter memories by type
        memories = self.memory_sessions[session_id][agent_id]
        filtered_memories = [
            memory for memory in memories 
            if memory.get("memory_type") == memory_type
        ]
        
        # Sort by timestamp (newest first)
        sorted_memories = sorted(
            filtered_memories,
            key=lambda x: datetime.fromisoformat(x["timestamp"]),
            reverse=True
        )
        
        return sorted_memories
    
    async def get(self, agent_id: str, session_id: str, 
                filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve agent memories matching filters.
        
        Args:
            agent_id: ID of the agent to retrieve data for
            filters: Optional filters to apply to the query
            session_id: Session ID
            
        Returns:
            List[Dict]: Matching memory entries
        """
        # Check if session and agent exist in memory
        if session_id not in self.memory_sessions or \
           agent_id not in self.memory_sessions[session_id]:
            return []
        
        # Get all memories for the agent in the session
        memories = self.memory_sessions[session_id][agent_id]
        
        # Apply filters if provided
        if filters:
            filtered_memories = []
            for memory in memories:
                match = True
                for key, value in filters.items():
                    # Simple filter implementation
                    if key not in memory or memory[key] != value:
                        match = False
                
                if match:
                    filtered_memories.append(memory)
            
            return filtered_memories
        
        return memories
    
    # Additional methods will be implemented in the full version