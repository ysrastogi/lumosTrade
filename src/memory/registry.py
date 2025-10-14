"""
Agent Registry for LumosTrade Memory System

This module implements the registry component that manages agent metadata,
registrations, and subscriptions.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set

import redis

logger = logging.getLogger(__name__)

class AgentRegistry:
    """
    Central directory for all registered agents in the memory system.
    
    This component manages agent metadata, registrations, and subscription mappings.
    It serves as the reference point for agent discovery and message routing.
    """
    
    def __init__(self, memory_core):
        """
        Initialize the Agent Registry.
        
        Args:
            memory_core: Reference to the parent MemoryCore instance
        """
        self.memory_core = memory_core
        self.agents = {}  # Dictionary to store agent metadata
        self.subscriptions = {}  # Dictionary to track topic subscriptions
        self.lock = asyncio.Lock()  # Lock for thread safety
        
        # Redis connection for persistence (if enabled)
        self.redis = None
        if memory_core.config.get('use_redis', True):
            self._get_redis_connection()
    
    def _get_redis_connection(self):
        """Get Redis connection from memory core config"""
        try:
            redis_url = self.memory_core.config.get('redis_url', "redis://localhost:6379/0")
            self.redis = redis.from_url(redis_url)
            logger.info("Connected to Redis for agent registry")
        except Exception as e:
            logger.error(f"Failed to connect to Redis for agent registry: {e}")
            self.redis = None
    
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
        async with self.lock:
            if agent_id in self.agents:
                logger.warning(f"Agent {agent_id} is already registered")
                return False
                
            # Create agent metadata
            self.agents[agent_id] = {
                "id": agent_id,
                "role": role,
                "output_schema": output_schema,
                "dependencies": dependencies or [],
                "registered_at": datetime.now().isoformat(),
                "active": True,
                "last_active": datetime.now().isoformat()
            }
            
            logger.info(f"Agent {agent_id} registered with role: {role}")
            return True
    
    async def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Dict or None: Agent metadata if found, None otherwise
        """
        if agent_id not in self.agents:
            return None
            
        # Update last active timestamp
        self.agents[agent_id]["last_active"] = datetime.now().isoformat()
        return self.agents[agent_id]
    
    async def subscribe_agent_to_topic(self, agent_id: str, topic: str) -> bool:
        """
        Subscribe an agent to a specific topic.
        
        Args:
            agent_id: Unique identifier for the agent
            topic: Topic name to subscribe to
            
        Returns:
            bool: Success status
        """
        if agent_id not in self.agents:
            logger.warning(f"Cannot subscribe unknown agent {agent_id} to topic {topic}")
            return False
            
        async with self.lock:
            if topic not in self.subscriptions:
                self.subscriptions[topic] = set()
                
            self.subscriptions[topic].add(agent_id)
            logger.info(f"Agent {agent_id} subscribed to topic: {topic}")
            return True
    
    async def unsubscribe_agent_from_topic(self, agent_id: str, topic: str) -> bool:
        """
        Unsubscribe an agent from a specific topic.
        
        Args:
            agent_id: Unique identifier for the agent
            topic: Topic name to unsubscribe from
            
        Returns:
            bool: Success status
        """
        if topic not in self.subscriptions or agent_id not in self.subscriptions[topic]:
            logger.warning(f"Agent {agent_id} is not subscribed to topic {topic}")
            return False
            
        async with self.lock:
            self.subscriptions[topic].remove(agent_id)
            logger.info(f"Agent {agent_id} unsubscribed from topic: {topic}")
            return True
    
    async def get_topic_subscribers(self, topic: str) -> List[str]:
        """
        Get all agents subscribed to a specific topic.
        
        Args:
            topic: Topic name to check
            
        Returns:
            List[str]: List of agent IDs subscribed to the topic
        """
        if topic not in self.subscriptions:
            return []
            
        return list(self.subscriptions[topic])
    
    # Additional methods will be implemented in the full version