"""
Temporal Event Log for LumosTrade Memory System

This module implements the chronological log of agent activities and system events.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set

import redis

logger = logging.getLogger(__name__)

class TemporalEventLog:
    """
    Chronological log of agent activities and system events.
    
    This component maintains a timeline of all agent activities,
    enabling traceability, explainability, and temporal reasoning.
    """
    
    def __init__(self, memory_core):
        """
        Initialize the Temporal Event Log.
        
        Args:
            memory_core: Reference to the parent MemoryCore instance
        """
        self.memory_core = memory_core
        
        # In-memory event log
        self.events = []
        self.max_in_memory_events = memory_core.config.get("max_in_memory_events", 1000)
        
        # Redis connection for persistence
        self.redis = None
        if memory_core.config.get('use_redis', True):
            self._setup_redis()
    
    def _setup_redis(self):
        """Set up Redis connection for event log persistence"""
        try:
            redis_url = self.memory_core.config.get('redis_url', "redis://localhost:6379/0")
            self.redis = redis.from_url(redis_url)
            logger.info("Connected to Redis for event log persistence")
        except Exception as e:
            logger.error(f"Failed to connect to Redis for event log: {e}")
    
    def _get_event_key(self, session_id=None):
        """Generate Redis key for events"""
        if session_id:
            return f"lumos:memory:events:{session_id}"
        return "lumos:memory:events:global"
    
    async def log_event(self, agent_id: str, event_type: str, 
                       metadata: Dict[str, Any] = None,
                       data: Dict[str, Any] = None, 
                       dependencies: Optional[List[str]] = None) -> str:
        """
        Log an event to the temporal event log.
        
        Args:
            agent_id: ID of the agent generating the event
            event_type: Type of event (e.g., 'store', 'query', 'action')
            metadata: Event metadata (e.g., market information, prices)
            data: Additional event data
            dependencies: Optional list of event IDs this event depends on
            
        Returns:
            str: Event ID
        """
        # Generate unique event ID
        timestamp = datetime.now()
        event_id = f"evt_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Get current session
        session_id = self.memory_core.current_session_id
        
        # Create event entry
        event = {
            "event_id": event_id,
            "agent_id": agent_id,
            "event_type": event_type,
            "timestamp": timestamp.isoformat(),
            "session_id": session_id,
            "dependencies": dependencies or []
        }
        
        # Add metadata or data to the event
        if metadata is not None:
            event["metadata"] = metadata
        if data is not None:
            event["data"] = data
        
        # Add to in-memory log
        self.events.append(event)
        
        # Trim in-memory log if it gets too large
        if len(self.events) > self.max_in_memory_events:
            self.events = self.events[-self.max_in_memory_events:]
        
        return event_id
        
    async def get_recent_events(self, agent_id: str = None, limit: int = 10, 
                            session_id: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve recent events from the temporal log.
        
        Args:
            agent_id: Optional agent ID to filter by
            limit: Maximum number of events to retrieve
            session_id: Optional session ID to filter by (uses current session if None)
            
        Returns:
            List[Dict[str, Any]]: List of event entries
        """
        # Use current session if not specified
        if session_id is None and hasattr(self.memory_core, 'current_session_id'):
            session_id = self.memory_core.current_session_id
            
        # Filter events
        filtered_events = self.events
        
        if agent_id:
            filtered_events = [e for e in filtered_events if e.get("agent_id") == agent_id]
        
        if session_id:
            filtered_events = [e for e in filtered_events if e.get("session_id") == session_id]
            
        # Sort by timestamp (newest first)
        sorted_events = sorted(
            filtered_events,
            key=lambda x: datetime.fromisoformat(x["timestamp"]),
            reverse=True
        )
        
        # Return limited number of events
        return sorted_events[:limit]
        
    async def get_events_by_type(self, event_type: str, agent_id: str = None,
                             session_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve events of a specific type from the temporal log.
        
        Args:
            event_type: Type of events to retrieve
            agent_id: Optional agent ID to filter by
            session_id: Optional session ID to filter by (uses current session if None)
            limit: Maximum number of events to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of event entries
        """
        # Use current session if not specified
        if session_id is None and hasattr(self.memory_core, 'current_session_id'):
            session_id = self.memory_core.current_session_id
            
        # Filter events by type
        filtered_events = [e for e in self.events if e.get("event_type") == event_type]
        
        # Apply additional filters if specified
        if agent_id:
            filtered_events = [e for e in filtered_events if e.get("agent_id") == agent_id]
        
        if session_id:
            filtered_events = [e for e in filtered_events if e.get("session_id") == session_id]
            
        # Sort by timestamp (newest first)
        sorted_events = sorted(
            filtered_events,
            key=lambda x: datetime.fromisoformat(x["timestamp"]),
            reverse=True
        )
        
        # Return limited number of events
        return sorted_events[:limit]
    
    async def get_events(self, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       agent_id: Optional[str] = None,
                       event_type: Optional[str] = None,
                       session_id: Optional[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get events matching the specified filters.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            agent_id: Optional agent ID filter
            event_type: Optional event type filter
            session_id: Optional session ID filter
            limit: Maximum number of events to return
            
        Returns:
            List[Dict]: Matching events
        """
        # Placeholder implementation
        filtered_events = []
        count = 0
        
        # Apply basic filtering to in-memory events
        for event in reversed(self.events):  # Most recent first
            count += 1
            if count > limit:
                break
                
            filtered_events.append(event)
        
        return filtered_events
    
    # Additional methods will be implemented in the full version