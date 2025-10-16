"""
Query Engine for LumosTrade Memory System

This module implements the unified query interface for retrieving information
from the memory system.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from qdrant_client import models

logger = logging.getLogger(__name__)

class QueryEngine:
    """
    Unified query interface for the memory system.
    
    This component provides a high-level query interface for agents,
    abstracting the underlying memory stores and enabling complex queries.
    """
    
    def __init__(self, memory_core):
        """
        Initialize the QueryEngine.
        
        Args:
            memory_core: The MemoryCore instance
        """
        self.memory_core = memory_core
    
    def _build_qdrant_filter(self, query_spec: Dict[str, Any]) -> Optional[models.Filter]:
        """
        Build a Qdrant Filter object from query specifications.
        
        Args:
            query_spec: Query specification dictionary
            
        Returns:
            Qdrant Filter object or None if no filters specified
        """
        filter_conditions = []
        
        # Agent ID filter
        agent_id = query_spec.get("agent_id")
        if agent_id:
            filter_conditions.append(
                models.FieldCondition(
                    key="agent_id",
                    match=models.MatchValue(value=agent_id)
                )
            )
        
        # Memory types filter
        memory_types = query_spec.get("memory_types", [])
        if memory_types:
            # Extract memory types from data.type field
            type_conditions = []
            for mem_type in memory_types:
                type_conditions.append(
                    models.FieldCondition(
                        key="data.type",
                        match=models.MatchValue(value=mem_type)
                    )
                )
            if type_conditions:
                filter_conditions.append(
                    models.Condition(any=type_conditions)
                )
        
        # Tags filter
        tags = query_spec.get("tags", [])
        if tags:
            # Tags are stored in data.tags array
            tag_conditions = []
            for tag in tags:
                tag_conditions.append(
                    models.FieldCondition(
                        key="data.tags",
                        match=models.MatchAny(any=[tag])
                    )
                )
            if tag_conditions:
                filter_conditions.append(
                    models.Condition(any=tag_conditions)
                )
        
        # Time range filters - support both datetime and timestamp ranges
        filters = query_spec.get("filters", {})
        time_start = filters.get("time_start") or filters.get("start_time")
        time_end = filters.get("time_end") or filters.get("end_time")
        
        # Also check for direct timestamp range in filter_spec
        timestamp_range = query_spec.get("timestamp")
        if timestamp_range:
            time_conditions = []
            if "gte" in timestamp_range:
                time_conditions.append(
                    models.FieldCondition(
                        key="timestamp",
                        range=models.Range(gte=timestamp_range["gte"])
                    )
                )
            if "lte" in timestamp_range:
                time_conditions.append(
                    models.FieldCondition(
                        key="timestamp",
                        range=models.Range(lte=timestamp_range["lte"])
                    )
                )
            if time_conditions:
                filter_conditions.extend(time_conditions)
        elif time_start or time_end:
            time_conditions = []
            
            if time_start:
                # Parse time_start if it's a string
                if isinstance(time_start, str):
                    try:
                        time_start = datetime.fromisoformat(time_start.replace('Z', '+00:00'))
                    except ValueError:
                        logger.warning(f"Invalid time_start format: {time_start}")
                        time_start = None
                
                if time_start:
                    time_conditions.append(
                        models.FieldCondition(
                            key="timestamp",
                            range=models.DatetimeRange(gte=time_start)
                        )
                    )
            
            if time_end:
                # Parse time_end if it's a string
                if isinstance(time_end, str):
                    try:
                        time_end = datetime.fromisoformat(time_end.replace('Z', '+00:00'))
                    except ValueError:
                        logger.warning(f"Invalid time_end format: {time_end}")
                        time_end = None
                
                if time_end:
                    time_conditions.append(
                        models.FieldCondition(
                            key="timestamp",
                            range=models.DatetimeRange(lte=time_end)
                        )
                    )
            
            if time_conditions:
                filter_conditions.extend(time_conditions)
        
        # Session ID filter
        session_id = filters.get("session_id")
        if session_id:
            filter_conditions.append(
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id)
                )
            )
        
        # Build the final filter
        if filter_conditions:
            return models.Filter(must=filter_conditions)
        
        return None
    
    async def scroll_query(self, limit: int = 100, offset: Optional[str] = None,
                          filter_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a scroll query for pagination using Qdrant scroll API.
        
        Args:
            limit: Maximum number of results to return
            offset: Offset for pagination (from previous scroll response)
            filter_spec: Filter specifications to build Qdrant filter
            
        Returns:
            Dict: Scroll query results with next_page_offset
        """
        if not self.memory_core.semantic_store or not self.memory_core.semantic_store.vector_client:
            return {
                "results": [],
                "next_page_offset": None
            }
        
        # Build Qdrant filter if filter_spec provided
        qdrant_filter = None
        if filter_spec:
            qdrant_filter = self._build_qdrant_filter(filter_spec)
        
        try:
            # Execute scroll query
            scroll_result = self.memory_core.semantic_store.vector_client.scroll(
                collection_name=self.memory_core.semantic_store.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                scroll_filter=qdrant_filter
            )
            
            # Process results
            results = []
            for point in scroll_result[0]:  # scroll_result is (points, next_offset)
                memory_data = {
                    "memory_id": point.payload.get("memory_id"),
                    "agent_id": point.payload.get("agent_id"),
                    "session_id": point.payload.get("session_id"),
                    "timestamp": point.payload.get("timestamp"),
                    "data": point.payload.get("data")
                }
                results.append(memory_data)
            
            return {
                "results": results,
                "next_page_offset": scroll_result[1]  # Next offset for pagination
            }
            
        except Exception as e:
            logger.error(f"Error executing scroll query: {e}")
            return {
                "results": [],
                "next_page_offset": None
            }
    
    async def query(self, query: str = None, agent_id: str = None, 
                limit: int = 10, query_type: str = None,
                memory_types: List[str] = None, tags: List[str] = None,
                filters: Dict[str, Any] = None, query_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a query across memory stores.
        
        Args:
            query: Natural language or keyword query string
            agent_id: ID of the agent to query memories for
            limit: Maximum number of results to return
            query_type: Type of query (semantic, temporal, episodic, hybrid)
            filters: Additional filters to apply
            query_spec: Structured query specification (legacy parameter)
                {
                    "type": "semantic" | "temporal" | "episodic" | "hybrid",
                    "filters": {...},
                    "limit": 10,
                    "sort": [...],
                    "include_dependencies": False,
                    ...
                }
            
        Returns:
            Dict: Query results
        """
        # Normalize parameters
        if query_spec is None:
            query_spec = {
                "query": query,
                "agent_id": agent_id,
                "limit": limit,
                "filters": filters or {}
            }
            
            # Add additional filters if provided
            if memory_types:
                query_spec["memory_types"] = memory_types
            if tags:
                query_spec["tags"] = tags
            
        if query_type is not None:
            query_spec["type"] = query_type
        
        query_type = query_spec.get("type", "episodic")
        
        if query_type == "semantic":
            return await self._semantic_query(query_spec)
        elif query_type == "temporal":
            return await self._temporal_query(query_spec)
        elif query_type == "episodic":
            return await self._episodic_query(query_spec)
        elif query_type == "hybrid":
            return await self._hybrid_query(query_spec)
        else:
            logger.error(f"Unsupported query type: {query_type}")
            return {"error": f"Unsupported query type: {query_type}"}
    
    async def _semantic_query(self, query_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a semantic query using Qdrant vector search with filters"""
        query = query_spec.get("query", "")
        limit = query_spec.get("limit", 10)
        
        if not self.memory_core.semantic_store:
            return {
                "query_type": "semantic",
                "query_text": query,
                "result_count": 0,
                "results": []
            }
        
        # Build Qdrant filter from query specifications
        qdrant_filter = self._build_qdrant_filter(query_spec)
        
        # Call the semantic store search method with Qdrant filter
        results = await self.memory_core.semantic_store.search(
            query=query,
            limit=limit,
            score_threshold=0.0,  # Lower threshold to find more results
            qdrant_filter=qdrant_filter
        )
        
        return {
            "query_type": "semantic",
            "query_text": query,
            "result_count": len(results),
            "results": results
        }
    
    async def _temporal_query(self, query_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a temporal query using Qdrant datetime filtering with range queries"""
        agent_id = query_spec.get("agent_id")
        limit = query_spec.get("limit", 10)
        filters = query_spec.get("filters", {})
        
        if not self.memory_core.semantic_store:
            return {
                "query_type": "temporal",
                "result_count": 0,
                "results": []
            }
        
        # Build filter specification with datetime range
        filter_spec = {}
        if agent_id:
            filter_spec["agent_ids"] = [agent_id]
        
        # Add datetime range filtering using Qdrant range queries
        start_time = filters.get("start_time")
        end_time = filters.get("end_time")
        
        if start_time or end_time:
            timestamp_filter = {}
            if start_time:
                # Convert datetime to timestamp if needed
                if isinstance(start_time, str):
                    from datetime import datetime
                    start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                if hasattr(start_time, 'timestamp'):
                    timestamp_filter["gte"] = start_time.timestamp()
                else:
                    timestamp_filter["gte"] = start_time
            if end_time:
                # Convert datetime to timestamp if needed
                if isinstance(end_time, str):
                    from datetime import datetime
                    end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                if hasattr(end_time, 'timestamp'):
                    timestamp_filter["lte"] = end_time.timestamp()
                else:
                    timestamp_filter["lte"] = end_time
            filter_spec["timestamp"] = timestamp_filter
        
        # Build Qdrant filter
        qdrant_filter = self._build_qdrant_filter(filter_spec)
        
        # For temporal queries, we search with empty query to get all memories in time range
        results = await self.memory_core.semantic_store.search(
            query="",  # Empty query for time-based retrieval
            limit=limit,
            score_threshold=0.0,
            qdrant_filter=qdrant_filter
        )
        
        return {
            "query_type": "temporal",
            "agent_id": agent_id,
            "result_count": len(results),
            "results": results
        }
    
    async def _episodic_query(self, query_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an episodic memory query"""
        # Stub implementation
        return {
            "query_type": "episodic",
            "agent_id": query_spec.get("agent_id", ""),
            "result_count": 0,
            "results": []
        }
    
    async def _hybrid_query(self, query_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a hybrid query combining semantic and temporal filtering"""
        query = query_spec.get("query", "")
        limit = query_spec.get("limit", 10)
        
        if not self.memory_core.semantic_store:
            return {
                "query_type": "hybrid",
                "result_count": 0,
                "results": []
            }
        
        # Build Qdrant filter for complex filtering
        qdrant_filter = self._build_qdrant_filter(query_spec)
        
        # Perform semantic search with filters
        results = await self.memory_core.semantic_store.search(
            query=query,
            limit=limit,
            score_threshold=0.0,
            qdrant_filter=qdrant_filter
        )
        
        return {
            "query_type": "hybrid",
            "query_text": query,
            "result_count": len(results),
            "results": results
        }
    
    async def get_agent_context(self, agent_id: str, 
                              context_size: int = 10) -> Dict[str, Any]:
        """
        Get the current context for an agent using Qdrant filtering.
        
        Args:
            agent_id: ID of the agent
            context_size: Number of recent items to include
            
        Returns:
            Dict: Agent context
        """
        if not self.memory_core.semantic_store:
            return {
                "agent_id": agent_id,
                "session_id": self.memory_core.get_current_session(),
                "own_memories": [],
                "dependency_memories": {},
                "recent_events": []
            }
        
        # Build Qdrant filter for agent-specific context
        query_spec = {
            "agent_id": agent_id,
            "limit": context_size
        }
        qdrant_filter = self._build_qdrant_filter(query_spec)
        
        # Get memories for this agent using Qdrant filter
        agent_memories = await self.memory_core.semantic_store.search(
            query="",  # Empty query to get all memories for this agent
            limit=context_size,
            score_threshold=0.0,
            qdrant_filter=qdrant_filter
        )
        
        return {
            "agent_id": agent_id,
            "session_id": self.memory_core.get_current_session(),
            "own_memories_count": len(agent_memories),
            "own_memories": agent_memories,
            "dependency_memories": {},
            "recent_events": []
        }
    
    # Additional methods will be implemented in the full version