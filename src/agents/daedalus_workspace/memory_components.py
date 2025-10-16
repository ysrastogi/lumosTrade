"""
Memory Components for Daedalus Agent

This module implements specialized memory components for the Daedalus agent system:
- AthenaMemory: Knowledge and information processing
- ApolloMemory: Prediction and pattern recognition
- ChronosMemory: Temporal reasoning and time-based analysis
"""

from typing import List, Dict, Any, Optional, Union
import uuid
import json
import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta

try:
    from src.memory.memory_core import MemoryCore
    from src.memory.episodic_store import EpisodicMemoryStore
    from src.memory.semantic_store import SemanticMemoryStore
    from src.memory.temporal_log import TemporalEventLog
    from src.memory.message_bus import MessageBus
    MEMORY_SYSTEM_AVAILABLE = True
except ImportError:
    # Fall back to local implementation if global memory system isn't available
    MEMORY_SYSTEM_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


class AthenaMemory:
    """
    Knowledge and information processing component of Daedalus memory.
    
    Athena is responsible for storing and retrieving structured knowledge,
    information synthesis, and maintaining the wisdom base.
    """
    
    def __init__(self, parent_memory, agent_id: str):
        """
        Initialize AthenaMemory.
        
        Args:
            parent_memory: Reference to the parent memory manager
            agent_id: ID of the agent using this memory component
        """
        self.parent = parent_memory
        self.agent_id = agent_id
        self.memory_core = getattr(parent_memory, 'memory_core', None)
        self.use_global_memory = (
            getattr(parent_memory, 'use_global_memory', False) and 
            MEMORY_SYSTEM_AVAILABLE and 
            self.memory_core is not None
        )
        
        # Local knowledge cache for quick access
        self.knowledge_cache = {}
        
    async def store_knowledge(self, content: Dict[str, Any], 
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store structured knowledge in the semantic memory.
        
        Args:
            content: Knowledge content
            metadata: Optional metadata
            
        Returns:
            str: Memory entry ID
        """
        # Generate a unique ID for this knowledge entry
        timestamp = datetime.now()
        knowledge_id = f"knowledge_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Prepare entry metadata
        meta = metadata or {}
        meta.update({
            "component": "athena",
            "knowledge_type": content.get("type", "general"),
            "source": meta.get("source", "daedalus_analysis")
        })
        
        # Update local cache
        self.knowledge_cache[knowledge_id] = {
            "content": content,
            "metadata": meta,
            "timestamp": timestamp.isoformat()
        }
        
        # Store in global memory if available
        if self.use_global_memory:
            try:
                # Store in semantic memory
                await self.memory_core.semantic_store.store(
                    agent_id=self.agent_id,
                    content=content,
                    metadata=meta
                )
                
                # Log the event
                await self.memory_core.temporal_log.log_event(
                    agent_id=self.agent_id,
                    event_type="knowledge_stored",
                    data={"memory_id": knowledge_id, "content_summary": content.get("summary", "")},
                    metadata=meta
                )
            except Exception as e:
                logger.error(f"Failed to store knowledge in global memory: {e}")
        
        return knowledge_id
        
    async def retrieve_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge relevant to the given query.
        
        Args:
            query: The query string
            limit: Maximum number of results
            
        Returns:
            List of knowledge entries
        """
        results = []
        
        # Try global memory first if available
        if self.use_global_memory:
            try:
                # Query the semantic store
                global_results = await self.memory_core.query_engine.semantic_search(
                    agent_id=self.agent_id,
                    query=query,
                    limit=limit,
                    metadata_filter={"component": "athena"}
                )
                
                results.extend(global_results)
                
                # Log the retrieval
                await self.memory_core.temporal_log.log_event(
                    agent_id=self.agent_id,
                    event_type="knowledge_retrieved",
                    data={"query": query, "result_count": len(results)}
                )
            except Exception as e:
                logger.error(f"Failed to retrieve knowledge from global memory: {e}")
        
        # Fall back to local cache if needed
        if not results:
            # Simple keyword matching in local cache
            query_terms = query.lower().split()
            for key, entry in self.knowledge_cache.items():
                content_str = json.dumps(entry["content"]).lower()
                
                # Check if all query terms are in the content
                if all(term in content_str for term in query_terms):
                    results.append({
                        "id": key,
                        "content": entry["content"],
                        "metadata": entry["metadata"],
                        "timestamp": entry["timestamp"]
                    })
                    
                    # Limit results
                    if len(results) >= limit:
                        break
        
        return results
        
    async def synthesize_insights(self, query: str) -> Dict[str, Any]:
        """
        Synthesize insights from multiple knowledge entries.
        
        Args:
            query: The question or topic to synthesize insights for
            
        Returns:
            Dictionary containing the synthesized insights
        """
        # Retrieve relevant knowledge
        knowledge_entries = await self.retrieve_knowledge(query, limit=10)
        
        if not knowledge_entries:
            return {
                "insight": "Insufficient knowledge available to synthesize insights on this topic.",
                "sources": [],
                "confidence": 0.1
            }
            
        if self.use_global_memory and hasattr(self.memory_core, 'summarizer'):
            try:
                # Use summarization engine to synthesize
                insight = await self.memory_core.summarizer.summarize(
                    content=[entry["content"] for entry in knowledge_entries],
                    prompt=f"Synthesize insights about: {query}"
                )
            except Exception as e:
                logger.error(f"Failed to use global summarizer: {e}")
                insight = self._local_synthesize_insights(knowledge_entries, query)
        else:
            insight = self._local_synthesize_insights(knowledge_entries, query)
        
        # Store the synthesis as new knowledge
        synthesis_id = await self.store_knowledge({
            "type": "synthesis",
            "query": query,
            "insight": insight,
            "sources": [entry["id"] for entry in knowledge_entries],
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "id": synthesis_id,
            "insight": insight,
            "source_count": len(knowledge_entries),
            "confidence": min(0.3 + (len(knowledge_entries) * 0.05), 0.9)  # Simple confidence heuristic
        }
    
    def _local_synthesize_insights(self, knowledge_entries: List[Dict[str, Any]], query: str) -> str:
        """Local fallback for insight synthesis when global summarizer is unavailable"""
        # Simple insight extraction
        key_points = []
        
        # Extract key information from each entry
        for entry in knowledge_entries:
            content = entry["content"]
            if "summary" in content:
                key_points.append(content["summary"])
            elif "insight" in content:
                key_points.append(content["insight"])
                
        # Combine key points
        if key_points:
            return "Key insights:\n- " + "\n- ".join(key_points)
        else:
            return f"No specific insights available for query: {query}"


class ApolloMemory:
    """
    Prediction and pattern recognition component of Daedalus memory.
    
    Apollo is responsible for storing predictions, tracking their accuracy,
    identifying patterns, and learning from historical data.
    """
    
    def __init__(self, parent_memory, agent_id: str):
        """
        Initialize ApolloMemory.
        
        Args:
            parent_memory: Reference to the parent memory manager
            agent_id: ID of the agent using this memory component
        """
        self.parent = parent_memory
        self.agent_id = agent_id
        self.memory_core = getattr(parent_memory, 'memory_core', None)
        self.use_global_memory = (
            getattr(parent_memory, 'use_global_memory', False) and 
            MEMORY_SYSTEM_AVAILABLE and 
            self.memory_core is not None
        )
        
        # Prediction tracking
        self.predictions = {}
        self.accuracy_history = defaultdict(list)
        
    async def store_prediction(self, prediction: Dict[str, Any], 
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a prediction in memory.
        
        Args:
            prediction: The prediction content
            metadata: Optional metadata
            
        Returns:
            str: Prediction ID
        """
        # Generate prediction ID
        timestamp = datetime.now()
        prediction_id = f"pred_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Prepare entry metadata
        meta = metadata or {}
        meta.update({
            "component": "apollo",
            "prediction_type": prediction.get("type", "general")
        })
        
        # Add timestamps
        prediction["created_at"] = timestamp.isoformat()
        prediction["target_time"] = prediction.get("target_time") or "unknown"
        
        # Add to local tracking
        self.predictions[prediction_id] = {
            "content": prediction,
            "metadata": meta,
            "verified": False,
            "outcome": None,
            "created_at": timestamp
        }
        
        # Store in global memory if available
        if self.use_global_memory:
            try:
                # Store in episodic memory
                await self.memory_core.episodic_store.store(
                    session_id=self.memory_core.current_session_id,
                    agent_id=self.agent_id,
                    memory_id=prediction_id,
                    content=prediction,
                    metadata=meta
                )
                
                # Log the prediction event
                await self.memory_core.temporal_log.log_event(
                    agent_id=self.agent_id,
                    event_type="prediction_created",
                    data={
                        "prediction_id": prediction_id,
                        "subject": prediction.get("subject", "unknown"),
                        "confidence": prediction.get("confidence", 0)
                    }
                )
            except Exception as e:
                logger.error(f"Failed to store prediction in global memory: {e}")
        
        return prediction_id
        
    async def verify_prediction(self, prediction_id: str, 
                              outcome: bool, 
                              evidence: Optional[Dict[str, Any]] = None) -> bool:
        """
        Verify a prediction against actual outcome.
        
        Args:
            prediction_id: ID of the prediction to verify
            outcome: Whether the prediction was correct
            evidence: Optional evidence supporting the verification
            
        Returns:
            bool: Success status
        """
        if prediction_id not in self.predictions:
            logger.warning(f"Prediction {prediction_id} not found")
            return False
        
        # Update prediction record
        pred_record = self.predictions[prediction_id]
        pred_record["verified"] = True
        pred_record["outcome"] = outcome
        pred_record["verified_at"] = datetime.now()
        
        if evidence:
            pred_record["evidence"] = evidence
        
        # Track accuracy for this prediction type
        pred_type = pred_record["metadata"].get("prediction_type", "general")
        self.accuracy_history[pred_type].append(outcome)
        
        # Update in global memory if available
        if self.use_global_memory:
            try:
                # Log the verification event
                await self.memory_core.temporal_log.log_event(
                    agent_id=self.agent_id,
                    event_type="prediction_verified",
                    data={
                        "prediction_id": prediction_id,
                        "outcome": outcome,
                        "evidence": evidence,
                        "type": pred_type
                    }
                )
                
                # Update the prediction record
                original_prediction = pred_record["content"]
                updated_prediction = original_prediction.copy()
                updated_prediction["verified"] = True
                updated_prediction["outcome"] = outcome
                updated_prediction["verified_at"] = datetime.now().isoformat()
                
                if evidence:
                    updated_prediction["evidence"] = evidence
                
                # Store updated prediction
                await self.memory_core.episodic_store.store(
                    session_id=self.memory_core.current_session_id,
                    agent_id=self.agent_id,
                    memory_id=prediction_id,
                    content=updated_prediction,
                    metadata=pred_record["metadata"]
                )
                
                # Publish verification message
                if hasattr(self.memory_core, 'message_bus'):
                    await self.memory_core.message_bus.publish(
                        topic=f"{self.agent_id}.predictions.verified",
                        message={
                            "prediction_id": prediction_id,
                            "outcome": outcome,
                            "prediction_type": pred_type,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to update prediction in global memory: {e}")
        
        return True
        
    async def retrieve_predictions(self, query: str = None, 
                                prediction_type: str = None,
                                verified_only: bool = False,
                                limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve predictions relevant to the given criteria.
        
        Args:
            query: Optional search query
            prediction_type: Optional type of prediction to filter by
            verified_only: Whether to only include verified predictions
            limit: Maximum number of results
            
        Returns:
            List of prediction entries
        """
        results = []
        
        # Try global memory first if available
        if self.use_global_memory:
            try:
                # Prepare metadata filter
                metadata_filter = {"component": "apollo"}
                if prediction_type:
                    metadata_filter["prediction_type"] = prediction_type
                
                # Search episodic memory
                global_results = []
                if query:
                    global_results = await self.memory_core.query_engine.search(
                        agent_id=self.agent_id,
                        query=query,
                        limit=limit * 2,  # Get more than needed for filtering
                        metadata_filter=metadata_filter
                    )
                else:
                    # Get recent predictions if no query provided
                    memories = await self.memory_core.episodic_store.get_recent(
                        agent_id=self.agent_id,
                        limit=limit * 2,
                        metadata_filter=metadata_filter
                    )
                    global_results = [
                        {"id": m["id"], "content": m["content"], "metadata": m["metadata"]}
                        for m in memories
                    ]
                
                # Filter verified if requested
                if verified_only:
                    global_results = [
                        r for r in global_results
                        if r.get("content", {}).get("verified", False)
                    ]
                
                results.extend(global_results[:limit])
            except Exception as e:
                logger.error(f"Failed to retrieve predictions from global memory: {e}")
        
        # Fall back to local cache if needed or combine with local results
        if not results or len(results) < limit:
            remaining_limit = limit - len(results)
            
            # Filter predictions from local cache
            local_results = []
            for pred_id, pred in self.predictions.items():
                if verified_only and not pred.get("verified"):
                    continue
                    
                if prediction_type and pred["metadata"].get("prediction_type") != prediction_type:
                    continue
                    
                if query:
                    # Simple keyword matching
                    content_str = json.dumps(pred["content"]).lower()
                    if query.lower() not in content_str:
                        continue
                
                local_results.append({
                    "id": pred_id,
                    "content": pred["content"],
                    "metadata": pred["metadata"]
                })
                
                if len(local_results) >= remaining_limit:
                    break
            
            results.extend(local_results)
        
        return results
        
    def get_accuracy_stats(self, prediction_type: Optional[str] = None) -> Dict[str, float]:
        """
        Get accuracy statistics for predictions.
        
        Args:
            prediction_type: Optional type to filter by
            
        Returns:
            Dictionary of accuracy statistics
        """
        stats = {}
        
        if prediction_type:
            # Get stats for specific type
            if prediction_type in self.accuracy_history:
                history = self.accuracy_history[prediction_type]
                stats[prediction_type] = sum(history) / len(history) if history else 0
        else:
            # Get stats for all types
            for pred_type, history in self.accuracy_history.items():
                stats[pred_type] = sum(history) / len(history) if history else 0
            
            # Overall accuracy
            all_predictions = [outcome for history in self.accuracy_history.values() 
                              for outcome in history]
            stats["overall"] = sum(all_predictions) / len(all_predictions) if all_predictions else 0
            
        return stats


class ChronosMemory:
    """
    Temporal reasoning and time-based analysis component of Daedalus memory.
    
    Chronos is responsible for tracking event sequences, temporal patterns,
    and maintaining a coherent timeline of market and agent activities.
    """
    
    def __init__(self, parent_memory, agent_id: str):
        """
        Initialize ChronosMemory.
        
        Args:
            parent_memory: Reference to the parent memory manager
            agent_id: ID of the agent using this memory component
        """
        self.parent = parent_memory
        self.agent_id = agent_id
        self.memory_core = getattr(parent_memory, 'memory_core', None)
        self.use_global_memory = (
            getattr(parent_memory, 'use_global_memory', False) and 
            MEMORY_SYSTEM_AVAILABLE and 
            self.memory_core is not None
        )
        
        # Timeline and sequence tracking
        self.timelines = defaultdict(list)  # timeline_id -> events
        self.active_timeline_id = f"timeline_{uuid.uuid4().hex[:8]}"
        
        # Market regime tracking
        self.regimes = {}
        self.current_regime = None
        
        # Risk management
        self.risk_violations = []
        self.position_history = []
        self.portfolio_snapshots = []
        
    async def create_timeline(self, name: str, description: Optional[str] = None) -> str:
        """
        Create a new timeline for tracking temporal events.
        
        Args:
            name: Name of the timeline
            description: Optional description
            
        Returns:
            Timeline ID
        """
        # Create unique timeline ID
        timestamp = datetime.now()
        timeline_id = f"timeline_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Store locally
        self.timelines[timeline_id] = []
        
        # Store in global memory if available
        if self.use_global_memory:
            try:
                # Store timeline metadata in semantic store
                await self.memory_core.semantic_store.store(
                    agent_id=self.agent_id,
                    content={
                        "type": "timeline_metadata",
                        "name": name,
                        "description": description or f"Timeline created by {self.agent_id}",
                        "created_at": timestamp.isoformat()
                    },
                    metadata={
                        "object_type": "timeline",
                        "timeline_id": timeline_id
                    }
                )
                
                # Log timeline creation event
                await self.memory_core.temporal_log.log_event(
                    agent_id=self.agent_id,
                    event_type="timeline_created",
                    data={
                        "timeline_id": timeline_id,
                        "name": name,
                        "description": description
                    }
                )
            except Exception as e:
                logger.error(f"Failed to create timeline in global memory: {e}")
        
        return timeline_id
    
    async def set_active_timeline(self, timeline_id: str) -> bool:
        """
        Set the active timeline for subsequent events.
        
        Args:
            timeline_id: ID of timeline to set as active
            
        Returns:
            bool: Success status
        """
        # Verify timeline exists
        if timeline_id in self.timelines or timeline_id.startswith("timeline_"):
            self.active_timeline_id = timeline_id
            return True
        return False
    
    async def store_temporal_observation(self, 
                                       content: Dict[str, Any], 
                                       timeline_id: Optional[str] = None,
                                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a temporal observation in memory.
        
        Args:
            content: The observation content
            timeline_id: Optional timeline ID (uses active timeline if None)
            metadata: Optional metadata
            
        Returns:
            str: Observation ID
        """
        # Generate observation ID and use specified or active timeline
        timestamp = datetime.now()
        observation_id = f"obs_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        timeline_id = timeline_id or self.active_timeline_id
        
        # Prepare entry metadata
        meta = metadata or {}
        meta.update({
            "component": "chronos",
            "timeline_id": timeline_id,
            "temporal_type": content.get("temporal_type", "event")
        })
        
        # Add timestamps if not present
        if "timestamp" not in content:
            content["timestamp"] = timestamp.isoformat()
            
        # Add to local timeline
        self.timelines[timeline_id].append({
            "id": observation_id,
            "content": content,
            "metadata": meta,
            "timestamp": timestamp
        })
        
        # Keep timelines sorted by timestamp
        self.timelines[timeline_id].sort(key=lambda x: x["timestamp"])
        
        # Store in global memory if available
        if self.use_global_memory:
            try:
                # Store in temporal log
                await self.memory_core.temporal_log.log_event(
                    agent_id=self.agent_id,
                    event_type="temporal_observation",
                    data=content,
                    metadata=meta
                )
                
                # Store full observation in episodic memory
                await self.memory_core.episodic_store.store(
                    session_id=self.memory_core.current_session_id,
                    agent_id=self.agent_id,
                    memory_id=observation_id,
                    content={
                        "observation_id": observation_id,
                        "timeline_id": timeline_id,
                        "content": content,
                        "timestamp": timestamp.isoformat()
                    },
                    metadata=meta
                )
            except Exception as e:
                logger.error(f"Failed to store temporal observation in global memory: {e}")
        
        return observation_id
        
    async def retrieve_timeline_events(self, 
                                    timeline_id: Optional[str] = None, 
                                    start_time: Optional[Union[datetime, str]] = None,
                                    end_time: Optional[Union[datetime, str]] = None,
                                    event_types: Optional[List[str]] = None,
                                    limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve events from a timeline.
        
        Args:
            timeline_id: Optional timeline ID (uses active timeline if None)
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            event_types: Optional list of event types to filter by
            limit: Maximum number of events to retrieve
            
        Returns:
            List of timeline events
        """
        # Use active timeline if none specified
        timeline_id = timeline_id or self.active_timeline_id
        
        # Convert string times to datetime if needed
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)
        
        results = []
        
        # Try global memory first if available
        if self.use_global_memory:
            try:
                # Query by metadata to get events for this timeline
                events = await self.memory_core.temporal_log.get_events_by_metadata(
                    metadata_key="timeline_id",
                    metadata_value=timeline_id,
                    limit=limit * 2  # Get more than needed for filtering
                )
                
                # Filter by time range and event types
                filtered_events = []
                for event in events:
                    event_time = datetime.fromisoformat(event["timestamp"])
                    
                    # Apply time filters
                    if start_time and event_time < start_time:
                        continue
                    if end_time and event_time > end_time:
                        continue
                        
                    # Apply event type filter
                    if event_types and event.get("event_type") not in event_types:
                        continue
                        
                    filtered_events.append(event)
                
                # Sort by timestamp (oldest to newest)
                sorted_events = sorted(
                    filtered_events,
                    key=lambda x: datetime.fromisoformat(x["timestamp"])
                )
                
                results.extend(sorted_events[:limit])
            except Exception as e:
                logger.error(f"Failed to retrieve timeline events from global memory: {e}")
        
        # Fall back to local timeline or combine with local results
        if not results or len(results) < limit:
            remaining_limit = limit - len(results)
            local_events = self.timelines.get(timeline_id, [])
            
            # Filter local events
            filtered_local = []
            for event in local_events:
                # Apply time filters
                if start_time and event["timestamp"] < start_time:
                    continue
                if end_time and event["timestamp"] > end_time:
                    continue
                    
                # Apply event type filter
                event_type = event.get("content", {}).get("temporal_type")
                if event_types and event_type not in event_types:
                    continue
                    
                filtered_local.append(event)
            
            # Sort and limit
            sorted_local = sorted(filtered_local, key=lambda x: x["timestamp"])
            results.extend(sorted_local[:remaining_limit])
        
        return results
        
    async def detect_temporal_patterns(self, 
                                    timeline_id: Optional[str] = None, 
                                    window_size: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Detect patterns in temporal events.
        
        Args:
            timeline_id: Optional timeline ID (uses active timeline if None)
            window_size: Optional time window to analyze (defaults to all events)
            
        Returns:
            Dictionary with detected patterns and insights
        """
        # Use active timeline if none specified
        timeline_id = timeline_id or self.active_timeline_id
        
        # Define time window for analysis
        end_time = datetime.now()
        start_time = end_time - (window_size or timedelta(days=30))
        
        # Get timeline events within window
        events = await self.retrieve_timeline_events(
            timeline_id=timeline_id,
            start_time=start_time,
            end_time=end_time,
            limit=200  # Get a substantial sample for pattern detection
        )
        
        if not events:
            return {"error": "No events found in timeline"}
            
        # Analyze event distribution
        event_types = defaultdict(int)
        event_times = []
        event_time_deltas = []
        
        for i, event in enumerate(events):
            # Track event types
            event_type = (
                event.get("event_type") or 
                event.get("content", {}).get("temporal_type", "unknown")
            )
            event_types[event_type] += 1
            
            # Track event timing
            if "timestamp" in event:
                if isinstance(event["timestamp"], str):
                    event_time = datetime.fromisoformat(event["timestamp"])
                else:
                    event_time = event["timestamp"]
            else:
                event_time = datetime.fromisoformat(event.get("content", {}).get("timestamp"))
                
            event_times.append(event_time)
            
            # Calculate time between events
            if i > 0:
                prev_time = event_times[i-1]
                delta = (event_time - prev_time).total_seconds()
                event_time_deltas.append(delta)
        
        # Calculate timing statistics
        avg_time_between_events = (
            sum(event_time_deltas) / len(event_time_deltas)
            if event_time_deltas else 0
        )
        
        # Basic pattern detection
        # This would be expanded with more sophisticated algorithms in production
        pattern_results = {
            "timeline_id": timeline_id,
            "total_events": len(events),
            "event_distribution": dict(event_types),
            "time_range": {
                "start": events[0].get("timestamp") if events else None,
                "end": events[-1].get("timestamp") if events else None
            },
            "timing_metrics": {
                "avg_time_between_events_seconds": avg_time_between_events,
                "min_time_between_events_seconds": min(event_time_deltas) if event_time_deltas else None,
                "max_time_between_events_seconds": max(event_time_deltas) if event_time_deltas else None
            }
        }
        
        # Detect bursts of activity
        if event_time_deltas:
            std_dev = (
                sum((x - avg_time_between_events) ** 2 for x in event_time_deltas) 
                / len(event_time_deltas)
            ) ** 0.5
            
            burst_threshold = avg_time_between_events - (std_dev * 0.5)
            bursts = []
            current_burst = []
            
            for i, delta in enumerate(event_time_deltas):
                if delta < burst_threshold:
                    if not current_burst:  # Start new burst
                        current_burst = [i]
                    current_burst.append(i + 1)
                elif current_burst:  # End current burst
                    if len(current_burst) >= 3:  # Only count as burst if at least 3 events
                        burst_start = events[current_burst[0]].get("timestamp")
                        burst_end = events[current_burst[-1]].get("timestamp")
                        bursts.append({
                            "start_idx": current_burst[0],
                            "end_idx": current_burst[-1],
                            "start_time": burst_start,
                            "end_time": burst_end,
                            "event_count": len(current_burst)
                        })
                    current_burst = []
            
            pattern_results["activity_bursts"] = bursts
            pattern_results["burst_count"] = len(bursts)
        
        return pattern_results
    
    # ============================================================================
    # Specialized Risk Management Methods
    # ============================================================================
    
    async def record_risk_violation(self, 
                                  violation_type: str, 
                                  severity: str,
                                  details: Dict[str, Any],
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record a risk violation event.
        
        Args:
            violation_type: Type of risk violation
            severity: Severity level (e.g., "low", "medium", "high", "critical")
            details: Violation details
            metadata: Optional metadata
            
        Returns:
            str: Violation event ID
        """
        # Generate violation ID
        timestamp = datetime.now()
        violation_id = f"risk_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create violation record
        violation_record = {
            "id": violation_id,
            "type": violation_type,
            "severity": severity,
            "details": details,
            "timestamp": timestamp.isoformat(),
            "resolved": False
        }
        
        # Add to local tracking
        self.risk_violations.append(violation_record)
        
        # Prepare metadata
        meta = metadata or {}
        meta.update({
            "component": "chronos",
            "risk_type": violation_type,
            "severity": severity
        })
        
        # Store in global memory if available
        if self.use_global_memory:
            try:
                # Log in temporal log
                await self.memory_core.temporal_log.log_event(
                    agent_id=self.agent_id,
                    event_type="risk_violation",
                    data=violation_record,
                    metadata=meta
                )
                
                # Publish alert
                if hasattr(self.memory_core, 'message_bus'):
                    await self.memory_core.message_bus.publish(
                        topic=f"{self.agent_id}.risk.violation",
                        message={
                            "event": "risk_violation",
                            "violation_id": violation_id,
                            "type": violation_type,
                            "severity": severity,
                            "timestamp": timestamp.isoformat()
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to record risk violation in global memory: {e}")
        
        return violation_id
    
    async def resolve_risk_violation(self, 
                                   violation_id: str, 
                                   resolution_details: Dict[str, Any]) -> bool:
        """
        Mark a risk violation as resolved.
        
        Args:
            violation_id: ID of the violation to resolve
            resolution_details: Details about the resolution
            
        Returns:
            bool: Success status
        """
        # Find violation in local tracking
        for violation in self.risk_violations:
            if violation["id"] == violation_id:
                violation["resolved"] = True
                violation["resolution"] = resolution_details
                violation["resolved_at"] = datetime.now().isoformat()
                
                # Update in global memory if available
                if self.use_global_memory:
                    try:
                        # Log resolution event
                        await self.memory_core.temporal_log.log_event(
                            agent_id=self.agent_id,
                            event_type="risk_resolution",
                            data={
                                "violation_id": violation_id,
                                "resolution": resolution_details,
                                "resolved_at": violation["resolved_at"]
                            },
                            metadata={
                                "component": "chronos",
                                "risk_type": violation["type"],
                                "severity": violation["severity"]
                            }
                        )
                    except Exception as e:
                        logger.error(f"Failed to record risk resolution in global memory: {e}")
                
                return True
                
        return False
        
    async def record_portfolio_snapshot(self, 
                                      balance: float, 
                                      positions: Dict[str, Any],
                                      metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Record a snapshot of the portfolio.
        
        Args:
            balance: Current balance
            positions: Current positions
            metrics: Optional portfolio metrics
            
        Returns:
            str: Snapshot ID
        """
        # Generate snapshot ID
        timestamp = datetime.now()
        snapshot_id = f"portfolio_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create snapshot record
        snapshot = {
            "id": snapshot_id,
            "balance": balance,
            "positions": positions,
            "metrics": metrics or {},
            "timestamp": timestamp.isoformat()
        }
        
        # Add to local tracking
        self.portfolio_snapshots.append(snapshot)
        
        # Keep only recent snapshots in memory
        if len(self.portfolio_snapshots) > 100:
            self.portfolio_snapshots = self.portfolio_snapshots[-100:]
        
        # Store in global memory if available
        if self.use_global_memory:
            try:
                # Store in episodic memory
                await self.memory_core.episodic_store.store(
                    session_id=self.memory_core.current_session_id,
                    agent_id=self.agent_id,
                    memory_id=snapshot_id,
                    content=snapshot,
                    metadata={
                        "component": "chronos",
                        "object_type": "portfolio_snapshot"
                    }
                )
                
                # Also log in temporal log
                await self.memory_core.temporal_log.log_event(
                    agent_id=self.agent_id,
                    event_type="portfolio_snapshot",
                    data={
                        "snapshot_id": snapshot_id,
                        "balance": balance,
                        "position_count": len(positions),
                        "metrics_summary": {
                            k: v for k, v in (metrics or {}).items()
                            if k in ["total_pnl", "drawdown", "exposure", "sharpe"]
                        }
                    },
                    metadata={
                        "component": "chronos",
                        "balance": balance
                    }
                )
            except Exception as e:
                logger.error(f"Failed to record portfolio snapshot in global memory: {e}")
        
        return snapshot_id
    
    async def get_balance_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get historical balance data.
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            List of balance history points
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Start with local snapshots
        snapshots = [
            s for s in self.portfolio_snapshots
            if datetime.fromisoformat(s["timestamp"]) >= cutoff_date
        ]
        
        # Get additional data from global memory if available
        if self.use_global_memory:
            try:
                # Query events with balance metadata
                events = await self.memory_core.temporal_log.get_events_by_type(
                    event_type="portfolio_snapshot",
                    agent_id=self.agent_id,
                    start_time=cutoff_date,
                    limit=100
                )
                
                # Extract balance data
                for event in events:
                    data = event.get("data", {})
                    if "balance" in data:
                        snapshots.append({
                            "timestamp": event["timestamp"],
                            "balance": data["balance"]
                        })
            except Exception as e:
                logger.error(f"Failed to retrieve balance history from global memory: {e}")
        
        # Sort by timestamp
        sorted_snapshots = sorted(
            snapshots,
            key=lambda x: datetime.fromisoformat(x["timestamp"]) if isinstance(x["timestamp"], str)
            else x["timestamp"]
        )
        
        return sorted_snapshots
    
    async def detect_market_regime(self, 
                                 market_data: Dict[str, Any],
                                 indicators: Optional[Dict[str, Any]] = None) -> str:
        """
        Detect the current market regime based on market data.
        
        Args:
            market_data: Current market data
            indicators: Optional technical indicators
            
        Returns:
            str: Detected market regime
        """
        # This would normally involve sophisticated analysis
        # For now, implement a simple detection logic
        volatility = market_data.get("volatility", 0)
        trend = market_data.get("trend", 0)
        volume = market_data.get("volume", 0)
        
        # Simple regime detection logic
        if volatility > 0.5:
            if trend > 0.3:
                regime = "volatile_bullish"
            elif trend < -0.3:
                regime = "volatile_bearish"
            else:
                regime = "choppy"
        else:
            if trend > 0.3:
                regime = "trending_bullish"
            elif trend < -0.3:
                regime = "trending_bearish"
            else:
                regime = "ranging"
                
        # Store the regime detection
        timestamp = datetime.now()
        self.current_regime = regime
        self.regimes[regime] = {
            "last_detected": timestamp.isoformat(),
            "market_data": market_data,
            "indicators": indicators
        }
        
        # Log regime change in global memory if available
        if self.use_global_memory:
            try:
                await self.memory_core.temporal_log.log_event(
                    agent_id=self.agent_id,
                    event_type="market_regime",
                    data={
                        "regime": regime,
                        "market_data": market_data,
                        "indicators": indicators
                    },
                    metadata={
                        "component": "chronos",
                        "regime_type": regime
                    }
                )
            except Exception as e:
                logger.error(f"Failed to log market regime in global memory: {e}")
        
        return regime