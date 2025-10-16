from typing import List, Dict, Any, Optional, Union
import uuid
import json
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from src.agents.hermes_workspace.models import AgentVote, ConsensusResult, ConflictType

# Import memory system components
try:
    from src.memory.memory_core import MemoryCore
    from src.memory.temporal_log import TemporalEventLog
    from src.memory.message_bus import MessageBus
    MEMORY_SYSTEM_AVAILABLE = True
except ImportError:
    # Fall back to local implementation if global memory system isn't available
    MEMORY_SYSTEM_AVAILABLE = False


class HermesMemory:
    """Centralized memory management for HERMES with integration to global memory system"""
    
    def __init__(self, agent_id: str = "hermes", use_global_memory: bool = True):
        self.agent_id = agent_id
        self.use_global_memory = use_global_memory and MEMORY_SYSTEM_AVAILABLE
        
        # Local memory storage
        self.decision_history = []
        self.agent_performance = defaultdict(lambda: {
            "total_votes": 0,
            "correct_predictions": 0,
            "avg_confidence": 0.0,
            "signals_given": defaultdict(int)
        })
        self.conflict_patterns = defaultdict(int)
        self.trust_evolution = []
        
        # Initialize connection to global memory system if available
        self.memory_core = None
        if self.use_global_memory:
            self._initialize_global_memory()
    
    def _initialize_global_memory(self):
        """Initialize connection to the global memory system"""
        try:
            self.memory_core = MemoryCore()
            self.memory_core.initialize_components()
            
            # Register agent with memory system
            if hasattr(self.memory_core, 'registry'):
                # This would normally be async, but we're calling it synchronously for simplicity
                print(f"Registering agent {self.agent_id} with memory system")
                # In production code, this would be:
                # await self.memory_core.registry.register_agent(self.agent_id, metadata={...})
        except Exception as e:
            print(f"Failed to initialize global memory: {e}")
            self.memory_core = None
            self.use_global_memory = False
    
    async def record_decision_async(self, consensus: ConsensusResult, votes: List[AgentVote], outcome: Optional[bool] = None):
        """
        Asynchronously record a decision and update agent performance
        
        This method integrates with the global memory system if available.
        """
        # Create decision record
        timestamp = datetime.now()
        decision_record = {
            "consensus": consensus,
            "votes": votes,
            "outcome": outcome,
            "timestamp": timestamp.isoformat()
        }
        
        # Store in local memory
        self.decision_history.append(decision_record)
        
        # Update agent performance locally
        for vote in votes:
            self._update_agent_performance(vote, consensus, outcome)
            
        # If global memory is available, store there too
        if self.use_global_memory and self.memory_core:
            try:
                # Prepare data for storage in global memory
                decision_data = {
                    "decision": consensus.decision.value,
                    "confidence": consensus.confidence,
                    "method": consensus.method,
                    "participating_agents": consensus.participating_agents,
                    "dissenting_agents": consensus.dissenting_agents,
                    "vote_breakdown": {agent: (signal.value, conf) for agent, (signal, conf) in consensus.vote_breakdown.items()},
                    "outcome": outcome
                }
                
                # Log decision event in temporal log
                await self.memory_core.temporal_log.log_event(
                    agent_id=self.agent_id,
                    event_type="consensus_decision",
                    data=decision_data,
                    metadata={
                        "decision_type": consensus.decision.value,
                        "confidence": consensus.confidence,
                        "has_dissent": bool(consensus.dissenting_agents)
                    }
                )
                
                # Store agent votes
                for vote in votes:
                    await self.memory_core.temporal_log.log_event(
                        agent_id=vote.agent_name,
                        event_type="agent_vote",
                        data={
                            "signal": vote.signal.value,
                            "confidence": vote.confidence,
                            "reasoning": vote.reasoning
                        },
                        metadata={
                            "consensus_decision": consensus.decision.value,
                            "was_supporting": vote.signal == consensus.decision
                        }
                    )
                    
                # Publish message to message bus
                if hasattr(self.memory_core, 'message_bus'):
                    await self.memory_core.message_bus.publish(
                        topic="hermes.decisions",
                        message={
                            "event": "new_decision",
                            "agent_id": self.agent_id,
                            "decision": consensus.decision.value,
                            "confidence": consensus.confidence,
                            "timestamp": timestamp.isoformat()
                        }
                    )
            except Exception as e:
                print(f"Failed to record decision in global memory: {e}")
    
    def record_decision(self, consensus: ConsensusResult, votes: List[AgentVote], outcome: Optional[bool] = None):
        """
        Record a decision and update agent performance (synchronous version)
        
        This method maintains backward compatibility while supporting the global memory system.
        """
        # Record in local memory immediately
        self.decision_history.append({
            "consensus": consensus,
            "votes": votes,
            "outcome": outcome,
            "timestamp": datetime.now()
        })
        
        # Update agent performance
        for vote in votes:
            self._update_agent_performance(vote, consensus, outcome)
            
        # Schedule async update to global memory if available
        if self.use_global_memory and self.memory_core:
            # Create and schedule a background task for the async operation
            try:
                loop = asyncio.get_event_loop()
                asyncio.create_task(self.record_decision_async(consensus, votes, outcome))
            except RuntimeError:
                # No running event loop
                print("Warning: Could not update global memory (no running event loop)")
    
    def _update_agent_performance(self, vote: AgentVote, consensus: ConsensusResult, outcome: Optional[bool]):
        """Helper method to update agent performance metrics"""
        perf = self.agent_performance[vote.agent_name]
        perf["total_votes"] += 1
        perf["signals_given"][vote.signal.value] += 1
        
        # Update running average of confidence
        total = perf["total_votes"]
        perf["avg_confidence"] = ((perf["avg_confidence"] * (total - 1)) + vote.confidence) / total
        
        # Update correct predictions if outcome is known
        if outcome is not None and vote.signal == consensus.decision:
            perf["correct_predictions"] += outcome
    
    async def record_conflict_pattern_async(self, conflict_type: ConflictType):
        """Asynchronously track conflict patterns over time"""
        # Update local tracking
        self.conflict_patterns[conflict_type.value] += 1
        
        # If global memory is available, log the conflict
        if self.use_global_memory and self.memory_core:
            try:
                await self.memory_core.temporal_log.log_event(
                    agent_id=self.agent_id,
                    event_type="conflict_pattern",
                    data={
                        "conflict_type": conflict_type.value,
                        "count": self.conflict_patterns[conflict_type.value]
                    },
                    metadata={
                        "pattern_type": conflict_type.value
                    }
                )
            except Exception as e:
                print(f"Failed to record conflict pattern in global memory: {e}")
    
    def record_conflict_pattern(self, conflict_type: ConflictType):
        """Track conflict patterns over time (synchronous version)"""
        # Update local tracking
        self.conflict_patterns[conflict_type.value] += 1
        
        # Schedule async update to global memory if available
        if self.use_global_memory and self.memory_core:
            try:
                loop = asyncio.get_event_loop()
                asyncio.create_task(self.record_conflict_pattern_async(conflict_type))
            except RuntimeError:
                # No running event loop
                print("Warning: Could not update global memory (no running event loop)")
    
    async def snapshot_trust_scores_async(self, trust_scores: Dict[str, float]):
        """Asynchronously take a snapshot of current trust scores"""
        timestamp = datetime.now()
        
        # Update local tracking
        self.trust_evolution.append({
            "timestamp": timestamp,
            "scores": trust_scores.copy()
        })
        
        # If global memory is available, log the trust scores
        if self.use_global_memory and self.memory_core:
            try:
                for agent_name, score in trust_scores.items():
                    await self.memory_core.episodic_store.store(
                        session_id=self.memory_core.current_session_id,
                        agent_id=agent_name,
                        memory_id=f"trust_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                        content={
                            "trust_score": score,
                            "timestamp": timestamp.isoformat()
                        },
                        metadata={
                            "memory_type": "trust_score",
                            "recorder_agent": self.agent_id
                        }
                    )
                
                # Log overall trust state
                await self.memory_core.temporal_log.log_event(
                    agent_id=self.agent_id,
                    event_type="trust_snapshot",
                    data={
                        "trust_scores": trust_scores.copy(),
                        "min_score": min(trust_scores.values()) if trust_scores else 0,
                        "max_score": max(trust_scores.values()) if trust_scores else 0,
                        "avg_score": sum(trust_scores.values()) / len(trust_scores) if trust_scores else 0
                    },
                    metadata={
                        "snapshot_time": timestamp.isoformat()
                    }
                )
            except Exception as e:
                print(f"Failed to record trust scores in global memory: {e}")
    
    def snapshot_trust_scores(self, trust_scores: Dict[str, float]):
        """Take a snapshot of current trust scores (synchronous version)"""
        timestamp = datetime.now()
        
        # Update local tracking
        self.trust_evolution.append({
            "timestamp": timestamp,
            "scores": trust_scores.copy()
        })
        
        # Schedule async update to global memory if available
        if self.use_global_memory and self.memory_core:
            try:
                loop = asyncio.get_event_loop()
                asyncio.create_task(self.snapshot_trust_scores_async(trust_scores))
            except RuntimeError:
                # No running event loop
                print("Warning: Could not update global memory (no running event loop)")
    
    async def get_agent_track_record_async(self, agent_name: str) -> Dict[str, Any]:
        """Asynchronously get complete track record for an agent"""
        # Get local track record
        track_record = dict(self.agent_performance[agent_name])
        
        # If global memory is available, enhance with global data
        if self.use_global_memory and self.memory_core:
            try:
                # Get agent's vote history from temporal log
                recent_votes = await self.memory_core.temporal_log.get_events_by_type(
                    event_type="agent_vote",
                    agent_id=agent_name,
                    limit=50
                )
                
                # Calculate global metrics
                if recent_votes:
                    global_signals = defaultdict(int)
                    global_confidence = []
                    
                    for vote in recent_votes:
                        data = vote.get("data", {})
                        global_signals[data.get("signal", "unknown")] += 1
                        if "confidence" in data:
                            global_confidence.append(data["confidence"])
                    
                    # Add global metrics to track record
                    track_record["global_signals"] = dict(global_signals)
                    track_record["global_avg_confidence"] = (
                        sum(global_confidence) / len(global_confidence)
                        if global_confidence else 0
                    )
                    track_record["global_vote_count"] = len(recent_votes)
            except Exception as e:
                print(f"Failed to retrieve global track record: {e}")
        
        return track_record
    
    def get_agent_track_record(self, agent_name: str) -> Dict[str, Any]:
        """Get complete track record for an agent (synchronous version)"""
        # Return local track record immediately
        return dict(self.agent_performance[agent_name])
    
    async def get_recent_consensus_pattern_async(self, limit: int = 20) -> Dict[str, int]:
        """Asynchronously analyze recent decision patterns"""
        patterns = defaultdict(int)
        
        # Get patterns from local history
        local_recent = self.decision_history[-limit:]
        for record in local_recent:
            patterns[record["consensus"].decision.value] += 1
        
        # If global memory is available, get patterns from global history
        if self.use_global_memory and self.memory_core:
            try:
                # Get recent consensus decisions from temporal log
                global_decisions = await self.memory_core.temporal_log.get_events_by_type(
                    event_type="consensus_decision",
                    agent_id=self.agent_id,
                    limit=limit
                )
                
                # Combine with local patterns
                for decision in global_decisions:
                    data = decision.get("data", {})
                    decision_value = data.get("decision", "unknown")
                    patterns[decision_value] += 1
            except Exception as e:
                print(f"Failed to retrieve global consensus patterns: {e}")
        
        return dict(patterns)
    
    def get_recent_consensus_pattern(self, limit: int = 20) -> Dict[str, int]:
        """Analyze recent decision patterns (synchronous version)"""
        # Get patterns from local history
        recent = self.decision_history[-limit:]
        patterns = defaultdict(int)
        for record in recent:
            patterns[record["consensus"].decision.value] += 1
        return dict(patterns)
        
    # ============================================================================
    # Temporal Memory Methods (Chronos Integration)
    # ============================================================================
    
    async def create_timeline(self, name: str, description: str = None) -> str:
        """
        Create a new timeline for tracking sequential events.
        
        Args:
            name: Name of the timeline
            description: Optional description
            
        Returns:
            Timeline ID
        """
        if not self.use_global_memory or not self.memory_core:
            timeline_id = f"timeline_{uuid.uuid4().hex[:8]}"
            print(f"Warning: Created local timeline only: {timeline_id}")
            return timeline_id
            
        try:
            # Create unique timeline ID
            timestamp = datetime.now()
            timeline_id = f"timeline_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
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
            
            return timeline_id
        except Exception as e:
            print(f"Failed to create timeline: {e}")
            return f"local_timeline_{uuid.uuid4().hex[:8]}"
    
    async def record_temporal_event(self, 
                                  timeline_id: str, 
                                  event_type: str, 
                                  content: Dict[str, Any],
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record an event in the temporal memory system.
        
        Args:
            timeline_id: ID of the timeline to add the event to
            event_type: Type of event
            content: Event content/data
            metadata: Optional event metadata
            
        Returns:
            Event ID
        """
        timestamp = datetime.now()
        event_id = f"evt_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Prepare metadata
        meta = metadata or {}
        meta.update({
            "timeline_id": timeline_id,
            "event_type": event_type
        })
        
        if not self.use_global_memory or not self.memory_core:
            print(f"Warning: Could not record temporal event {event_id} to global memory")
            return event_id
        
        try:
            # Log event in temporal log
            await self.memory_core.temporal_log.log_event(
                agent_id=self.agent_id,
                event_type=event_type,
                data=content,
                metadata=meta
            )
            
            # Store event in episodic memory for richer content
            await self.memory_core.episodic_store.store(
                session_id=self.memory_core.current_session_id,
                agent_id=self.agent_id,
                memory_id=event_id,
                content={
                    "event_id": event_id,
                    "timeline_id": timeline_id,
                    "event_type": event_type,
                    "content": content,
                    "timestamp": timestamp.isoformat()
                },
                metadata=meta
            )
            
            return event_id
        except Exception as e:
            print(f"Failed to record temporal event: {e}")
            return event_id
    
    async def get_timeline_events(self, 
                               timeline_id: str, 
                               start_time: Optional[Union[datetime, str]] = None,
                               end_time: Optional[Union[datetime, str]] = None,
                               event_type: Optional[str] = None,
                               limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve events from a timeline.
        
        Args:
            timeline_id: ID of the timeline to retrieve events from
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            event_type: Optional event type to filter by
            limit: Maximum number of events to retrieve
            
        Returns:
            List of timeline events
        """
        if not self.use_global_memory or not self.memory_core:
            print("Warning: Global memory not available, cannot retrieve timeline events")
            return []
            
        try:
            # Convert string times to datetime if needed
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time)
                
            # Query by metadata to get events for this timeline
            events = await self.memory_core.temporal_log.get_events_by_metadata(
                metadata_key="timeline_id",
                metadata_value=timeline_id,
                limit=limit * 2  # Get more than needed for filtering
            )
            
            # Filter by time range if specified
            filtered_events = []
            for event in events:
                event_time = datetime.fromisoformat(event["timestamp"])
                
                # Apply time filters
                if start_time and event_time < start_time:
                    continue
                if end_time and event_time > end_time:
                    continue
                    
                # Apply event type filter
                if event_type and event.get("event_type") != event_type:
                    continue
                    
                filtered_events.append(event)
            
            # Sort by timestamp (oldest to newest)
            sorted_events = sorted(
                filtered_events,
                key=lambda x: datetime.fromisoformat(x["timestamp"])
            )
            
            return sorted_events[:limit]
        except Exception as e:
            print(f"Failed to retrieve timeline events: {e}")
            return []
    
    async def detect_temporal_patterns(self, 
                                    timeline_id: str, 
                                    window_size: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Detect patterns in temporal events.
        
        Args:
            timeline_id: ID of timeline to analyze
            window_size: Optional time window to analyze (defaults to all events)
            
        Returns:
            Dictionary with detected patterns and insights
        """
        if not self.use_global_memory or not self.memory_core:
            print("Warning: Global memory not available, cannot detect temporal patterns")
            return {"error": "Global memory not available"}
            
        try:
            # Define time window for analysis
            end_time = datetime.now()
            start_time = end_time - (window_size or timedelta(days=30))
            
            # Get timeline events within window
            events = await self.get_timeline_events(
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
                event_type = event.get("event_type", "unknown")
                event_types[event_type] += 1
                
                # Track event timing
                event_time = datetime.fromisoformat(event["timestamp"])
                event_times.append(event_time)
                
                # Calculate time between events
                if i > 0:
                    prev_time = datetime.fromisoformat(events[i-1]["timestamp"])
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
                    "start": events[0]["timestamp"] if events else None,
                    "end": events[-1]["timestamp"] if events else None
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
                            burst_start = events[current_burst[0]]["timestamp"]
                            burst_end = events[current_burst[-1]]["timestamp"]
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
        except Exception as e:
            print(f"Failed to detect temporal patterns: {e}")
            return {"error": f"Pattern detection failed: {str(e)}"}
    
    # ============================================================================
    # Integration with other agent memories
    # ============================================================================
    
    async def subscribe_to_agent_events(self, agent_id: str, event_types: List[str] = None):
        """
        Subscribe to events from another agent.
        
        Args:
            agent_id: ID of the agent to subscribe to
            event_types: Optional list of event types to filter by
        """
        if not self.use_global_memory or not self.memory_core:
            print(f"Warning: Cannot subscribe to agent {agent_id} (global memory not available)")
            return
            
        try:
            if not hasattr(self.memory_core, 'message_bus'):
                print("Warning: Message bus not available in memory core")
                return
                
            # Subscribe to agent-specific topics
            topics = [f"{agent_id}.events"]
            
            # Subscribe to specific event types if requested
            if event_types:
                for event_type in event_types:
                    topics.append(f"{agent_id}.events.{event_type}")
            
            # Register subscriptions
            for topic in topics:
                await self.memory_core.message_bus.subscribe(topic, self._handle_agent_event)
                print(f"Subscribed to {topic}")
        except Exception as e:
            print(f"Failed to subscribe to agent events: {e}")
    
    async def _handle_agent_event(self, topic: str, message: Dict[str, Any]):
        """Handle incoming messages from other agents"""
        print(f"Received message on topic {topic}: {json.dumps(message, indent=2)}")
        # Process incoming events based on type
        # This would be expanded in a production system