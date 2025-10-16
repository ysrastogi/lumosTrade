from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import logging

from src.memory.assistant import MemoryAssistant
from src.memory.memory_core import MemoryCore
from src.agents.apollo_workspace.models import Signal

logger = logging.getLogger(__name__)

# Import the global memory core from athena's memory manager
# This ensures all agents use the exact same memory core instance
from src.agents.athena_workspace.memory_manager import get_global_memory_core

class ApolloMemoryManager:
    """
    Memory manager for Apollo to access market data and analysis stored by Athena
    """
    
    def __init__(self, agent_id: str = "apollo_agent", use_redis: bool = True, memory_core: Optional[MemoryCore] = None):
        self.agent_id = agent_id
        self.use_redis = use_redis
        
        # Use a persistent session ID for this agent - each agent gets its own session
        self.persistent_session_id = f"persistent_session_{agent_id}"
        
        # Use provided memory core or get the global one
        if memory_core is not None:
            self.memory_core = memory_core
            logger.info(f"Using provided memory core for agent {agent_id}")
        else:
            # Use global memory core for cross-agent access
            self.memory_core = get_global_memory_core(use_redis=use_redis)
            logger.info(f"Using global memory core for cross-agent access")
        
        # Each agent gets its own memory assistant instance but shares the core
        self.memory_assistant = MemoryAssistant(memory_core=self.memory_core)
        # Set the agent ID on the assistant to ensure proper attribution
        self.memory_assistant.agent_id = agent_id
        self._initialized = False
        
        logger.info(f"Apollo Memory Manager created for agent {agent_id} with persistent session")
        
    async def initialize(self):
        """Initialize the memory system for Apollo agent"""
        if self._initialized:
            logger.debug("Memory manager already initialized")
            return True
        
        agent_metadata = {
            "name": "Apollo Trading Signal Analyzer",
            "role": "Signal explanation and narrative generation",
            "description": "Analyzes trading signals and generates explanations, invalidation criteria, and historical comparisons",
            "version": "1.0.0",
            "capabilities": ["signal_analysis", "narrative_generation", "historical_comparison"]
        }
    
        success = await self.memory_assistant.initialize_for_agent(
            agent_id=self.agent_id,
            agent_metadata=agent_metadata
        )
    
        if success:
            # Subscribe to relevant topics
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
            logger.info(f"Apollo agent {self.agent_id} successfully registered with memory system")
            return True
        
        logger.error("Failed to initialize Apollo memory manager")
        return False
        
    async def find_similar_patterns(self, pattern: str, symbol: str, regime: str = "", direction: str = "", limit: int = 10) -> List[Signal]:
        """
        Find similar patterns from memory based on pattern type, symbol, and market regime.
        This method looks for observations with similar characteristics in the memory system.
        
        Args:
            pattern: The pattern type to search for
            symbol: The market symbol
            regime: Optional market regime
            direction: Optional trade direction ('buy' or 'sell')
            limit: Maximum number of signals to return
            
        Returns:
            List of Signal objects matching the criteria
        """
        if not self._initialized:
            await self.initialize()
            
        # Get context from memory based on tags
        query = f"pattern:{pattern} symbol:{symbol}"
        if regime:
            query += f" regime:{regime}"
        if direction:
            query += f" direction:{direction}"
            
        # Search for relevant market analyses
        memories = await self.memory_assistant.search(
            query=query,
            memory_types=["market_analysis", "market_observation"],
            limit=limit
        )
        
        # Convert memories to Signal objects
        signals = []
        for memory in memories:
            try:
                if "results" in memory["data"]:
                    # This is a market_analysis memory
                    data = memory["data"]["results"]
                    
                    # Skip if it doesn't have the necessary data
                    if not all(key in data for key in ["pattern", "direction", "entry"]):
                        continue
                        
                    # Create a Signal from the memory data
                    signal = Signal(
                        id=memory["id"][:8],
                        timestamp=datetime.fromisoformat(memory["data"]["timestamp"].replace('Z', '+00:00')),
                        symbol=memory["data"]["symbol"],
                        pattern=data["pattern"],
                        direction=data["direction"],
                        confidence=data.get("confidence", 0.5) * 100,
                        entry=data.get("entry", 0.0),
                        time_horizon=data.get("time_horizon", "unknown"),
                        description=data.get("description", ""),
                        stop_loss=data.get("stop_loss", 0.0),
                        target=data.get("target", 0.0),
                        risk_reward=data.get("risk_reward", 1.0),
                        reasoning="",
                        invalidation_criteria=[],
                        supporting_factors=data.get("supporting_factors", []),
                        similar_historical_count=0,
                        historical_win_rate=0.5,
                        outcome=data.get("outcome", None),
                        pnl=data.get("pnl", None)
                    )
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error converting memory to signal: {str(e)}")
                continue
                
        return signals
    
    async def store_signal_analysis(self, signal: Signal, analysis_data: Dict[str, Any]) -> str:
        """
        Store analysis of a trading signal in memory.
        
        Args:
            signal: The signal being analyzed
            analysis_data: The analysis results
            
        Returns:
            str: Memory ID
        """
        if not self._initialized:
            await self.initialize()
            
        content = {
            "symbol": signal.symbol,
            "timestamp": datetime.now().isoformat(),
            "signal_id": signal.id,
            "pattern": signal.pattern,
            "direction": signal.direction,
            "analysis": analysis_data
        }
        
        memory_id = await self.memory_assistant.remember(
            content=content,
            memory_type="signal_analysis",
            tags=[signal.symbol, signal.pattern, signal.direction]
        )
        
        # Publish to message bus for other agents to use
        await self.memory_core.publish_message(
            sender_id=self.agent_id,
            topic="trading_signals",
            content=content
        )
        
        # Store in global memory for cross-agent access
        await self.store_to_global_memory(
            memory_type="signal_analysis",
            content=content,
            tags=[signal.symbol, signal.pattern, signal.direction]
        )
        
        return memory_id
        
    async def store_to_global_memory(self, memory_type: str, content: Dict[str, Any], 
                                    tags: List[str] = None) -> str:
        """
        Store any type of data in global memory context for cross-agent access.
        
        Args:
            memory_type: Type of memory (e.g. signal_analysis, market_insight, narrative)
            content: Content to store
            tags: Optional tags for memory retrieval
            
        Returns:
            str: Memory ID
        """
        if not self._initialized:
            await self.initialize()
            
        # Ensure content has timestamp and agent attribution
        if "timestamp" not in content:
            content["timestamp"] = datetime.now().isoformat()
            
        if "agent_id" not in content:
            content["agent_id"] = self.agent_id
            
        # Use the global memory core directly
        memory_id = await self.memory_core.store_memory(
            agent_id=self.agent_id,
            memory_type=memory_type,
            content=content,
            tags=tags or []
        )
        
        # Publish to message bus for immediate notification to other agents
        await self.memory_core.publish_message(
            sender_id=self.agent_id,
            topic=f"global_memory_{memory_type}",
            content=content
        )
        
        logger.info(f"Stored {memory_type} in global memory with ID {memory_id}")
        
        return memory_id
        
    async def get_market_context(self, symbol: str, agent_ids=None) -> Dict[str, Any]:
        """
        Get the latest market context for a symbol from across all agents
        
        Args:
            symbol: Market symbol to get context for
            agent_ids: Optional list of specific agent IDs to retrieve observations from
            
        Returns:
            Latest market observation for the symbol with patterns and analysis data
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # First, try to get recent observations from any agent (especially Athena)
            all_observations = await self.recall_cross_agent_observations(symbol=symbol, agent_ids=agent_ids)
            
            # Hold the observation and analysis data
            observation_data = {}
            patterns_data = []
            regime_data = {}
            
            # Check if Athena has observations for this symbol
            athena_agent_id = None
            for agent_id, observations in all_observations.items():
                if observations and len(observations) > 0:
                    # Sort by timestamp if available to get the most recent
                    observations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                    logger.info(f"Found market data for {symbol} from agent {agent_id}")
                    observation_data = observations[0]
                    athena_agent_id = agent_id
                    break
            
            # If we found Athena data, also look for pattern and regime data
            if athena_agent_id:
                try:
                    # For now, just make sure the observation_data has all needed fields
                    # Add regime if missing but available in data
                    if 'regime' not in observation_data and 'summary' in observation_data:
                        summary = observation_data.get('summary', '').lower()
                        if 'sideways' in summary:
                            observation_data['regime'] = 'sideways'
                        elif 'bullish' in summary or 'uptrend' in summary:
                            observation_data['regime'] = 'uptrend'
                        elif 'bearish' in summary or 'downtrend' in summary:
                            observation_data['regime'] = 'downtrend'
                    
                    # Add regime confidence if missing
                    if 'regime_confidence' not in observation_data:
                        observation_data['regime_confidence'] = 0.75
                    
                    # Extract patterns from summary if not present
                    if 'patterns' not in observation_data and 'summary' in observation_data:
                        summary = observation_data.get('summary', '')
                        patterns_data = []
                        
                        # Look for mean reversion pattern
                        if 'mean reversion' in summary.lower():
                            # Try to extract confidence
                            import re
                            confidence_match = re.search(r'(\d+)%\s*confidence', summary.lower())
                            confidence = float(confidence_match.group(1)) if confidence_match else 75.0
                            
                            # Determine bias
                            bias = 'neutral'
                            if 'bullish' in summary.lower():
                                bias = 'bullish'
                            elif 'bearish' in summary.lower():
                                bias = 'bearish'
                                
                            pattern = {
                                "type": "mean_reversion",
                                "confidence": confidence,
                                "description": "Mean reversion pattern extracted from summary",
                                "bias": bias
                            }
                            patterns_data.append(pattern)
                            logger.info(f"Extracted mean reversion pattern from summary with {confidence}% confidence")
                        
                        # Add to observation data
                        if patterns_data:
                            observation_data['patterns'] = patterns_data
                            
                except Exception as e:
                    logger.warning(f"Error extracting pattern data from summary: {e}")
            
            # If we found observation data, merge with patterns and regime
            if observation_data:
                # Add patterns and regime to the market data
                if patterns_data:
                    observation_data["patterns"] = patterns_data
                if regime_data:
                    if "regime" in regime_data:
                        observation_data["regime"] = regime_data.get("regime")
                    if "confidence" in regime_data:
                        observation_data["regime_confidence"] = regime_data.get("confidence")
                
                return observation_data
            
            # If no Athena data, try our own observations
            memories = await self.memory_assistant.recall_recent(
                limit=20,  # Get more to increase chances of finding the right symbol
                memory_types=["market_observation"]
            )
            
            # Manually filter for the right symbol
            filtered_memories = []
            for memory in memories:
                if memory.get("data", {}).get("symbol") == symbol:
                    filtered_memories.append(memory)
            
            # Sort by timestamp if available to get the most recent
            if filtered_memories:
                filtered_memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                return filtered_memories[0]["data"]
            
            logger.warning(f"No market data found for {symbol} in any agent's memory")
            return {}
            
        except Exception as e:
            logger.error(f"Error retrieving market context for {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"Memory retrieval error: {str(e)}"}
        
    async def get_athena_context(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the full memory context from Athena
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            Dict with full memory context
        """
        # Since both agents use the same memory system, we can directly query for context
        context = await self.memory_assistant.get_context(window_size=20)
        
        if symbol and "recent_memories" in context:
            context["recent_memories"] = [
                m for m in context["recent_memories"] 
                if m.get("data", {}).get("symbol") == symbol
            ]
                
        return context
        
    async def recall_cross_agent_observations(self, symbol: Optional[str] = None, 
                                             agent_ids: Optional[List[str]] = None, 
                                             limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Recall observations from multiple agents across the system.
        
        Args:
            symbol: Optional symbol to filter by
            agent_ids: List of agent IDs to query (if None, queries all agents)
            limit: Maximum number of observations per agent
            
        Returns:
            Dict mapping agent_id to list of observations
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            logger.debug(f"Recalling cross-agent observations for symbol: {symbol}, agents: {agent_ids}")
            
            # Determine which agents to query - if no specific agents, try common agent patterns
            if agent_ids:
                target_agents = agent_ids
            else:
                # Try common agent naming patterns + discover from current system
                base_agents = ['athena_agent', 'apollo_agent', 'chronos_agent', 'hermes_agent']
                # Add variations for different instances
                extended_agents = base_agents.copy()
                for base in ['athena', 'apollo', 'chronos', 'hermes']:
                    extended_agents.extend([
                        f'{base}_primary', f'{base}_secondary', f'{base}_agent_1', 
                        f'{base}_agent_2', f'{base}_main', f'{base}_backup'
                    ])
                target_agents = extended_agents
            
            # Query each agent's memories directly through the episodic store
            agent_observations = {}
            for target_agent in target_agents:
                try:
                    # Get memories by type for specific agent
                    agent_memories = await self.memory_core.episodic_store.get_by_type(
                        target_agent, 'market_observation'
                    )
                    
                    filtered_memories = []
                    for memory in agent_memories:
                        memory_data = memory.get("data", {})
                        
                        # Filter by symbol if specified  
                        if symbol and memory_data.get("symbol") != symbol:
                            continue
                            
                        filtered_memories.append(memory_data)
                    
                    if filtered_memories:
                        agent_observations[target_agent] = filtered_memories[:limit]
                        
                except Exception as e:
                    logger.debug(f"Could not retrieve observations for {target_agent}: {e}")
                    continue
            
            logger.debug(f"Cross-agent observations: {len(agent_observations)} agents, "
                        f"{sum(len(observations) for observations in agent_observations.values())} total observations")
            return agent_observations
            
        except Exception as e:
            logger.error(f"Error recalling cross-agent observations: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
            
    async def recall_cross_agent_insights(self, symbol: Optional[str] = None, 
                                         agent_ids: Optional[List[str]] = None, 
                                         limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Recall insights from multiple agents across the system.
        
        Args:
            symbol: Optional symbol to filter by
            agent_ids: List of agent IDs to query (if None, queries all agents)
            limit: Maximum number of insights per agent
            
        Returns:
            Dict mapping agent_id to list of insights
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            logger.debug(f"Recalling cross-agent insights for symbol: {symbol}, agents: {agent_ids}")
            
            # Determine which agents to query - if no specific agents, try common agent patterns
            if agent_ids:
                target_agents = agent_ids
            else:
                # Try common agent naming patterns + discover from current system
                base_agents = ['athena_agent', 'apollo_agent', 'chronos_agent', 'hermes_agent']
                # Add variations for different instances
                extended_agents = base_agents.copy()
                for base in ['athena', 'apollo', 'chronos', 'hermes']:
                    extended_agents.extend([
                        f'{base}_primary', f'{base}_secondary', f'{base}_agent_1', 
                        f'{base}_agent_2', f'{base}_main', f'{base}_backup'
                    ])
                target_agents = extended_agents
            
            # Query each agent's insights directly through the episodic store  
            agent_insights = {}
            for target_agent in target_agents:
                try:
                    # Get insights by type for specific agent
                    agent_memories = await self.memory_core.episodic_store.get_by_type(
                        target_agent, 'market_insight'
                    )
                    
                    filtered_memories = []
                    for memory in agent_memories:
                        memory_data = memory.get("data", {})
                        
                        # Filter by symbol if specified
                        if symbol and memory_data.get("symbol") != symbol:
                            continue
                            
                        filtered_memories.append(memory_data)
                    
                    if filtered_memories:
                        agent_insights[target_agent] = filtered_memories[:limit]
                        
                except Exception as e:
                    logger.debug(f"Could not retrieve insights for {target_agent}: {e}")
                    continue
            
            logger.debug(f"Cross-agent insights: {len(agent_insights)} agents, "
                        f"{sum(len(insights) for insights in agent_insights.values())} total insights")
            return agent_insights
            
        except Exception as e:
            logger.error(f"Error recalling cross-agent insights: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    async def get_agent_memory_summary(self, agent_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of another agent's memory for a specific symbol.
        
        Args:
            agent_id: ID of the agent to query
            symbol: Optional symbol to filter by
            
        Returns:
            Dict with agent memory summary
        """
        try:
            # Get observations for the agent
            agent_memories = await self.memory_core.episodic_store.get_by_type(
                agent_id, 'market_observation'
            )
            
            filtered_observations = []
            for memory in agent_memories:
                memory_data = memory.get("data", {})
                if symbol and memory_data.get("symbol") != symbol:
                    continue
                filtered_observations.append(memory_data)
            
            # Get insights for the agent
            agent_insights = await self.memory_core.episodic_store.get_by_type(
                agent_id, 'market_insight'
            )
            
            filtered_insights = []
            for memory in agent_insights:
                memory_data = memory.get("data", {})
                if symbol and memory_data.get("symbol") != symbol:
                    continue
                filtered_insights.append(memory_data)
                
            # Get analyses for the agent
            agent_analyses = await self.memory_core.episodic_store.get_by_type(
                agent_id, 'market_analysis'
            )
            
            filtered_analyses = []
            for memory in agent_analyses:
                memory_data = memory.get("data", {})
                if symbol and memory_data.get("symbol") != symbol:
                    continue
                filtered_analyses.append(memory_data)
                
            return {
                "agent_id": agent_id,
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "observations_count": len(filtered_observations),
                "observations": filtered_observations,
                "insights_count": len(filtered_insights),
                "insights": filtered_insights,
                "analyses_count": len(filtered_analyses),
                "analyses": filtered_analyses
            }
            
        except Exception as e:
            logger.error(f"Error getting agent memory summary for {agent_id}: {str(e)}")
            return {"error": f"Failed to get memory summary: {str(e)}"}
    
    async def get_context(self, symbol: Optional[str] = None, agent_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get comprehensive memory context including observations, insights, and other data.
        
        Args:
            symbol: Optional symbol to filter memories by
            agent_ids: Optional list of specific agent IDs to query
            
        Returns:
            Dictionary with memory context
        """
        # For compatibility with AthenaMemoryManager, delegate to global_memory_context
        return await self.get_global_memory_context(symbol=symbol, agent_ids=agent_ids)
    
    async def get_global_memory_context(self, symbol: Optional[str] = None, 
                              agent_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get global memory context across all agents for enhanced intelligence.
        
        Args:
            symbol: Optional symbol to filter by
            agent_ids: Optional list of specific agent IDs to query
            
        Returns:
            Dict with global memory context including all agent perspectives
        """
        try:
            # Get cross-agent observations and insights
            all_observations = await self.recall_cross_agent_observations(
                symbol=symbol, 
                agent_ids=agent_ids,
                limit=10
            )
            all_insights = await self.recall_cross_agent_insights(
                symbol=symbol, 
                agent_ids=agent_ids,
                limit=15
            )
            
            # Calculate statistics
            total_observations = sum(len(obs) for obs in all_observations.values())
            total_insights = sum(len(ins) for ins in all_insights.values())
            active_agents = set(all_observations.keys()) | set(all_insights.keys())
            
            # Get recent activity timestamps
            all_timestamps = []
            for obs_list in all_observations.values():
                all_timestamps.extend([obs.get("timestamp", "") for obs in obs_list])
            for ins_list in all_insights.values():
                all_timestamps.extend([ins.get("timestamp", "") for ins in ins_list])
            
            recent_activity = max(all_timestamps) if all_timestamps else None
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "global_stats": {
                    "active_agents": len(active_agents),
                    "total_observations": total_observations,
                    "total_insights": total_insights,
                    "most_recent_activity": recent_activity
                },
                "agent_observations": all_observations,
                "agent_insights": all_insights,
                "agent_list": list(active_agents)
            }
            
        except Exception as e:
            logger.error(f"Error getting global memory context: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"Global memory context error: {str(e)}"}