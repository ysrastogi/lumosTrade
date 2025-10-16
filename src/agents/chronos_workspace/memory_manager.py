import json
import datetime
import numpy as np
import logging
import asyncio
from collections import deque
from typing import Dict, List, Optional, Any, Union

from src.memory.memory_core import MemoryCore

# Use the global memory core from existing system
try:
    from src.agents.athena_workspace.memory_manager import get_global_memory_core
except ImportError:
    # Fallback if the import fails
    _global_memory_core = None
    
    def get_global_memory_core(use_redis: bool = True) -> MemoryCore:
        """
        Get or create the global memory core shared across all agents.
        This enables cross-agent memory access while maintaining session isolation.
        """
        global _global_memory_core
        if _global_memory_core is None:
            _global_memory_core = MemoryCore(use_redis=use_redis)
            _global_memory_core.initialize_components()
        return _global_memory_core

logger = logging.getLogger(__name__)

class ChronosMemory:
    """
    Memory system for CHRONOS risk management agent.
    Stores historical risk data, behavioral patterns, market regimes, and violations.
    Integrates with the LumosTrade memory system for cross-agent communication.
    """
    
    def __init__(self, user_id, max_history=365, use_redis=True):
        """
        Initialize memory systems for the CHRONOS agent.
        
        Parameters:
        -----------
        user_id : str
            Identifier for the trader
        max_history : int
            Maximum days of history to maintain
        use_redis : bool
            Whether to use Redis for global memory persistence
        """
        self.user_id = user_id
        self.max_history = max_history
        self.use_redis = use_redis
        
        # Initialize local memory components
        self.risk_history = deque(maxlen=max_history)
        self.behavioral_patterns = {
            "win_loss_behavior": [],
            "drawdown_responses": [],
            "position_sizing_trends": [],
            "detected_biases": {}
        }
        self.market_regime_memory = {
            "regimes": [],
            "transitions": [],
            "performance_by_regime": {}
        }
        self.violation_log = []
        
        # Initialize integration with global memory system
        self.agent_id = f"chronos_agent_{user_id}"
        self.memory_core = get_global_memory_core(use_redis=use_redis)
        self._initialized = False
        
        self._load_memory()
    
    async def initialize(self):
        """
        Initialize the memory integration by registering with the memory system.
        Must be called before using other memory integration features.
        
        Returns:
            bool: Success status
        """
        if self._initialized:
            return True
            
        agent_metadata = {
            "name": "Chronos Risk Manager",
            "role": "Portfolio risk management and behavioral analysis",
            "description": "Monitors trading risk factors, behavioral patterns, and prevents rule violations",
            "version": "1.0.0",
            "capabilities": ["risk_assessment", "behavioral_analysis", "regime_detection", "violation_prevention"],
            "interests": ["risk_assessment", "market_data", "trading_signals", "portfolio_updates"]
        }
        
        try:
            # Register agent directly with memory core
            success = await self.memory_core.register_agent(
                agent_id=self.agent_id,
                role="Portfolio risk management and behavioral analysis",
                output_schema=["text", "json"]
            )
            
            if success:
                # Subscribe to relevant topics
                await self.memory_core.registry.subscribe_agent_to_topic(
                    agent_id=self.agent_id,
                    topic="risk_assessment"
                )
                await self.memory_core.registry.subscribe_agent_to_topic(
                    agent_id=self.agent_id,
                    topic="market_data"
                )
                await self.memory_core.registry.subscribe_agent_to_topic(
                    agent_id=self.agent_id,
                    topic="trading_signals"
                )
                await self.memory_core.registry.subscribe_agent_to_topic(
                    agent_id=self.agent_id,
                    topic="portfolio_updates"
                )
                
                self._initialized = True
                logger.info(f"Chronos agent {self.agent_id} successfully registered with memory system")
                
                # Load memory from global system
                await self._load_memory_from_system()
                
                return True
                
            logger.error("Failed to initialize Chronos memory integration")
            return False
            
        except Exception as e:
            logger.error(f"Exception during memory integration initialization: {e}")
            return False
        
    async def update_risk_history(self, position_risk, drawdown_risk, var_metrics):
        """
        Update risk history with latest risk assessments.
        
        Parameters:
        -----------
        position_risk : dict
            Latest position sizing metrics
        drawdown_risk : dict
            Latest drawdown predictions
        var_metrics : dict
            Latest Value at Risk metrics
        """
        # Ensure we're initialized
        if not self._initialized:
            await self.initialize()
            
        timestamp = datetime.datetime.now().isoformat()
        
        entry = {
            "timestamp": timestamp,
            "position_risk": position_risk,
            "drawdown_risk": drawdown_risk,
            "var_metrics": var_metrics
        }
        
        # Update local memory
        self.risk_history.append(entry)
        
        # Store in global memory system
        try:
            # Store the detailed risk assessment
            memory_id = await self.memory_core.store_memory(
                agent_id=self.agent_id,
                content=entry,
                memory_type="risk_assessment",
                tags=["risk_history", "var", "drawdown", "position_sizing"]
            )
            
            # Also publish summary to risk_assessment topic for other agents
            await self.memory_core.publish_message(
                sender_id=self.agent_id,
                topic="risk_assessment",
                content={
                    "timestamp": timestamp,
                    "user_id": self.user_id,
                    "risk_summary": {
                        "var_daily": var_metrics.get("daily_var_pct", 0),
                        "max_drawdown": drawdown_risk.get("expected_drawdown", 0),
                        "position_risk": position_risk.get("largest_position_pct", 0)
                    }
                }
            )
            
            logger.debug(f"Risk assessment stored with ID: {memory_id}")
            
        except Exception as e:
            logger.error(f"Error storing risk assessment in memory system: {e}")
            
        # Save local memory backup
        await self._save_memory()
    
    async def update_behavioral_pattern(self, trade_data, emotional_state=None):
        """
        Update behavioral pattern memory with new trade data.
        
        Parameters:
        -----------
        trade_data : dict
            Data about the completed trade
        emotional_state : str
            Optional emotional state reported by trader
        """
        if not self._initialized:
            await self.initialize()
            
        is_win = trade_data.get("profit", 0) > 0
        hold_time = trade_data.get("hold_time_hours", 0)
        
        win_loss_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "is_win": is_win,
            "hold_time": hold_time,
            "position_size": trade_data.get("position_size", 0),
            "emotional_state": emotional_state
        }
        
        self.behavioral_patterns["win_loss_behavior"].append(win_loss_entry)
        
        position_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "position_size": trade_data.get("position_size", 0),
            "previous_outcome": trade_data.get("previous_outcome", "unknown")
        }
        
        # Update local memory
        self.behavioral_patterns["position_sizing_trends"].append(position_entry)
        
        # Ensure we don't exceed memory limits
        if len(self.behavioral_patterns["win_loss_behavior"]) > self.max_history:
            self.behavioral_patterns["win_loss_behavior"] = self.behavioral_patterns["win_loss_behavior"][-self.max_history:]
        
        if len(self.behavioral_patterns["position_sizing_trends"]) > self.max_history:
            self.behavioral_patterns["position_sizing_trends"] = self.behavioral_patterns["position_sizing_trends"][-self.max_history:]
        
        # Store in global memory system
        try:
            # Store win/loss behavior
            await self.memory_core.store_memory(agent_id=self.agent_id,
                content=win_loss_entry,
                memory_type="behavioral_pattern",
                tags=["win_loss_behavior", "trade_behavior"]
            )
            
            # Store position sizing trend
            await self.memory_core.store_memory(agent_id=self.agent_id,
                content=position_entry,
                memory_type="behavioral_pattern",
                tags=["position_sizing_trends", "trade_behavior"]
            )
            
            # If emotional state is provided, record it separately with more detailed tagging
            if emotional_state:
                await self.memory_core.store_memory(agent_id=self.agent_id,
                    content={
                        "timestamp": datetime.datetime.now().isoformat(),
                        "user_id": self.user_id,
                        "emotional_state": emotional_state,
                        "trade_outcome": "win" if is_win else "loss",
                        "trade_id": trade_data.get("id", None)
                    },
                    memory_type="emotional_state",
                    tags=["trader_psychology", emotional_state, "trade_behavior"]
                )
            
            logger.debug(f"Behavioral pattern updated for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error storing behavioral pattern in memory system: {e}")
        
        # Save local memory backup
        await self._save_memory()
    
    async def update_regime_memory(self, regime, volatility_metrics):
        """
        Update market regime memory.
        
        Parameters:
        -----------
        regime : str
            Current market regime classification
        volatility_metrics : dict
            Metrics about current market volatility
        """
        # Ensure we're initialized
        if not self._initialized:
            await self.initialize()
            
        timestamp = datetime.datetime.now().isoformat()
        
        # Create regime entry
        regime_entry = {
            "timestamp": timestamp,
            "regime": regime,
            "volatility": volatility_metrics.get("volatility", 0),
            "trend_strength": volatility_metrics.get("trend_strength", 0)
        }
        
        # Add to local memory
        self.market_regime_memory["regimes"].append(regime_entry)
        
        # Check if this is a regime transition
        transition_entry = None
        if len(self.market_regime_memory["regimes"]) > 1:
            previous_regime = self.market_regime_memory["regimes"][-2]["regime"]
            if previous_regime != regime:
                transition_entry = {
                    "timestamp": timestamp,
                    "from_regime": previous_regime,
                    "to_regime": regime,
                    "volatility_change": volatility_metrics.get("volatility_change", 0)
                }
                self.market_regime_memory["transitions"].append(transition_entry)
        
        # Ensure we don't exceed memory limits
        if len(self.market_regime_memory["regimes"]) > self.max_history:
            self.market_regime_memory["regimes"] = self.market_regime_memory["regimes"][-self.max_history:]
        
        if len(self.market_regime_memory["transitions"]) > self.max_history//2:
            self.market_regime_memory["transitions"] = self.market_regime_memory["transitions"][-(self.max_history//2):]
        
        try:
            # Store regime observation
            await self.memory_core.store_memory(agent_id=self.agent_id,
                content=regime_entry,
                memory_type="market_regime",
                tags=["market_regime", "volatility", regime]
            )
            
            # If this is a transition, store that too and notify other agents
            if transition_entry:
                await self.memory_core.store_memory(agent_id=self.agent_id,
                    content=transition_entry,
                    memory_type="regime_transition",
                    tags=["market_regime", "transition", previous_regime, regime]
                )
                
                # Publish transition message for other agents
                await self.memory_core.publish_message(
                    sender_id=self.agent_id,
                    topic="market_data",
                    content={
                        "timestamp": timestamp,
                        "message_type": "regime_transition",
                        "from_regime": previous_regime,
                        "to_regime": regime,
                        "volatility_change": volatility_metrics.get("volatility_change", 0)
                    }
                )
            
            logger.debug(f"Market regime memory updated for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error storing market regime in memory system: {e}")
        
        # Save local memory backup
        await self._save_memory()
    
    async def log_violation(self, portfolio, market_data, trade=None):
        """
        Log a risk rule violation.
        
        Parameters:
        -----------
        portfolio : dict
            Current portfolio state
        market_data : dict
            Current market conditions
        trade : dict
            Trade that caused the violation (if applicable)
        """
        # Ensure we're initialized
        if not self._initialized:
            await self.initialize()
            
        violation_type = self._determine_violation_type(portfolio, trade)
        
        violation_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "portfolio_state": {
                "total_value": sum(p.get("value", 0) for p in portfolio),
                "largest_position": max(p.get("allocation", 0) for p in portfolio),
                "num_positions": len(portfolio)
            },
            "market_state": {
                "vix": market_data.get("vix", 0),
                "regime": market_data.get("regime", "unknown")
            },
            "violation_type": violation_type,
            "trade": trade
        }
        
        # Add to local memory
        self.violation_log.append(violation_entry)
        
        # Store in global memory system
        try:
            # Store the violation
            memory_id = await self.memory_core.store_memory(
                agent_id=self.agent_id,
                content=violation_entry,
                memory_type="risk_violation",
                tags=["violation", violation_type]
            )
            
            # Send alert message to other agents
            await self.memory_core.publish_message(
                sender_id=self.agent_id,
                topic="risk_assessment",
                content={
                    "message_type": "violation_alert",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "user_id": self.user_id,
                    "violation_type": violation_type,
                    "portfolio_risk": violation_entry["portfolio_state"],
                    "market_state": violation_entry["market_state"],
                    "alert_level": "high"
                }
            )
            
            logger.debug(f"Risk violation logged with ID: {memory_id}")
            
        except Exception as e:
            logger.error(f"Error storing violation in memory system: {e}")
        
        # Save local memory backup
        await self._save_memory()
    
    async def get_historical_comparison(self, lookback_days=30):
        """
        Get comparison between current risk metrics and historical averages.
        
        Parameters:
        -----------
        lookback_days : int
            Number of days to look back for comparison
        
        Returns:
        --------
        dict: Comparison metrics between current and historical risk
        """
        # Ensure we're initialized
        if not self._initialized:
            await self.initialize()
            
        if not self.risk_history:
            # Try to load from memory system first
            await self._load_memory_from_system()
            
            if not self.risk_history:
                return {"status": "insufficient_data"}
        
        # Get current risk metrics (most recent)
        current = self.risk_history[-1]
        
        # Try to get more historical data if needed
        if len(self.risk_history) < lookback_days:
            try:
                # Query memory system for more risk history
                memories = await self.memory_core.episodic_store.get_recent(
                    agent_id=self.agent_id,
                    limit=lookback_days,
                    memory_types=["risk_assessment"]
                )
                
                # Add to local cache if not already there
                for memory in memories:
                    if memory["content"] not in self.risk_history:
                        self.risk_history.append(memory["content"])
            except Exception as e:
                logger.error(f"Error loading additional risk history: {e}")
        
        # Calculate historical averages
        historical = []
        for entry in list(self.risk_history)[-lookback_days:-1]:
            historical.append(entry)
        
        if not historical:
            return {
                "current": current,
                "historical_comparison": "insufficient_data"
            }
        
        # Calculate average historical metrics
        avg_var = np.mean([h["var_metrics"]["daily_var_pct"] for h in historical])
        avg_drawdown = np.mean([h["drawdown_risk"]["expected_drawdown"] for h in historical])
        
        return {
            "current": {
                "var": current["var_metrics"]["daily_var_pct"],
                "drawdown": current["drawdown_risk"]["expected_drawdown"]
            },
            "historical_avg": {
                "var": avg_var,
                "drawdown": avg_drawdown
            },
            "percent_change": {
                "var": ((current["var_metrics"]["daily_var_pct"] / avg_var) - 1) * 100 if avg_var > 0 else 0,
                "drawdown": ((current["drawdown_risk"]["expected_drawdown"] / avg_drawdown) - 1) * 100 if avg_drawdown > 0 else 0
            }
        }
        
    async def recall_risk_history(self, days=30, limit=100):
        """
        Retrieve risk history from the memory system.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of records to return
            
        Returns:
            List of risk history records
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Query memory system for risk assessments
            memories = await self.memory_core.episodic_store.get_recent(
                agent_id=self.agent_id,
                limit=limit,
                memory_types=["risk_assessment"]
            )
            
            return [memory["content"] for memory in memories]
            
        except Exception as e:
            logger.error(f"Error recalling risk history: {e}")
            # Fall back to local cache
            return list(self.risk_history)
            
    async def recall_violations(self, limit=20):
        """
        Retrieve violation history from the memory system.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of violation records
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Query memory system for violations
            memories = await self.memory_core.episodic_store.get_recent(
                agent_id=self.agent_id,
                limit=limit,
                memory_types=["risk_violation"]
            )
            
            return [memory["content"] for memory in memories]
            
        except Exception as e:
            logger.error(f"Error recalling violations: {e}")
            # Fall back to local cache
            return self.violation_log
    
    async def get_behavioral_patterns(self):
        """
        Get analysis of behavioral patterns.
        
        Returns:
        --------
        dict: Analysis of behavioral patterns
        """
        # Ensure we're initialized
        if not self._initialized:
            await self.initialize()
            
        if not self.behavioral_patterns["win_loss_behavior"]:
            # Try to load from memory system first
            await self._load_memory_from_system()
            
            if not self.behavioral_patterns["win_loss_behavior"]:
                return {"status": "insufficient_data"}
        
        # Analyze win-loss behavior
        wins = [t for t in self.behavioral_patterns["win_loss_behavior"] if t["is_win"]]
        losses = [t for t in self.behavioral_patterns["win_loss_behavior"] if not t["is_win"]]
        
        avg_win_hold_time = np.mean([w["hold_time"] for w in wins]) if wins else 0
        avg_loss_hold_time = np.mean([l["hold_time"] for l in losses]) if losses else 0
        
        # Analyze position sizing patterns
        recent_sizes = [t["position_size"] for t in self.behavioral_patterns["position_sizing_trends"][-10:]]
        trend = "increasing" if len(recent_sizes) > 1 and recent_sizes[-1] > recent_sizes[0] else "stable"
        
        # Get detected biases from memory system if possible
        try:
            bias_memories = await self.memory_core.episodic_store.get_recent(
                agent_id=self.agent_id,
                limit=10,
                memory_types=["behavioral_pattern"]
            )
            
            # Update detected biases from memory if we found any
            if bias_memories:
                detected_biases = {}
                for memory in bias_memories:
                    if "bias_type" in memory["content"]:
                        bias_type = memory["content"]["bias_type"]
                        detected_biases[bias_type] = memory["content"].get("description", "")
                
                if detected_biases:
                    self.behavioral_patterns["detected_biases"] = detected_biases
        except Exception as e:
            logger.error(f"Error loading bias data from memory: {e}")
        
        return {
            "win_rate": len(wins) / len(self.behavioral_patterns["win_loss_behavior"]) if self.behavioral_patterns["win_loss_behavior"] else 0,
            "avg_hold_times": {
                "wins": avg_win_hold_time,
                "losses": avg_loss_hold_time,
                "ratio": avg_loss_hold_time / avg_win_hold_time if avg_win_hold_time > 0 else 0
            },
            "position_sizing": {
                "trend": trend,
                "volatility": np.std(recent_sizes) if len(recent_sizes) > 1 else 0
            },
            "detected_biases": self.behavioral_patterns["detected_biases"]
        }
        
    async def get_memory_context(self):
        """
        Get comprehensive memory context for cross-agent integration.
        Similar to Athena's memory context method.
        
        Returns:
        --------
        dict: Memory context for risk management and trader behavior
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Get recent risk assessments
            risk_history = await self.recall_risk_history(days=30, limit=5)
            
            # Get recent violations
            violations = await self.recall_violations(limit=3)
            
            # Get recent regime transitions
            regime_memories = await self.memory_core.episodic_store.get_recent(
                agent_id=self.agent_id,
                limit=3,
                memory_types=["regime_transition"]
            )
            regime_transitions = [m["content"] for m in regime_memories]
            
            # Get behavioral insights
            behavioral_patterns = await self.get_behavioral_patterns()
            
            # Get cross-agent intelligence (simplified - would need multiple calls for different agents)
            cross_agent_insights = []  # TODO: Implement cross-agent memory retrieval if needed
            
            # Compile context
            context = {
                "timestamp": datetime.datetime.now().isoformat(),
                "agent_id": self.agent_id,
                "user_id": self.user_id,
                "risk_summary": {
                    "recent_assessments": risk_history,
                    "recent_violations": violations,
                    "behavioral_patterns": behavioral_patterns
                },
                "market_context": {
                    "recent_regime_transitions": regime_transitions,
                    "current_regime": self.market_regime_memory["regimes"][-1] if self.market_regime_memory["regimes"] else None
                },
                "cross_agent_insights": cross_agent_insights
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error building memory context: {e}")
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "agent_id": self.agent_id,
                "user_id": self.user_id,
                "error": str(e)
            }
    
    def _determine_violation_type(self, portfolio, trade):
        """Determine the type of violation based on portfolio and trade"""
        if trade and trade.get("position_size", 0) > 0.2:
            return "position_size_violation"
        elif len(portfolio) > 20:
            return "overtrading_violation"
        else:
            return "unknown_violation"
    
    async def _save_memory(self):
        """Save memory to persistent storage"""
        # If not initialized with memory system, just print notice
        if not self._initialized:
            print(f"Memory updated for user {self.user_id}")
            return
            
        try:
            # Store comprehensive memory state in memory system
            memory_state = {
                "user_id": self.user_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "risk_history_size": len(self.risk_history),
                "behavioral_patterns": self.behavioral_patterns,
                "market_regime_memory": self.market_regime_memory,
                "violation_log_size": len(self.violation_log)
            }
            
            # Store memory state snapshot
            await self.memory_core.store_memory(agent_id=self.agent_id,
                content=memory_state,
                memory_type="memory_state",
                tags=["chronos_memory_state", "risk_management"]
            )
            
            logger.debug(f"Memory state snapshot saved for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error saving memory state: {e}")
    
    def _load_memory(self):
        """Legacy method for backward compatibility"""
        # This is called by __init__, we'll do the actual loading in _load_memory_from_system
        pass
        
    async def _load_memory_from_system(self):
        """Load memory from the memory system"""
        if not self._initialized:
            logger.warning("Cannot load memory from system - not initialized")
            return
            
        try:
            # Retrieve memory state from memory system
            memories = await self.memory_core.episodic_store.get_recent(
                agent_id=self.agent_id,
                limit=1,
                memory_types=["memory_state"]
            )
            
            if memories:
                # Get the memory state
                memory_state = memories[0]["content"]
                
                # Update behavioral patterns and market regime memory
                self.behavioral_patterns = memory_state.get("behavioral_patterns", self.behavioral_patterns)
                self.market_regime_memory = memory_state.get("market_regime_memory", self.market_regime_memory)
                
                logger.info(f"Loaded memory state for user {self.user_id}")
                
                # Also load risk history
                risk_history = await self.recall_risk_history(days=self.max_history, limit=self.max_history)
                if risk_history:
                    self.risk_history = deque(risk_history, maxlen=self.max_history)
                    
                # And violations
                violations = await self.recall_violations(limit=100)
                if violations:
                    self.violation_log = violations
                    
                logger.info(f"Successfully loaded {len(self.risk_history)} risk history entries and {len(self.violation_log)} violations")
                    
            else:
                logger.info(f"No existing memory state found for user {self.user_id}")
                
        except Exception as e:
            logger.error(f"Error loading memory from system: {e}")
            import traceback
            logger.error(traceback.format_exc())