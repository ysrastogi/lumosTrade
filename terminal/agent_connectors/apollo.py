"""
Apollo Agent Connector
Bridges terminal commands to the Apollo agent instance
Handles agent initialization, lifecycle, and method calls for signal generation
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base import AgentConnector
from src.agents.apollo_workspace.models import Signal
from src.utils.signal_formatter import normalize_signal_dict, normalize_signal_list



logger = logging.getLogger(__name__)


class ApolloConnector(AgentConnector):
    """Connector for Apollo - Signal Generation Agent"""
    
    def __init__(self, athena_connector=None):
        super().__init__("Apollo")
        self.athena_connector = athena_connector
    
    async def initialize(self):
        """Initialize Apollo agent with shared memory"""
        try:
            from src.agents.apollo_workspace.apollo import ApolloAgent
            from src.llm.client import gemini
            from src.agents.apollo_workspace.memory_manager import ApolloMemoryManager
            from src.agents.athena_workspace.memory_manager import get_global_memory_core
            
            # Use global memory core for cross-agent sharing
            global_memory_core = get_global_memory_core(use_redis=True)
            memory_manager = ApolloMemoryManager(use_redis=True, memory_core=global_memory_core)
            await memory_manager.initialize()
            
            self.agent_instance = ApolloAgent(
                llm_client=gemini,
                memory_system=memory_manager,
                use_redis=True
            )
            
            self.initialized = True
            logger.info("Apollo agent initialized successfully with shared memory")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Apollo: {e}", exc_info=True)
            return False
    
    async def generate_signal(self, symbol: str, signal_type: str = "any") -> Dict[str, Any]:
        """
        Generate trading signal using cross-agent memory
        
        Args:
            symbol: Trading symbol
            signal_type: Type of signal to generate
            
        Returns:
            Trading signal dictionary
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Apollo agent not initialized"}
        
        try:
            # First try to use Apollo's own memory system to access cross-agent memory
            if hasattr(self.agent_instance, 'memory_system') and self.agent_instance.memory_system:
                try:
                    # Get global memory context directly from Apollo's memory manager
                    memory_context = await self.agent_instance.memory_system.get_global_memory_context(symbol=symbol)
                    
                    if memory_context and 'error' not in memory_context:
                        # Check if we have observations in the memory context
                        has_observations = False
                        
                        if 'agent_observations' in memory_context:
                            # Check all agents for observations about this symbol
                            for agent_id, observations in memory_context['agent_observations'].items():
                                if any(obs.get('symbol') == symbol for obs in observations):
                                    has_observations = True
                                    break
                        
                        if has_observations:
                            logger.info(f"Using cross-agent memory for {symbol}")
                            return await self.generate_signals_from_athena(symbol, memory_context)
                
                except Exception as e:
                    logger.warning(f"Failed to retrieve cross-agent memory context: {e}")
            
            # If cross-agent memory didn't work, fall back to Athena connector
            if self.athena_connector and hasattr(self.athena_connector, 'get_memory_context'):
                try:
                    athena_context = await self.athena_connector.get_memory_context(symbol)
                
                    if athena_context and 'error' not in athena_context:
                        logger.info(f"Using Athena memory context for {symbol}")
                        return await self.generate_signals_from_athena(symbol, athena_context)
                    
                except Exception as e:
                    logger.warning(f"Failed to retrieve Athena memory context: {e}")
            
            # Try to analyze market directly with Athena
            if self.athena_connector and hasattr(self.athena_connector, 'analyze_market'):
                try:
                    logger.info(f"Getting fresh market analysis for {symbol} from Athena")
                    athena_context = await self.athena_connector.analyze_market(symbol)
                    if 'error' not in athena_context:
                        return await self.generate_signals_from_athena(symbol, athena_context)
                except Exception as e:
                    logger.warning(f"Athena market analysis failed: {e}")
            
            # If we reach here, we need to generate signals without Athena
            logger.info(f"Generating standalone signal for {symbol} using Apollo's own analysis")
            
            # Use Apollo's native signal generation capabilities
            # This calls Apollo's own analysis without relying on Athena
            signals = await self.agent_instance.generate_signals(
                symbol=symbol,
                signal_type=signal_type
            )
            
            if not signals:
                return {
                    "symbol": symbol,
                    "signals": [],
                    "count": 0,
                    "message": "No signals generated in standalone mode"
                }
            
            # Use the signal formatter utility to normalize the signal
            signal_dict = normalize_signal_dict(signals[0])
            signal_dict["timestamp"] = datetime.now().isoformat()
            signal_dict["source"] = "apollo_standalone"
            
            return signal_dict
        except Exception as e:
            logger.error(f"Signal generation error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def generate_signals_from_athena(self, symbol: str, athena_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on Athena's market analysis
        
        Args:
            symbol: Trading symbol
            athena_context: Market analysis from Athena (could be direct analysis or from memory)
            
        Returns:
            Dictionary containing generated signals and metadata
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Apollo agent not initialized"}
        
        try:
            # Handle different context formats (direct observation vs memory context)
            observation_data = athena_context
            
            # If we got memory context with observations from the standard context structure
            if 'observations' in athena_context and isinstance(athena_context['observations'], list):
                # Get most recent observation for this symbol
                symbol_observations = [
                    obs for obs in athena_context['observations'] 
                    if obs.get('content', {}).get('symbol') == symbol or
                       obs.get('content', {}).get('data', {}).get('symbol') == symbol
                ]
                
                if symbol_observations:
                    # Sort by timestamp if available and use the most recent
                    symbol_observations.sort(
                        key=lambda x: x.get('timestamp', ''), 
                        reverse=True
                    )
                    observation_data = symbol_observations[0]
                    
            # Check for cross-agent memory structure 
            elif 'agent_observations' in athena_context and isinstance(athena_context['agent_observations'], dict):
                # Look for Athena observations in the cross-agent structure
                all_observations = []
                for agent_id, observations in athena_context['agent_observations'].items():
                    # Only include observations for the requested symbol
                    for obs in observations:
                        if obs.get('symbol') == symbol:
                            all_observations.append(obs)
                
                if all_observations:
                    # Sort by timestamp if available and use the most recent
                    all_observations.sort(
                        key=lambda x: x.get('timestamp', ''), 
                        reverse=True
                    )
                    observation_data = all_observations[0]
            
            # Use Apollo's process_athena_observation method
            signals = await self.agent_instance.process_athena_observation(symbol, observation_data)
            
            if not signals:
                return {
                    "symbol": symbol,
                    "signals": [],
                    "count": 0,
                    "message": "No signals generated from current market conditions",
                    "athena_regime": observation_data.get('regime', 'Unknown'),
                    "source": "memory" if 'observations' in athena_context else "direct"
                }
            
            # Use the signal formatter utility to normalize signals
            signal_dicts = normalize_signal_list(signals)
            
            return {
                "symbol": symbol,
                "signals": signal_dicts,
                "count": len(signal_dicts),
                "athena_regime": athena_context.get('regime', 'Unknown'),
                "regime_confidence": athena_context.get('regime_confidence', 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Athena-based signal generation error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def get_stored_signals(self, symbol: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        Retrieve previously generated signals from memory
        
        Args:
            symbol: Optional symbol to filter signals
            limit: Maximum number of signals to retrieve
            
        Returns:
            Dictionary containing stored signals
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Apollo agent not initialized"}
        
        try:
            # Access memory system to retrieve stored signals
            if hasattr(self.agent_instance, 'memory_system') and self.agent_instance.memory_system:
                # Get signals from memory using global memory context
                memory_context = await self.agent_instance.memory_system.get_context(symbol)
                
                # Extract signals from memory context
                signals = []
                
                # Try to find signals in memory
                if 'agent_insights' in memory_context and isinstance(memory_context['agent_insights'], dict):
                    # Check in cross-agent memory structure
                    for agent_id, insights in memory_context['agent_insights'].items():
                        for insight in insights:
                            # Only include signals for this symbol
                            if symbol and insight.get('symbol') != symbol:
                                continue
                                
                            if insight.get('insight_type') == 'trading_signal':
                                from src.agents.apollo_workspace.models import Signal
                                try:
                                    signal = Signal(
                                        symbol=insight.get('symbol'),
                                        direction=insight.get('direction', '').lower(),
                                        entry=insight.get('entry_price'),
                                        stop_loss=insight.get('stop_loss'),
                                        take_profit=insight.get('take_profit'),
                                        pattern=insight.get('pattern', 'unknown'),
                                        confidence=insight.get('confidence', 0.5),
                                        timeframe=insight.get('timeframe', '1h')
                                    )
                                    signals.append(signal)
                                except Exception as e:
                                    logger.debug(f"Failed to create Signal from insight: {e}")
                
                # If no signals found in cross-agent structure, try direct retrieval
                if not signals:
                    signals = await self.agent_instance.memory_system.get_signals(
                        symbol=symbol,
                        limit=limit
                    )
                
                if signals:
                    # Use the signal formatter utility to normalize signals
                    signal_dicts = normalize_signal_list(signals)
                    
                    # Ensure all signals have a timestamp
                    for signal in signal_dicts:
                        if not signal.get('timestamp'):
                            signal['timestamp'] = datetime.now().isoformat()
                    
                    return {
                        "signals": signal_dicts,
                        "count": len(signal_dicts),
                        "symbol_filter": symbol,
                        "limit": limit,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "signals": [],
                        "count": 0,
                        "symbol_filter": symbol,
                        "limit": limit,
                        "message": "No signals found in memory"
                    }
            else:
                return {"error": "Memory system not available"}
                
        except Exception as e:
            logger.error(f"Signal retrieval error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def validate_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a trading signal using historical data and memory context
        
        Args:
            signal: Signal dictionary to validate
            
        Returns:
            Validation results
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Apollo agent not initialized"}
        
        try:
            # Use Apollo's backtest validator if available
            if hasattr(self.agent_instance, 'backtest_validator') and self.agent_instance.backtest_validator:
                # Create a Signal object from dictionary if needed
                from src.agents.apollo_workspace.models import Signal
                
                # Convert to Signal object if needed
                if not isinstance(signal, Signal):
                    # Normalize the signal dictionary to ensure consistent keys
                    norm_signal = normalize_signal_dict(signal)
                    
                    signal_obj = Signal(
                        symbol=norm_signal.get('symbol'),
                        direction=norm_signal.get('direction', '').lower(),
                        entry=norm_signal.get('entry_price'),
                        stop_loss=norm_signal.get('stop_loss'),
                        take_profit=norm_signal.get('take_profit'),
                        pattern=norm_signal.get('pattern', 'unknown'),
                        confidence=norm_signal.get('confidence', 0.5)/100 if norm_signal.get('confidence', 0.5) > 1 else norm_signal.get('confidence', 0.5),
                        timeframe=norm_signal.get('timeframe', '1h')
                    )
                else:
                    signal_obj = signal
                    
                # Get additional context from memory to enrich validation
                market_context = {}
                if hasattr(self.agent_instance, 'memory_system') and self.agent_instance.memory_system:
                    try:
                        # Get global memory context for this symbol
                        memory_context = await self.agent_instance.memory_system.get_global_memory_context(
                            symbol=signal_obj.symbol
                        )
                        
                        if memory_context and 'error' not in memory_context:
                            market_context = memory_context
                    except Exception as e:
                        logger.debug(f"Failed to get memory context for validation: {e}")
                
                # Run actual historical validation with market context
                validation_result = await self.agent_instance.backtest_validator.validate_signal(
                    signal_obj, 
                    market_context=market_context
                )
                
                # Enrich validation result with memory insights if available
                if market_context and 'agent_insights' in market_context:
                    validation_result['memory_context_available'] = True
                    
                    # Add summary of market regime from memory
                    if 'global_stats' in market_context:
                        validation_result['context_stats'] = market_context['global_stats']
                
                return validation_result
            else:
                logger.warning("Signal validation requested but backtest validator not available")
                return {
                    "valid": None,  # None indicates validation was not performed
                    "message": "Validation not available - backtest validator not configured"
                }
                
        except Exception as e:
            logger.error(f"Signal validation error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def batch_generate_signals(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Generate signals for multiple symbols
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary with signals for all symbols
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Apollo agent not initialized"}
        
        try:
            all_signals = {}
            
            # Generate signals for each symbol
            for symbol in symbols:
                # First try to use Apollo's own memory system to access cross-agent memory
                if hasattr(self.agent_instance, 'memory_system') and self.agent_instance.memory_system:
                    try:
                        # Get global memory context directly from Apollo's memory manager
                        memory_context = await self.agent_instance.memory_system.get_global_memory_context(symbol=symbol)
                        if memory_context and 'error' not in memory_context:
                            # Check if we have observations in the memory context
                            has_observations = (
                                ('agent_observations' in memory_context and 
                                 any(len(obs) > 0 for obs in memory_context['agent_observations'].values()))
                            )
                            
                            if has_observations:
                                result = await self.generate_signals_from_athena(symbol, memory_context)
                                all_signals[symbol] = result
                                logger.info(f"Generated signals for {symbol} using cross-agent memory")
                                continue
                    except Exception as e:
                        logger.debug(f"Cross-agent memory retrieval failed for {symbol}: {e}")
                
                # Then try to get context from Athena's memory system as fallback
                if self.athena_connector and hasattr(self.athena_connector, 'get_memory_context'):
                    try:
                        # Try to get context from Athena memory
                        memory_context = await self.athena_connector.get_memory_context(symbol)
                        if memory_context and 'error' not in memory_context:
                            result = await self.generate_signals_from_athena(symbol, memory_context)
                            all_signals[symbol] = result
                            logger.info(f"Generated signals for {symbol} using Athena memory context")
                            continue
                    except Exception as e:
                        logger.debug(f"Athena memory context retrieval failed for {symbol}, trying direct analysis: {e}")
                
                # If memory context not available, try direct market analysis
                if self.athena_connector and hasattr(self.athena_connector, 'analyze_market'):
                    try:
                        athena_context = await self.athena_connector.analyze_market(symbol)
                        if 'error' not in athena_context:
                            result = await self.generate_signals_from_athena(symbol, athena_context)
                            all_signals[symbol] = result
                            logger.info(f"Generated signals for {symbol} using fresh analysis")
                            continue
                    except Exception as e:
                        logger.warning(f"Athena integration failed for {symbol}, using standalone: {e}")
                
                # Fallback to standalone signal generation using Apollo's native capabilities
                try:
                    logger.info(f"Generating standalone signal for {symbol} using Apollo's own analysis")
                    # Generate signals directly using Apollo's native analysis
                    signals = await self.agent_instance.generate_signals(
                        symbol=symbol,
                        signal_type="any"  # Any valid signal type
                    )
                    
                    if signals:
                        # Use the signal formatter utility to normalize signals
                        signal_dicts = normalize_signal_list(signals)
                        
                        # Ensure all signals have a timestamp
                        for signal in signal_dicts:
                            if not signal.get('timestamp'):
                                signal['timestamp'] = datetime.now().isoformat()
                            
                        all_signals[symbol] = {
                            "symbol": symbol,
                            "signals": signal_dicts,
                            "count": len(signal_dicts),
                            "source": "apollo_standalone",
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        all_signals[symbol] = {
                            "symbol": symbol,
                            "signals": [],
                            "count": 0,
                            "message": "No signals generated in standalone mode",
                            "timestamp": datetime.now().isoformat()
                        }
                except Exception as e:
                    logger.error(f"Failed to generate standalone signal for {symbol}: {e}")
                    all_signals[symbol] = {
                        "symbol": symbol,
                        "error": str(e),
                        "signals": [],
                        "count": 0
                    }
            
            return {
                "symbols": symbols,
                "results": all_signals,
                "total_symbols": len(symbols),
                "total_signals": sum(r.get('count', 0) for r in all_signals.values()),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Batch signal generation error: {e}", exc_info=True)
            return {"error": str(e)}