"""
Athena Agent Connector
Bridges terminal commands to the Athena agent instance
Handles agent initialization, lifecycle, and method calls
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base import AgentConnector


logger = logging.getLogger(__name__)


class AthenaConnector(AgentConnector):
    """Connector for Athena - Market Intelligence Agent"""
    
    def __init__(self):
        super().__init__("Athena")
    
    async def initialize(self):
        """Initialize Athena agent"""
        try:
            from src.agents.athena_workspace.athena import AthenaAgent
            
            self.agent_instance = AthenaAgent(
                config_path="config/tradding_config.yaml",
                use_llm=True,
                use_memory=True,
                use_redis=True
            )
            
            await self.agent_instance.initialize()
            self.initialized = True
            logger.info("Athena agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Athena: {e}", exc_info=True)
            return False
    
    async def analyze_market(self, symbol: str, interval: int = 3600, count: int = 100) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis
        
        Args:
            symbol: Trading symbol (e.g., BTCUSD)
            interval: Timeframe in seconds
            count: Number of historical candles
            
        Returns:
            Market analysis result dictionary
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Athena agent not initialized"}
        
        try:
            result = await self.agent_instance.observe(symbol, interval, count)
            return result
        except Exception as e:
            logger.error(f"Athena analysis error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def detect_regime(self, symbol: str, interval: int = 3600) -> Dict[str, Any]:
        """
        Detect current market regime
        
        Args:
            symbol: Trading symbol
            interval: Timeframe in seconds
            
        Returns:
            Regime detection result
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Athena agent not initialized"}
        
        try:
            # Get market analysis which includes regime detection
            result = await self.agent_instance.observe(symbol, interval, 100)
            
            if 'regime' in result:
                return {
                    "regime": result['regime'],
                    "confidence": result.get('regime_confidence', 0),
                    "symbol": symbol,
                    "timestamp": result.get('timestamp', datetime.now().isoformat())
                }
            else:
                return {"error": "Regime detection not available in result"}
                
        except Exception as e:
            logger.error(f"Regime detection error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def detect_patterns(self, symbol: str, interval: int = 3600, count: int = 100) -> Dict[str, Any]:
        """
        Detect chart patterns in the market
        
        Args:
            symbol: Trading symbol
            interval: Timeframe in seconds
            count: Number of historical candles
            
        Returns:
            Pattern detection results
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Athena agent not initialized"}
        
        try:
            result = await self.agent_instance.observe(symbol, interval, count)
            
            if 'patterns' in result:
                return {
                    "patterns": result['patterns'],
                    "trading_bias": result.get('trading_bias', {}),
                    "symbol": symbol,
                    "timestamp": result.get('timestamp', datetime.now().isoformat())
                }
            else:
                return {"error": "Pattern detection not available"}
                
        except Exception as e:
            logger.error(f"Pattern detection error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def get_memory_context(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve memory context from Athena's memory system
        
        Args:
            symbol: Optional symbol to filter context
            
        Returns:
            Memory context dictionary
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Athena agent not initialized"}
        
        try:
            context = await self.agent_instance.get_memory_context(symbol=symbol)
            return context
        except Exception as e:
            logger.error(f"Memory context retrieval error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def get_recent_history(self, symbol: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        Get recent observation history from memory
        
        Args:
            symbol: Optional symbol to filter by
            limit: Maximum number of observations
            
        Returns:
            Recent observation history
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Athena agent not initialized"}
        
        try:
            history = await self.agent_instance.get_recent_history(symbol=symbol, limit=limit)
            return {
                "history": history,
                "count": len(history),
                "symbol_filter": symbol,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"History retrieval error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def get_current_insights(self, top_n: int = 3) -> Dict[str, Any]:
        """
        Get consolidated current market insights
        
        Args:
            top_n: Number of top insights to retrieve
            
        Returns:
            Current market insights
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Athena agent not initialized"}
        
        try:
            insights = self.agent_instance.get_current_insights(top_n=top_n)
            return insights
        except Exception as e:
            logger.error(f"Insights retrieval error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def observe_multiple(self, symbols: List[str], interval: int = 3600) -> Dict[str, Any]:
        """
        Observe multiple symbols and rank by opportunity
        
        Args:
            symbols: List of trading symbols
            interval: Timeframe in seconds
            
        Returns:
            Sorted list of market contexts
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.agent_instance:
            return {"error": "Athena agent not initialized"}
        
        try:
            contexts = await self.agent_instance.observe_multiple(symbols, interval)
            return {
                "contexts": contexts,
                "count": len(contexts),
                "symbols": symbols,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Multi-symbol observation error: {e}", exc_info=True)
            return {"error": str(e)}