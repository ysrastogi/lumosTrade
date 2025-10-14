import asyncio
import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from settings import settings

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from src.agents.athena import AthenaAgent
from src.agents.models import (
    MarketContext,
    MarketInsightsResponse,
    SymbolRequest,
    MultiSymbolRequest
)

logger = logging.getLogger(__name__)

athena_router = APIRouter(prefix="/api/athena", tags=["athena"])

API_ONLY_MODE = os.environ.get("API_ONLY_MODE", "false").lower() == "true"

# Singleton instance cache for AthenaAgent
_athena_agent_instance = None


def get_athena_agent(force_new: bool = False) -> AthenaAgent:
    global _athena_agent_instance
        
    if _athena_agent_instance is None or force_new:
        try:
            gemini_api_key = settings.gemini_api_key
            _athena_agent_instance = AthenaAgent(
                gemini_api_key=gemini_api_key
            )
            logger.info("Created new Athena agent instance")
        except Exception as e:
            logger.error(f"Error creating Athena agent: {e}")
    return _athena_agent_instance


@athena_router.post("/analyze/symbol", response_model=MarketContext)
async def analyze_symbol(request: SymbolRequest):
    """Analyze a single symbol with Athena"""
    try:
        
        agent = get_athena_agent()
        await agent.initialize()
        context = await agent.observe(
            symbol=request.symbol, 
            interval=request.interval
        )
        
        if "error" in context:
            raise HTTPException(status_code=400, detail=f"Analysis error: {context['error']}")
            
        return context
    
    except Exception as e:
        logger.error(f"Error analyzing symbol {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@athena_router.post("/analyze/multi", response_model=List[MarketContext])
async def analyze_multiple_symbols(request: MultiSymbolRequest):
    """Analyze multiple symbols with Athena"""
    try:
        if API_ONLY_MODE:
            # In API-only mode, respond with a message that analysis is handled by worker
            logger.info(f"Multi-symbol analysis request received in API-only mode")
            return [
                {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "message": "Analysis request received. Processing by worker.",
                    "status": "queued"
                }
                for symbol in request.symbols
            ]
        
        # Normal mode - get or create Athena agent
        agent = get_athena_agent()
        
        # Always initialize the agent to ensure connections are ready
        await agent.initialize()
            
        # Analyze the requested symbols
        contexts = await agent.observe_multiple(
            symbols=request.symbols,
            interval=request.interval
        )
        
        return contexts
    
    except Exception as e:
        logger.error(f"Error analyzing multiple symbols: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multi-symbol analysis failed: {str(e)}")


@athena_router.get("/insights", response_model=MarketInsightsResponse)
async def get_market_insights(top_n: int = Query(3, ge=1, le=10)):
    """Get consolidated market insights from Athena"""
    try:
        if API_ONLY_MODE:
            # In API-only mode, respond with a message that insights are handled by worker
            logger.info(f"Market insights request received in API-only mode")
            return {
                "timestamp": datetime.now().isoformat(),
                "message": "Market insights request received. Processing by worker.",
                "status": "queued"
            }
        
        # Normal mode - get or create Athena agent
        agent = get_athena_agent()
        
        # Check if we have any insights available
        if not agent.context_history:
            # Need to fetch some data first - analyze a default set of symbols
            default_symbols = ["R_100", "R_50", "BOOM1000", "CRASH1000"]
            await agent.observe_multiple(symbols=default_symbols)
            
        # Get insights
        insights = agent.get_current_insights(top_n=top_n)
        return insights
    
    except Exception as e:
        logger.error(f"Error getting market insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")


# Add methods to DashboardAPI class to integrate Athena
def extend_dashboard_api_with_athena():
    """Extend the DashboardAPI class with Athena methods"""
    from src.api.dashboard_api import DashboardAPI
    
    async def get_athena_analysis(self, symbol: str, interval: int = 60) -> MarketContext:
        """
        Get Athena analysis for a specific symbol.
        
        Args:
            symbol: Symbol to analyze
            interval: Timeframe in seconds
            
        Returns:
            Complete market context
        """
        if not hasattr(self, '_athena_agent') or self._athena_agent is None:
            self._athena_agent = get_athena_agent()
            await self._athena_agent.initialize()
            
        context = await self._athena_agent.observe(symbol, interval)
        return context
    
    async def get_market_insights(self, top_n: int = 3) -> MarketInsightsResponse:
        """
        Get consolidated market insights.
        
        Args:
            top_n: Number of top contexts to include
            
        Returns:
            Market insights response
        """
        if not hasattr(self, '_athena_agent') or self._athena_agent is None:
            self._athena_agent = get_athena_agent()
            await self._athena_agent.initialize()
            
        return self._athena_agent.get_current_insights(top_n)
    
    # Add methods to the DashboardAPI class
    DashboardAPI.get_athena_analysis = get_athena_analysis
    DashboardAPI.get_market_insights = get_market_insights
    
    return DashboardAPI


# Register extensions
extend_dashboard_api_with_athena()