"""
Agent Orchestrator for LumosTrade Terminal
Routes commands to appropriate agents and handles responses
"""

from typing import Dict, Any, Optional, Tuple
from terminal.agent_manager import AgentManager
from terminal.command_parser import CommandType
from terminal.chart import create_chart
import asyncio
import logging

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrate command routing to agents"""
    
    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        self.conversation_history = []
        
        # Lazy load flow manager and connector manager
        self._flow_manager = None
        self._connector_manager = None
    
    @property
    def flow_manager(self):
        """Lazy load flow manager"""
        if self._flow_manager is None:
            from terminal.agent_flow import AgentFlowManager
            self._flow_manager = AgentFlowManager(connector_manager=self.connector_manager)
        return self._flow_manager
    
    @property
    def connector_manager(self):
        """Lazy load connector manager"""
        if self._connector_manager is None:
            from terminal.agent_connectors import AgentConnectorManager
            self._connector_manager = AgentConnectorManager()
        return self._connector_manager
    
    def route(self, command_type: CommandType, metadata: Dict[str, Any]) -> Tuple[str, str, Optional[Dict]]:
        """
        Route command to appropriate handler
        
        Args:
            command_type: Type of command
            metadata: Command metadata from parser
            
        Returns:
            Tuple of (agent_name, response_message, response_metadata)
        """
        if command_type == CommandType.EXIT:
            return self._handle_exit(metadata)
        
        elif command_type == CommandType.HELP:
            return self._handle_help(metadata)
        
        elif command_type == CommandType.STATUS:
            return self._handle_status(metadata)
        
        elif command_type == CommandType.CHART:
            return self._handle_chart(metadata)
        
        elif command_type == CommandType.LIVE_CHART:
            return self._handle_live_chart(metadata)
        
        elif command_type == CommandType.AGENT_FLOW:
            return self._handle_agent_flow(metadata)
        
        elif command_type == CommandType.AGENT_QUERY:
            return self._handle_agent_query(metadata)
        
        else:
            return self._handle_unknown(metadata)
    
    def _handle_exit(self, metadata: Dict[str, Any]) -> Tuple[str, str, Optional[Dict]]:
        """Handle exit command"""
        return 'system', 'EXIT_COMMAND', None
    
    def _handle_help(self, metadata: Dict[str, Any]) -> Tuple[str, str, Optional[Dict]]:
        """Handle help command - returns special marker for CLI to show help"""
        return 'system', 'HELP_COMMAND', None
    
    def _handle_status(self, metadata: Dict[str, Any]) -> Tuple[str, str, Optional[Dict]]:
        """Handle status command - returns special marker for CLI to show status"""
        return 'system', 'STATUS_COMMAND', None
    
    def _handle_chart(self, metadata: Dict[str, Any]) -> Tuple[str, str, Optional[Dict]]:
        """
        Handle chart/plot command
        
        Args:
            metadata: Contains 'symbol' and 'interval'
            
        Returns:
            Tuple of (agent_name, chart_output, metadata)
        """
        symbol = metadata.get('symbol')
        interval = metadata.get('interval', 60)
        
        # Check for errors in parsing
        if 'error' in metadata:
            return 'system', metadata['error'], None
        
        # Validate symbol is provided
        if not symbol:
            return 'system', "❌ Please specify a symbol. Usage: /chart <SYMBOL> [interval]\nExample: /chart BTCUSD 1m", None
        
        # Generate the chart
        try:
            chart_output = create_chart(symbol, interval)
            return 'chart', chart_output, {'symbol': symbol, 'interval': interval}
        except Exception as e:
            logger.error(f"Error generating chart: {e}", exc_info=True)
            return 'system', f"❌ Failed to generate chart: {str(e)}", None
    
    def _handle_live_chart(self, metadata: Dict[str, Any]) -> Tuple[str, str, Optional[Dict]]:
        """
        Handle live chart command
        
        Args:
            metadata: Contains 'symbol'/'symbols', 'interval', and 'refresh_rate'
            
        Returns:
            Tuple of (agent_name, response, metadata)
        """
        from terminal.live_chart import start_live_chart, start_live_multi_chart
        
        # Check for errors in parsing
        if 'error' in metadata:
            return 'system', metadata['error'], None
        
        symbol = metadata.get('symbol')
        symbols = metadata.get('symbols')
        interval = metadata.get('interval', 60)
        refresh_rate = metadata.get('refresh_rate', 1.0)
        
        # Validate symbol or symbols is provided
        if not symbol and not symbols:
            return 'system', (
                "❌ Please specify a symbol. Usage: /live <SYMBOL> [interval] [refresh_rate]\n"
                "Examples:\n"
                "  /live BTCUSD 1m 1.0\n"
                "  /live BTCUSD,ETHUSD 5m 2.0"
            ), None
        
        # Start the appropriate live chart
        try:
            if symbols:
                # Multi-symbol chart
                start_live_multi_chart(symbols, interval, refresh_rate)
            else:
                # Single symbol chart
                start_live_chart(symbol, interval, refresh_rate)
            
            # Return after live chart exits
            return 'chart', '\n✅ Live chart session ended.', {
                'symbol': symbol or ','.join(symbols),
                'interval': interval,
                'refresh_rate': refresh_rate
            }
        except KeyboardInterrupt:
            return 'chart', '\n✅ Live chart session ended.', None
        except Exception as e:
            logger.error(f"Error in live chart: {e}", exc_info=True)
            return 'system', f"❌ Failed to start live chart: {str(e)}", None
    
    def _handle_agent_query(self, metadata: Dict[str, Any]) -> Tuple[str, str, Optional[Dict]]:
        """
        Handle query to an agent
        
        Args:
            metadata: Contains 'query' and 'suggested_agent'
            
        Returns:
            Tuple of (agent_name, response, metadata)
        """
        query = metadata.get('query', '')
        suggested_agent = metadata.get('suggested_agent')
        
        # If no agent suggested, use a general routing logic
        if not suggested_agent:
            suggested_agent = 'athena'  # Default to Athena for general queries
        
        # Check if agent is active
        if not self.agent_manager.is_active(suggested_agent):
            return 'system', f"Agent {suggested_agent.upper()} is currently inactive.", None
        
        # Get simulated response from agent
        response = self._get_agent_response(suggested_agent, query)
        
        # Store in conversation history
        self.conversation_history.append({
            'agent': suggested_agent,
            'query': query,
            'response': response,
            'timestamp': self.agent_manager.start_time
        })
        
        response_meta = {
            'query': query,
            'confidence': 0.85  # Simulated confidence
        }
        
        return suggested_agent, response, response_meta
    
    def _handle_agent_flow(self, metadata: Dict[str, Any]) -> Tuple[str, str, Optional[Dict]]:
        """
        Handle interactive agent flow
        
        Args:
            metadata: Contains 'agent' and 'flow_type'
            
        Returns:
            Tuple of (agent_name, response, metadata)
        """
        agent_name = metadata.get('agent', '')
        flow_type = metadata.get('flow_type', '')
        
        if not agent_name or not flow_type:
            return 'system', 'Invalid flow command', None
        
        # Check if agent is active
        if not self.agent_manager.is_active(agent_name):
            return 'system', f"Agent {agent_name.upper()} is currently inactive.", None
        
        try:
            # Get the flow handler
            flow = self.flow_manager.get_flow(agent_name)
            if not flow:
                return 'system', f"No flow handler found for agent: {agent_name}", None
            
            # Get the agent connector
            connector = self.connector_manager.get_connector(agent_name)
            if connector:
                flow.agent_connector = connector
            
            # Run the flow asynchronously
            result = asyncio.run(self.flow_manager.run_flow(agent_name, flow_type))
            
            if 'error' in result:
                return agent_name, f"Flow error: {result['error']}", None
            
            # Flow completed successfully
            return agent_name, "\n✅ Interactive flow completed.", result
            
        except KeyboardInterrupt:
            return agent_name, "\n⚠️ Flow cancelled by user.", None
        except Exception as e:
            logger.error(f"Flow error for {agent_name}/{flow_type}: {e}", exc_info=True)
            return 'system', f"Error running flow: {str(e)}", None
    
    def _handle_unknown(self, metadata: Dict[str, Any]) -> Tuple[str, str, Optional[Dict]]:
        """Handle unknown command"""
        error_msg = metadata.get('error', 'Unknown command')
        return 'system', f"I didn't understand that command. {error_msg}\nType /help for assistance.", None
    
    def _get_agent_response(self, agent_name: str, query: str) -> str:
        """
        Get simulated response from agent
        (In future, this will call actual agent instances)
        
        Args:
            agent_name: Name of agent
            query: User query
            
        Returns:
            Simulated response string
        """
        # Simulated responses for each agent
        responses = {
            'athena': self._athena_response(query),
            'apollo': self._apollo_response(query),
            'chronos': self._chronos_response(query),
            'daedalus': self._daedalus_response(query),
            'hermes': self._hermes_response(query)
        }
        
        return responses.get(agent_name, "I'm processing your request...")
    
    def _athena_response(self, query: str) -> str:
        """Simulated Athena (Market Intelligence) response"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['btc', 'bitcoin']):
            return """Based on current market analysis:

• BTC is showing bullish momentum with strong support at $60,000
• 24h volume: +15% (indicating increased interest)
• Market regime: TRENDING BULLISH
• Key resistance levels: $65,000, $68,000

Technical indicators suggest continuation pattern forming.
Watch for breakout above $65K for potential rally to $70K."""

        elif any(word in query_lower for word in ['eth', 'ethereum']):
            return """Ethereum market analysis:

• ETH trading at strong support zone around $3,200
• Relative strength vs BTC: Neutral
• Network activity: High (gas fees elevated)
• Market regime: CONSOLIDATING

Waiting for directional move. Monitor BTC correlation."""

        elif any(word in query_lower for word in ['market', 'trend', 'overview']):
            return """Overall market overview:

• Market Sentiment: CAUTIOUSLY BULLISH
• Major Indices: Crypto market cap +8% this week
• Volatility: Moderate (VIX equivalent: 45)
• Top Movers: BTC (+5%), ETH (+3%), SOL (+12%)

Key themes: Institutional accumulation continues, 
regulatory clarity improving in major markets."""

        else:
            return f"""Market Intelligence Analysis for your query:
"{query}"

I'm currently analyzing market conditions. Here's what I can tell you:
• Market is in a consolidation phase
• Looking for confirmation signals
• Risk-reward ratio appears favorable

Would you like me to dive deeper into any specific aspect?"""
    
    def _apollo_response(self, query: str) -> str:
        """Simulated Apollo (Signal Generation) response"""
        query_lower = query.lower()
        
        if 'signal' in query_lower or 'trade' in query_lower:
            return """Trading Signal Analysis:

🎯 SIGNAL DETECTED: BTC LONG OPPORTUNITY

Entry Zone: $62,000 - $62,500
Target 1: $65,000 (+4.0%)
Target 2: $68,000 (+8.8%)
Stop Loss: $60,000 (-3.3%)

Confidence: 78%
Timeframe: 4H - Daily

Supporting Factors:
✓ Higher lows forming on 4H chart
✓ Volume profile supports upside
✓ RSI showing positive divergence
✓ Confluence with key Fibonacci level

Risk Factors:
⚠ Macro uncertainty (Fed policy)
⚠ Resistance at $65K historically strong

Recommendation: Consider scaled entry with partial position."""

        else:
            return f"""Signal Analysis for: "{query}"

Currently scanning for high-probability setups...

Active Signals:
• BTC: BULLISH (Confidence: 78%)
• ETH: NEUTRAL (Waiting for confirmation)
• Market Conditions: FAVORABLE

I recommend waiting for optimal entry points.
Type 'show signal details' for comprehensive analysis."""
    
    def _chronos_response(self, query: str) -> str:
        """Simulated Chronos (Risk Management) response"""
        query_lower = query.lower()
        
        if 'risk' in query_lower or 'portfolio' in query_lower or 'position' in query_lower:
            return """Risk Assessment Summary:

📊 PORTFOLIO HEALTH CHECK

Current Exposure: 65% (MODERATE)
Recommended: 50-70% (You're within range)

Position Breakdown:
• BTC: 40% of portfolio
• ETH: 25% of portfolio  
• Alts: 35% of portfolio

Risk Metrics:
✓ Portfolio Sharpe Ratio: 1.8 (Good)
✓ Maximum Drawdown: -15% (Acceptable)
✓ Correlation Risk: LOW
⚠ Volatility: MODERATE (monitor closely)

Recommendations:
1. Current positions well-balanced
2. Consider taking partial profits on BTC above $65K
3. Maintain 30% cash reserve for opportunities
4. Stop-loss discipline: CRITICAL

Overall Risk Level: MODERATE ✓"""

        elif 'balance' in query_lower:
            return """Balance & Position Overview:

Total Portfolio Value: $100,000 (simulated)
Available Cash: $35,000 (35%)
Invested: $65,000 (65%)

Today's P&L: +$2,340 (+2.34%)
Week's P&L: +$5,780 (+5.78%)

Position limits are healthy. Risk parameters within tolerance."""

        else:
            return f"""Risk Analysis for: "{query}"

Current risk exposure is MODERATE.
Portfolio is well-positioned for current market conditions.

Key Metrics:
• Risk Score: 6.5/10
• Volatility: Medium
• Diversification: Good

All systems nominal. Standing by for position adjustments."""
    
    def _daedalus_response(self, query: str) -> str:
        """Simulated Daedalus (Strategy Simulation) response"""
        query_lower = query.lower()
        
        if 'backtest' in query_lower or 'simulate' in query_lower or 'strategy' in query_lower:
            return """Strategy Simulation Results:

🏛️ BACKTEST SUMMARY

Strategy: Mean Reversion + Momentum Hybrid
Period: Last 90 days
Initial Capital: $10,000

Performance Metrics:
• Total Return: +28.5%
• Sharpe Ratio: 2.1
• Max Drawdown: -8.2%
• Win Rate: 64%
• Profit Factor: 2.4

Trade Statistics:
• Total Trades: 47
• Winning Trades: 30
• Average Win: +4.2%
• Average Loss: -1.8%

Monte Carlo Analysis (1000 runs):
• 95% Confidence Interval: [+15%, +42%]
• Probability of Profit: 87%
• Risk of Ruin: <1%

Assessment: STRATEGY VIABLE
Recommendation: Consider live testing with reduced position size."""

        else:
            return f"""Strategy Analysis for: "{query}"

Running scenario simulations...

I can help you:
• Backtest trading strategies
• Run Monte Carlo simulations
• Optimize parameters
• Analyze risk scenarios

What aspect would you like to explore?"""
    
    def _hermes_response(self, query: str) -> str:
        """Simulated Hermes (Consensus) response"""
        query_lower = query.lower()
        
        if 'consensus' in query_lower or 'vote' in query_lower or 'decision' in query_lower:
            return """Multi-Agent Consensus Report:

🕊️ COLLECTIVE DECISION ANALYSIS

Query: Should we enter BTC long position?

Agent Votes:
• 🧭 ATHENA: BUY (Confidence: 82%)
  "Market structure supports upside"
  
• ⚡ APOLLO: BUY (Confidence: 78%)
  "Strong technical setup detected"
  
• ⏱️ CHRONOS: HOLD (Confidence: 65%)
  "Risk acceptable but monitor closely"
  
• 🏛️ DAEDALUS: BUY (Confidence: 71%)
  "Backtest supports this entry point"

CONSENSUS: BULLISH (75% confidence)
Recommendation: ENTER POSITION

Action Plan:
1. Entry: $62,000 - $62,500
2. Position Size: 3-5% of portfolio
3. Stop Loss: $60,000 (mandatory)
4. Monitor: Chronos risk alerts

Decision Quality: HIGH
All agents aligned on directional bias."""

        else:
            return f"""Consensus Analysis for: "{query}"

Collecting agent perspectives...

I facilitate multi-agent decision making by:
• Aggregating agent signals
• Resolving conflicts
• Building consensus
• Tracking decision quality

Waiting for all agents to weigh in..."""
    
    def get_conversation_history(self) -> list:
        """Get conversation history"""
        return self.conversation_history
