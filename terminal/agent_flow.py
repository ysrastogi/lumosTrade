"""
Agent Interaction Flow Module
Provides interactive step-by-step user flows for each trading agent
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal styling"""
    AQUA = '\033[38;2;15;240;252m'      # #0FF0FC
    MAGENTA = '\033[38;2;255;0;128m'    # #FF0080
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class AgentUtils:
    """Utility functions for agent flows"""
    
    @staticmethod
    def parse_interval(interval_str: str) -> int:
        """Convert interval string to seconds"""
        mapping = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        return mapping.get(interval_str.lower(), 3600)
    
    @staticmethod
    def calculate_risk_reward(entry: float, stop: float, target: float) -> Tuple[float, float, float]:
        """Calculate risk, reward, and risk-reward ratio from price levels"""
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr_ratio = reward / risk if risk > 0 else 0
        return risk, reward, rr_ratio


class AgentFormatter:
    """Formatting utilities for agent outputs"""
    
    @staticmethod
    def format_signal(signal: Dict[str, Any], detailed: bool = False) -> str:
        """Format trading signal for display
        
        Args:
            signal: Signal data dictionary
            detailed: Whether to include detailed information
            
        Returns:
            Formatted signal string
        """
        output = []
        output.append(f"Symbol: {signal.get('symbol', 'N/A')}")
        output.append(f"Direction: {Colors.GREEN if signal.get('direction', '').upper() == 'BUY' else Colors.RED}{signal.get('direction', 'N/A').upper()}{Colors.RESET}")
        output.append(f"Entry Price: ${signal.get('entry_price', 0):,.2f}")
        output.append(f"Stop Loss: ${signal.get('stop_loss', 0):,.2f}")
        output.append(f"Take Profit: ${signal.get('take_profit', 0):,.2f}")
        output.append(f"Confidence: {signal.get('confidence', 0):.2%}")
        
        if detailed:
            # Add pattern info if available
            if 'pattern' in signal and signal['pattern']:
                output.append(f"Pattern: {signal.get('pattern', 'N/A')}")
                
            # Calculate and add R:R
            entry = signal.get('entry_price', 0)
            stop = signal.get('stop_loss', 0)
            target = signal.get('take_profit', 0)
            risk, reward, rr = AgentUtils.calculate_risk_reward(entry, stop, target)
            output.append(f"Risk/Reward: 1:{rr:.2f}")
            
            # Add reasoning if available
            if 'reasoning' in signal and signal['reasoning']:
                output.append(f"\nReasoning: {signal['reasoning']}")
            
            # Add invalidation criteria if available
            if 'invalidation_criteria' in signal:
                criteria = signal.get('invalidation_criteria', [])
                if criteria:
                    output.append("\nInvalidation Criteria:")
                    for i, criterion in enumerate(criteria, 1):
                        output.append(f"  {i}. {criterion}")
        
        return '\n'.join(output)
    
    @staticmethod
    def format_multiple_signals(signals: List[Dict[str, Any]], limit: int = 5) -> str:
        """Format multiple trading signals
        
        Args:
            signals: List of signal dictionaries
            limit: Maximum number of signals to display
            
        Returns:
            Formatted signals string
        """
        if not signals:
            return "No signals available"
        
        output = []
        for i, signal in enumerate(signals[:limit], 1):
            direction_color = Colors.GREEN if signal.get('direction', '').upper() == 'BUY' else Colors.RED
            output.append(f"{i}. {signal.get('symbol', 'N/A')} - "
                          f"{direction_color}{signal.get('direction', 'N/A').upper()}{Colors.RESET} "
                          f"@ ${signal.get('entry_price', 0):,.2f} "
                          f"(Confidence: {signal.get('confidence', 0):.2%})")
        
        if len(signals) > limit:
            output.append(f"\n...and {len(signals) - limit} more signals")
        
        return '\n'.join(output)
    
    @staticmethod
    def format_regime(result: Dict[str, Any]) -> str:
        """Format regime detection result
        
        Args:
            result: Regime detection data
            
        Returns:
            Formatted regime string
        """
        regime = result.get('regime', 'Unknown')
        confidence = result.get('confidence', 0)
        characteristics = result.get('characteristics', [])
        
        output = [f"Regime: {regime} (Confidence: {confidence:.2%})"]
        
        if characteristics:
            output.append("\nCharacteristics:")
            for i, char in enumerate(characteristics, 1):
                output.append(f"  {i}. {char}")
        
        return '\n'.join(output)
    
    @staticmethod
    def format_patterns(patterns: List[Dict[str, Any]]) -> str:
        """Format detected chart patterns
        
        Args:
            patterns: List of pattern dictionaries
            
        Returns:
            Formatted patterns string
        """
        if not patterns:
            return "No patterns detected"
        
        output = []
        for i, pattern in enumerate(patterns, 1):
            output.append(f"{i}. {pattern.get('name', 'Unknown')} - {pattern.get('type', '')} "
                         f"(Confidence: {pattern.get('confidence', 0):.2%})")
        
        return '\n'.join(output)
    
    @staticmethod
    def format_market_context(result: Dict[str, Any]) -> str:
        """Format market context information
        
        Args:
            result: Market context data
            
        Returns:
            Formatted context string
        """
        if 'error' in result:
            return f"Error: {result['error']}"
        
        regime = result.get('regime', {})
        regime_confidence = result.get('regime_confidence', 0)
        summary = result.get('summary', 'No summary available')
        
        output = []
        output.append(f"Symbol: {result.get('symbol', 'N/A')}")
        
        # Handle regime - could be string or dict
        if isinstance(regime, dict):
            regime_type = regime.get('regime', 'Unknown')
            confidence = regime.get('confidence', 0)
        else:
            regime_type = regime if regime else 'Unknown'
            confidence = regime_confidence
        
        output.append(f"Regime: {regime_type} (Confidence: {confidence:.2%})")
        output.append(f"\nSummary:\n{summary}")
        
        return '\n'.join(output)
    
    @staticmethod
    def format_risk_assessment(result: Dict[str, Any]) -> str:
        """Format risk assessment results"""
        output = []
        output.append(f"Total Exposure: {result.get('total_exposure', 0):.2%}")
        output.append(f"Maximum Drawdown: {result.get('max_drawdown', 0):.2%}")
        output.append(f"Value at Risk (95%): ${result.get('var_95', 0):,.2f}")
        
        return '\n'.join(output)
    
    @staticmethod
    def format_position_size(result: Dict[str, Any]) -> str:
        """Format position sizing recommendation"""
        output = []
        output.append(f"Recommended Position Size: {result.get('position_size', 0):.4f} units")
        output.append(f"Position Value: ${result.get('position_value', 0):,.2f}")
        output.append(f"Risk Amount: ${result.get('risk_amount', 0):,.2f}")
        output.append(f"Portfolio Allocation: {result.get('position_pct', 0):.2f}%")
        
        return '\n'.join(output)
    
    @staticmethod
    def format_backtest_results(result: Dict[str, Any]) -> str:
        """Format backtest results"""
        output = []
        output.append(f"Total Trades: {result.get('total_trades', 0)}")
        output.append(f"Winning Trades: {result.get('winning_trades', 0)}")
        output.append(f"Win Rate: {result.get('win_rate', 0):.2%}")
        output.append(f"Total Return: {result.get('total_return', 0):.2%}")
        output.append(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
        output.append(f"Max Drawdown: {result.get('max_drawdown', 0):.2%}")
        
        return '\n'.join(output)


class AgentFlow(ABC):
    """Base class for agent interaction flows"""
    
    def __init__(self, agent_name: str, agent_icon: str, agent_connector=None):
        """Initialize agent flow
        
        Args:
            agent_name: Display name of the agent
            agent_icon: Icon representing the agent (emoji)
            agent_connector: Optional connector to agent API
        """
        self.agent_name = agent_name
        self.agent_icon = agent_icon
        self.agent_connector = agent_connector
        self.session_data = {}
    
    def display_header(self, title: str) -> None:
        """Display a formatted header
        
        Args:
            title: Title to display in the header
        """
        border = "═" * 60
        print(f"\n{Colors.CYAN}{border}{Colors.RESET}")
        print(f"{Colors.BOLD}{self.agent_icon} {title}{Colors.RESET}")
        print(f"{Colors.CYAN}{border}{Colors.RESET}\n")
    
    def display_step(self, step_num: int, total_steps: int, description: str) -> None:
        """Display a step indicator
        
        Args:
            step_num: Current step number
            total_steps: Total number of steps
            description: Step description
        """
        print(f"{Colors.MAGENTA}[Step {step_num}/{total_steps}]{Colors.RESET} {description}")
    
    def get_input(self, prompt: str, default: Optional[str] = None) -> str:
        """Get user input with optional default
        
        Args:
            prompt: Input prompt text
            default: Default value if user enters nothing
            
        Returns:
            User input or default value
        """
        if default:
            full_prompt = f"{prompt} [{Colors.YELLOW}{default}{Colors.RESET}]: "
        else:
            full_prompt = f"{prompt}: "
        
        user_input = input(full_prompt).strip()
        return user_input if user_input else default
    
    def display_result(self, title: str, content: str) -> None:
        """Display formatted result
        
        Args:
            title: Result title
            content: Result content to display
        """
        print(f"\n{Colors.GREEN}✓ {title}{Colors.RESET}")
        print(f"{Colors.CYAN}{'─' * 60}{Colors.RESET}")
        print(content)
        print(f"{Colors.CYAN}{'─' * 60}{Colors.RESET}\n")
    
    def display_error(self, message: str) -> None:
        """Display error message
        
        Args:
            message: Error message to display
        """
        print(f"\n{Colors.RED}✗ Error:{Colors.RESET} {message}\n")
    
    def confirm_action(self, message: str) -> bool:
        """Get yes/no confirmation from user
        
        Args:
            message: Confirmation message
            
        Returns:
            True if user confirmed, False otherwise
        """
        response = input(f"{message} (y/n): ").strip().lower()
        return response in ['y', 'yes']
        
    async def execute_agent_action(
        self,
        action_fn: Callable[..., Any],
        error_msg: str = "An error occurred",
        demo_result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute an agent action with error handling and demo fallback
        
        Args:
            action_fn: Agent API function to call
            error_msg: Error message if action fails
            demo_result: Demo result to return if agent_connector not available
            **kwargs: Additional arguments to pass to the action function
            
        Returns:
            Action result or demo result
        """
        if self.agent_connector:
            try:
                return await action_fn(**kwargs)
            except Exception as e:
                logger.error(f"Error in {self.agent_name} action: {e}")
                self.display_error(f"{error_msg}: {str(e)}")
                return {"error": str(e)}
        else:
            # Demo mode
            if demo_result:
                return demo_result
            return {"demo": True}


class AthenaFlow(AgentFlow):
    """Interactive flow for Athena - Market Intelligence Agent"""
    
    def __init__(self, agent_connector=None):
        super().__init__("Athena", "🧭", agent_connector)
    
    async def run_market_analysis_flow(self) -> Dict[str, Any]:
        """Interactive flow for comprehensive market analysis"""
        self.display_header("ATHENA - Market Intelligence Analysis")
        
        total_steps = 4
        
        # Step 1: Get symbol
        self.display_step(1, total_steps, "Select Symbol to Analyze")
        symbol = self.get_input("Enter trading symbol (e.g., BTCUSD, ETHUSD)", "BTCUSD")
        
        # Step 2: Get timeframe
        self.display_step(2, total_steps, "Select Analysis Timeframe")
        print("Available intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d")
        interval_str = self.get_input("Enter interval", "1h")
        interval = AgentUtils.parse_interval(interval_str)
        
        # Step 3: Get history depth
        self.display_step(3, total_steps, "Select Historical Depth")
        count = int(self.get_input("Number of candles to analyze", "100"))
        
        # Step 4: Run analysis
        self.display_step(4, total_steps, "Running Market Analysis...")
        
        # Define the demo result for this flow
        demo_result = {
            "demo": True, 
            "symbol": symbol, 
            "interval": interval,
            "regime": {"regime": "TRENDING_UP", "confidence": 0.75},
            "summary": "Market showing strong bullish momentum with increasing volume.",
            "features": {"rsi": 64.5, "macd": 0.25, "volume_change": 0.15},
            "patterns": [
                {"name": "Bullish Flag", "type": "continuation", "confidence": 0.82}
            ]
        }
        
        # Use the base class method to execute the agent action
        if self.agent_connector:
            try:
                result = await self.agent_connector.analyze_market(symbol, interval, count)
                
                # Display results using our formatter
                self.display_result("Market Context Analysis", AgentFormatter.format_market_context(result))
                
                # Ask if user wants detailed breakdown
                if self.confirm_action("Would you like to see detailed technical indicators?"):
                    if 'features' in result:
                        self.display_result("Technical Indicators", self._format_indicators(result.get('features', {})))
                
                if self.confirm_action("Would you like to see pattern analysis?"):
                    if 'patterns' in result:
                        self.display_result("Pattern Detection", AgentFormatter.format_patterns(result.get('patterns', [])))
                
                return result
                
            except Exception as e:
                self.display_error(f"Analysis failed: {str(e)}")
                logger.error(f"Athena analysis error: {e}", exc_info=True)
                return {"error": str(e)}
        else:
            # Demo mode - display demo results
            self.display_result("Demo Analysis", f"[Demo Mode] Analysis for {symbol} at {interval_str} interval")
            self.display_result("Market Context", AgentFormatter.format_market_context(demo_result))
            
            return demo_result
    
    async def run_regime_detection_flow(self) -> Dict[str, Any]:
        """Interactive flow for market regime detection"""
        self.display_header("ATHENA - Market Regime Detection")
        
        total_steps = 3
        
        # Step 1: Get symbol
        self.display_step(1, total_steps, "Select Symbol")
        symbol = self.get_input("Enter trading symbol", "BTCUSD")
        
        # Step 2: Get timeframe
        self.display_step(2, total_steps, "Select Timeframe")
        interval_str = self.get_input("Enter interval (1m, 5m, 15m, 1h, 4h)", "1h")
        interval = AgentUtils.parse_interval(interval_str)
        
        # Step 3: Detect regime
        self.display_step(3, total_steps, "Detecting Market Regime...")
        
        # Define demo result
        demo_result = {
            "regime": "TRENDING_UP",
            "confidence": 0.75,
            "characteristics": ["Strong momentum", "Higher highs", "Healthy volume"]
        }
        
        # Use the execute_agent_action method for consistent error handling
        result = await self.execute_agent_action(
            action_fn=lambda: self.agent_connector.detect_regime(symbol, interval) if self.agent_connector else None,
            error_msg="Failed to detect market regime",
            demo_result=demo_result
        )
        
        # Display the result
        if 'error' not in result:
            self.display_result("Market Regime", AgentFormatter.format_regime(result))
            
        return result
    
    def _parse_interval(self, interval_str: str) -> int:
        """Convert interval string to seconds"""
        mapping = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        return mapping.get(interval_str.lower(), 3600)
    
    def _format_market_context(self, result: Dict) -> str:
        """Format market context for display"""
        if 'error' in result:
            return f"Error: {result['error']}"
        
        regime = result.get('regime', {})
        regime_confidence = result.get('regime_confidence', 0)
        summary = result.get('summary', 'No summary available')
        
        output = []
        output.append(f"Symbol: {result.get('symbol', 'N/A')}")
        
        # Handle regime - could be string or dict
        if isinstance(regime, dict):
            regime_type = regime.get('regime', 'Unknown')
            confidence = regime.get('confidence', 0)
        else:
            regime_type = regime if regime else 'Unknown'
            confidence = regime_confidence
        
        output.append(f"Regime: {regime_type} (Confidence: {confidence:.2%})")
        output.append(f"\nSummary:\n{summary}")
        
        return '\n'.join(output)
    
    def _format_indicators(self, features: Dict) -> str:
        """Format technical indicators"""
        if not features:
            return "No indicators available"
        
        output = []
        for key, value in features.items():
            if isinstance(value, (int, float)):
                output.append(f"{key}: {value:.4f}")
            else:
                output.append(f"{key}: {value}")
        
        return '\n'.join(output)
    
    def _format_patterns(self, patterns: List) -> str:
        """Format detected patterns"""
        if not patterns:
            return "No patterns detected"
        
        output = []
        for i, pattern in enumerate(patterns, 1):
            output.append(f"{i}. {pattern.get('name', 'Unknown')} - {pattern.get('type', '')} (Confidence: {pattern.get('confidence', 0):.2%})")
        
        return '\n'.join(output)
    
    def _format_regime(self, result: Dict) -> str:
        """Format regime detection result"""
        regime = result.get('regime', 'Unknown')
        confidence = result.get('confidence', 0)
        characteristics = result.get('characteristics', [])
        
        output = [f"Regime: {regime} (Confidence: {confidence:.2%})"]
        
        if characteristics:
            output.append("\nCharacteristics:")
            for char in characteristics:
                output.append(f"  • {char}")
        
        return '\n'.join(output)
    
    async def run_pattern_detection_flow(self) -> Dict[str, Any]:
        """Interactive flow for pattern detection"""
        self.display_header("ATHENA - Pattern Detection")
        
        total_steps = 3
        
        # Step 1: Get symbol
        self.display_step(1, total_steps, "Select Symbol")
        symbol = self.get_input("Enter trading symbol", "BTCUSD")
        
        # Step 2: Get timeframe
        self.display_step(2, total_steps, "Select Timeframe")
        interval_str = self.get_input("Enter interval (1m, 5m, 15m, 1h, 4h)", "1h")
        interval = self._parse_interval(interval_str)
        
        # Step 3: Detect patterns
        self.display_step(3, total_steps, "Detecting Chart Patterns...")
        
        if self.agent_connector:
            try:
                result = await self.agent_connector.detect_patterns(symbol, interval)
                
                # Display patterns
                patterns = result.get('patterns', [])
                bias = result.get('trading_bias', {})
                
                output = [f"Symbol: {symbol}"]
                output.append(f"Patterns Found: {len(patterns)}\n")
                
                if patterns:
                    output.append("Detected Patterns:")
                    for i, pattern in enumerate(patterns[:5], 1):  # Top 5
                        output.append(f"{i}. {pattern.get('name', 'Unknown')} - {pattern.get('type', '')} (Confidence: {pattern.get('confidence', 0):.2%})")
                else:
                    output.append("No significant patterns detected")
                
                if bias:
                    output.append(f"\nTrading Bias: {bias.get('bias', 'NEUTRAL')}")
                    output.append(f"Confidence: {bias.get('confidence', 0):.2%}")
                
                self.display_result("Pattern Analysis", '\n'.join(output))
                return result
                
            except Exception as e:
                self.display_error(f"Pattern detection failed: {str(e)}")
                return {"error": str(e)}
        else:
            # Demo mode
            demo_result = {
                "patterns": [
                    {"name": "Double Bottom", "type": "reversal", "confidence": 0.82},
                    {"name": "Ascending Triangle", "type": "continuation", "confidence": 0.75}
                ],
                "trading_bias": {"bias": "BULLISH", "confidence": 0.78}
            }
            self.display_result("Demo Patterns", self._format_patterns(demo_result.get('patterns', [])))
            return demo_result
    
    async def run_memory_context_flow(self) -> Dict[str, Any]:
        """Interactive flow for accessing memory context"""
        self.display_header("ATHENA - Memory Context Retrieval")
        
        total_steps = 2
        
        # Step 1: Get symbol filter (optional)
        self.display_step(1, total_steps, "Filter Options")
        use_filter = self.confirm_action("Would you like to filter by a specific symbol?")
        
        symbol = None
        if use_filter:
            symbol = self.get_input("Enter trading symbol", "BTCUSD")
        
        # Step 2: Retrieve memory context
        self.display_step(2, total_steps, "Retrieving Memory Context...")
        
        if self.agent_connector:
            try:
                result = await self.agent_connector.get_memory_context(symbol)
                
                if 'error' in result:
                    self.display_error(result['error'])
                    return result
                
                # Format and display memory context
                output = []
                
                if symbol:
                    output.append(f"Memory Context for: {symbol}")
                else:
                    output.append("Global Memory Context")
                
                # Display recent memories
                if 'recent_memories' in result:
                    memories = result['recent_memories']
                    output.append(f"\nRecent Memories: {len(memories)}")
                    for i, mem in enumerate(memories[:5], 1):
                        output.append(f"{i}. {mem.get('type', 'Unknown')}: {mem.get('summary', 'N/A')}")
                
                # Display insights
                if 'insights' in result:
                    insights = result['insights']
                    output.append(f"\nRecent Insights: {len(insights)}")
                    for i, insight in enumerate(insights[:3], 1):
                        output.append(f"{i}. {insight.get('type', 'Unknown')}: {insight.get('content', 'N/A')[:100]}...")
                
                self.display_result("Memory Context", '\n'.join(output))
                return result
                
            except Exception as e:
                self.display_error(f"Memory retrieval failed: {str(e)}")
                return {"error": str(e)}
        else:
            # Demo mode
            demo_result = {
                "recent_memories": [
                    {"type": "observation", "summary": "BTCUSD trending up with strong momentum"},
                    {"type": "pattern", "summary": "Bullish flag pattern detected"}
                ],
                "insights": [
                    {"type": "market_summary", "content": "Market showing strength across major pairs"}
                ]
            }
            self.display_result("Demo Memory", "Memory context (demo mode)")
            return demo_result
    
    async def run_multi_symbol_flow(self) -> Dict[str, Any]:
        """Interactive flow for multi-symbol analysis"""
        self.display_header("ATHENA - Multi-Symbol Analysis")
        
        total_steps = 3
        
        # Step 1: Get symbols
        self.display_step(1, total_steps, "Select Symbols to Analyze")
        symbols_input = self.get_input("Enter symbols separated by commas (e.g., BTCUSD,ETHUSD,SOLUSD)", "BTCUSD,ETHUSD")
        symbols = [s.strip() for s in symbols_input.split(',')]
        
        # Step 2: Get timeframe
        self.display_step(2, total_steps, "Select Timeframe")
        interval_str = self.get_input("Enter interval (1m, 5m, 15m, 1h, 4h)", "1h")
        interval = self._parse_interval(interval_str)
        
        # Step 3: Analyze
        self.display_step(3, total_steps, f"Analyzing {len(symbols)} symbols...")
        
        if self.agent_connector:
            try:
                result = await self.agent_connector.observe_multiple(symbols, interval)
                
                contexts = result.get('contexts', [])
                
                # Display summary
                output = [f"Analyzed {len(contexts)} symbols"]
                output.append(f"Sorted by confidence (highest opportunity first):\n")
                
                for i, ctx in enumerate(contexts[:10], 1):  # Top 10
                    symbol = ctx.get('symbol', 'N/A')
                    confidence = ctx.get('confidence', 0)
                    regime = ctx.get('regime', {})
                    regime_type = regime.get('regime', 'Unknown') if isinstance(regime, dict) else 'Unknown'
                    
                    output.append(f"{i}. {symbol}: {confidence:.2%} confidence - {regime_type}")
                
                self.display_result("Multi-Symbol Analysis", '\n'.join(output))
                
                # Ask if user wants details on top symbol
                if contexts and self.confirm_action("Would you like to see detailed analysis of the top symbol?"):
                    top_ctx = contexts[0]
                    self.display_result(f"Top Symbol: {top_ctx.get('symbol', 'N/A')}", 
                                      self._format_market_context(top_ctx))
                
                return result
                
            except Exception as e:
                self.display_error(f"Multi-symbol analysis failed: {str(e)}")
                return {"error": str(e)}
        else:
            # Demo mode
            demo_result = {
                "contexts": [
                    {"symbol": "BTCUSD", "confidence": 0.85, "regime": {"regime": "TRENDING_UP"}},
                    {"symbol": "ETHUSD", "confidence": 0.72, "regime": {"regime": "RANGING"}}
                ]
            }
            self.display_result("Demo Multi-Symbol", f"Analyzed {len(symbols)} symbols (demo mode)")
            return demo_result
    
    async def run_insights_flow(self) -> Dict[str, Any]:
        """Interactive flow for getting current market insights"""
        self.display_header("ATHENA - Current Market Insights")
        
        total_steps = 2
        
        # Step 1: Get number of insights
        self.display_step(1, total_steps, "Select Number of Insights")
        top_n = int(self.get_input("How many top insights to retrieve?", "3"))
        
        # Step 2: Retrieve insights
        self.display_step(2, total_steps, "Retrieving Market Insights...")
        
        if self.agent_connector:
            try:
                result = await self.agent_connector.get_current_insights(top_n)
                
                if result.get('status') == 'no_data':
                    self.display_error("No market data has been analyzed yet")
                    return result
                
                # Format insights
                output = []
                output.append(f"Status: {result.get('status', 'unknown')}")
                output.append(f"Symbols Analyzed: {result.get('symbols_analyzed', 0)}")
                
                opportunities = result.get('top_opportunities', [])
                if opportunities:
                    output.append(f"\nTop {len(opportunities)} Opportunities:\n")
                    for i, opp in enumerate(opportunities, 1):
                        symbol = opp.get('symbol', 'N/A')
                        confidence = opp.get('confidence', 0)
                        summary = opp.get('summary', 'No summary')[:100]
                        
                        output.append(f"{i}. {symbol} ({confidence:.2%} confidence)")
                        output.append(f"   {summary}...\n")
                
                self.display_result("Market Insights", '\n'.join(output))
                return result
                
            except Exception as e:
                self.display_error(f"Insights retrieval failed: {str(e)}")
                return {"error": str(e)}
        else:
            # Demo mode
            demo_result = {
                "status": "success",
                "symbols_analyzed": 5,
                "top_opportunities": [
                    {"symbol": "BTCUSD", "confidence": 0.85, "summary": "Strong bullish momentum"}
                ]
            }
            self.display_result("Demo Insights", "Market insights (demo mode)")
            return demo_result
    
    async def run_athena_apollo_integrated_flow(self, apollo_connector=None) -> Dict[str, Any]:
        """
        🔥 PRO MODE: Complete Athena → Apollo integrated pipeline
        This flow demonstrates the full power of cross-agent intelligence!
        """
        self.display_header("🚀 ATHENA + APOLLO - Complete Trading Intelligence Pipeline")
        
        total_steps = 6
        
        # Step 1: Get symbol
        self.display_step(1, total_steps, "Select Trading Pair")
        symbol = self.get_input("Enter trading symbol", "BTCUSD")
        
        # Step 2: Configure analysis
        self.display_step(2, total_steps, "Configure Market Analysis")
        print("Available intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d")
        interval_str = self.get_input("Enter interval", "1h")
        interval = self._parse_interval(interval_str)
        count = int(self.get_input("Number of candles to analyze", "100"))
        
        # Step 3: Athena market analysis
        self.display_step(3, total_steps, "🧭 ATHENA - Analyzing Market Intelligence...")
        
        if self.agent_connector:
            try:
                athena_result = await self.agent_connector.analyze_market(symbol, interval, count)
                
                if 'error' in athena_result:
                    self.display_error(f"Athena analysis failed: {athena_result['error']}")
                    return athena_result
                
                # Display Athena's comprehensive analysis
                regime = athena_result.get('regime', 'Unknown')
                regime_conf = athena_result.get('regime_confidence', 0)
                summary = athena_result.get('summary', 'No summary available')
                patterns = athena_result.get('patterns', [])
                
                athena_output = []
                athena_output.append(f"Market Regime: {regime}")
                athena_output.append(f"Confidence: {regime_conf:.2%}")
                athena_output.append(f"\nMarket Summary:\n{summary[:250]}...")
                
                if patterns:
                    athena_output.append(f"\nDetected Patterns ({len(patterns)}):")
                    for i, pattern in enumerate(patterns[:3], 1):
                        athena_output.append(f"  {i}. {pattern.get('name', 'Unknown')} ({pattern.get('confidence', 0):.2%})")
                
                self.display_result("🧭 Athena Market Intelligence", '\n'.join(athena_output))
                
                # Step 4: Memory integration
                self.display_step(4, total_steps, "💾 Storing Athena's insights in shared memory...")
                print(f"{Colors.GREEN}✓ Athena observations stored{Colors.RESET}")
                print(f"{Colors.CYAN}  → Cross-agent memory enabled{Colors.RESET}")
                print(f"{Colors.CYAN}  → Apollo can now access this intelligence{Colors.RESET}\n")
                
                # Step 5: Apollo signal generation
                self.display_step(5, total_steps, "⚡ APOLLO - Generating Trading Signals...")
                
                if apollo_connector:
                    try:
                        apollo_result = await apollo_connector.generate_signals_from_athena(symbol, athena_result)
                        
                        if 'error' not in apollo_result:
                            signals = apollo_result.get('signals', [])
                            
                            if signals:
                                apollo_output = []
                                apollo_output.append(f"Generated {len(signals)} trading signal(s)")
                                apollo_output.append(f"Based on {regime} market regime\n")
                                
                                # Show each signal
                                for i, signal in enumerate(signals, 1):
                                    direction = signal.get('direction', 'N/A')
                                    entry = signal.get('entry_price', 0)
                                    conf = signal.get('confidence', 0)
                                    pattern = signal.get('pattern', 'N/A')
                                    
                                    apollo_output.append(f"\nSignal #{i}:")
                                    apollo_output.append(f"  Direction: {direction}")
                                    apollo_output.append(f"  Entry: ${entry:,.2f}")
                                    apollo_output.append(f"  Confidence: {conf:.2%}")
                                    apollo_output.append(f"  Pattern: {pattern}")
                                    
                                    if 'reasoning' in signal and signal['reasoning']:
                                        reasoning_preview = signal['reasoning'][:150]
                                        apollo_output.append(f"  Reasoning: {reasoning_preview}...")
                                
                                self.display_result("⚡ Apollo Trading Signals", '\n'.join(apollo_output))
                                
                                # Step 6: Integration summary
                                self.display_step(6, total_steps, "Integration Summary")
                                
                                summary_output = []
                                summary_output.append(f"✅ Complete Analysis Pipeline Executed")
                                summary_output.append(f"\n🧭 Athena provided:")
                                summary_output.append(f"   • Market regime detection")
                                summary_output.append(f"   • Pattern recognition")
                                summary_output.append(f"   • Market intelligence")
                                summary_output.append(f"\n⚡ Apollo generated:")
                                summary_output.append(f"   • {len(signals)} actionable signal(s)")
                                summary_output.append(f"   • LLM-powered reasoning")
                                summary_output.append(f"   • Risk/reward analysis")
                                summary_output.append(f"\n💾 Memory System:")
                                summary_output.append(f"   • Cross-agent data sharing enabled")
                                summary_output.append(f"   • Historical context preserved")
                                summary_output.append(f"   • Future analysis enhanced")
                                
                                self.display_result("🚀 Integration Complete", '\n'.join(summary_output))
                                
                                # Offer detailed view
                                if self.confirm_action("View detailed signal analysis?"):
                                    for signal in signals:
                                        print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
                                        print(self._format_signal_detailed(signal))
                                
                                return {
                                    "athena_analysis": athena_result,
                                    "apollo_signals": apollo_result,
                                    "integration_success": True
                                }
                            else:
                                print(f"\n{Colors.YELLOW}⚠ Apollo: No signals generated - market conditions not favorable{Colors.RESET}\n")
                                return {
                                    "athena_analysis": athena_result,
                                    "apollo_signals": apollo_result,
                                    "integration_success": True,
                                    "note": "No signals - unfavorable conditions"
                                }
                        else:
                            self.display_error(f"Apollo signal generation failed: {apollo_result['error']}")
                            return {"error": apollo_result['error']}
                            
                    except Exception as e:
                        self.display_error(f"Apollo integration failed: {str(e)}")
                        return {"error": str(e)}
                else:
                    print(f"\n{Colors.YELLOW}⚠ Apollo connector not available - showing Athena analysis only{Colors.RESET}\n")
                    return {"athena_analysis": athena_result, "note": "Apollo not connected"}
                    
            except Exception as e:
                self.display_error(f"Athena analysis failed: {str(e)}")
                return {"error": str(e)}
        else:
            # Demo mode - show the full pipeline concept
            print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
            print(f"{Colors.BOLD}Demo: Athena-Apollo Integration Pipeline{Colors.RESET}")
            print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")
            
            print(f"{Colors.GREEN}✓ Athena Analysis:{Colors.RESET} Market regime detected (TRENDING_UP)")
            print(f"{Colors.GREEN}✓ Memory Storage:{Colors.RESET} Insights stored in shared memory")
            print(f"{Colors.GREEN}✓ Apollo Signals:{Colors.RESET} 2 signals generated based on Athena's analysis")
            print(f"{Colors.GREEN}✓ Integration:{Colors.RESET} Complete cross-agent intelligence pipeline\n")
            
            return {"demo": True, "symbol": symbol}
    
    def _format_signal_detailed(self, signal: Dict) -> str:
        """Format signal with full details"""
        output = []
        output.append(f"{Colors.BOLD}Trading Signal Details{Colors.RESET}")
        output.append(f"Symbol: {signal.get('symbol', 'N/A')}")
        output.append(f"Direction: {Colors.GREEN if signal.get('direction', '').upper() == 'BUY' else Colors.RED}{signal.get('direction', 'N/A').upper()}{Colors.RESET}")
        output.append(f"Entry Price: ${signal.get('entry_price', 0):,.2f}")
        output.append(f"Stop Loss: ${signal.get('stop_loss', 0):,.2f}")
        output.append(f"Take Profit: ${signal.get('take_profit', 0):,.2f}")
        output.append(f"Confidence: {signal.get('confidence', 0):.2%}")
        output.append(f"Pattern: {signal.get('pattern', 'N/A')}")
        
        # Calculate R:R
        entry = signal.get('entry_price', 0)
        stop = signal.get('stop_loss', 0)
        target = signal.get('take_profit', 0)
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = reward / risk if risk > 0 else 0
        output.append(f"Risk/Reward: 1:{rr:.2f}")
        
        if 'reasoning' in signal and signal['reasoning']:
            output.append(f"\n{Colors.CYAN}Reasoning:{Colors.RESET}")
            output.append(signal['reasoning'])
        
        return '\n'.join(output)


class ApolloFlow(AgentFlow):
    """Interactive flow for Apollo - Signal Generation Agent"""
    
    def __init__(self, agent_connector=None, athena_flow=None):
        super().__init__("Apollo", "⚡", agent_connector)
        self.athena_flow = athena_flow
    
    async def run_signal_generation_flow(self) -> Dict[str, Any]:
        """Interactive flow for generating trading signals (standalone mode)"""
        self.display_header("APOLLO - Signal Generation")
        
        total_steps = 4
        
        # Step 1: Get symbol
        self.display_step(1, total_steps, "Select Trading Pair")
        symbol = self.get_input("Enter trading symbol", "BTCUSD")
        
        # Step 2: Get market context (can integrate with Athena)
        self.display_step(2, total_steps, "Gather Market Context")
        print("Apollo will analyze current market conditions...")
        
        # Step 3: Signal type preference
        self.display_step(3, total_steps, "Signal Preferences")
        print("Signal types: 1) Breakout, 2) Mean Reversion, 3) Trend Following, 4) Any")
        signal_type = self.get_input("Select signal type (1-4)", "4")
        
        # Step 4: Generate signal
        self.display_step(4, total_steps, "Generating Trading Signal...")
        
        if self.agent_connector:
            try:
                result = await self.agent_connector.generate_signal(symbol, signal_type)
                self.display_result("Trading Signal", self._format_signal(result))
                
                # Ask for additional details
                if self.confirm_action("Would you like to see risk/reward analysis?"):
                    self.display_result("Risk Analysis", self._format_risk_reward(result))
                
                if self.confirm_action("Would you like to see invalidation criteria?"):
                    self.display_result("Invalidation Criteria", self._format_invalidation(result))
                
                return result
            except Exception as e:
                self.display_error(f"Signal generation failed: {str(e)}")
                return {"error": str(e)}
        else:
            # Demo signal
            demo_signal = {
                "symbol": symbol,
                "direction": "BUY",
                "entry_price": 45000.0,
                "stop_loss": 44500.0,
                "take_profit": 46500.0,
                "confidence": 0.78,
                "reasoning": "Strong bullish momentum with breakout confirmation"
            }
            self.display_result("Demo Signal", self._format_signal(demo_signal))
            return demo_signal
    
    async def run_athena_based_signal_flow(self) -> Dict[str, Any]:
        """
        Interactive flow for generating signals based on Athena's market analysis
        This is the PRO mode - full Athena-Apollo integration!
        """
        self.display_header("APOLLO + ATHENA - AI-Powered Signal Generation")
        
        total_steps = 5
        
        # Step 1: Get symbol
        self.display_step(1, total_steps, "Select Trading Pair")
        symbol = self.get_input("Enter trading symbol", "BTCUSD")
        
        # Step 2: Get Athena analysis parameters
        self.display_step(2, total_steps, "Configure Market Analysis")
        print("Available intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d")
        interval_str = self.get_input("Enter interval", "1h")
        interval = self._parse_interval(interval_str)
        count = int(self.get_input("Number of candles to analyze", "100"))
        
        # Step 3: Run Athena analysis
        self.display_step(3, total_steps, "🧭 Athena analyzing market conditions...")
        
        athena_context = None
        if self.agent_connector and hasattr(self.agent_connector, 'athena_connector') and self.agent_connector.athena_connector:
            try:
                athena_context = await self.agent_connector.athena_connector.analyze_market(symbol, interval, count)
                
                if 'error' not in athena_context:
                    # Display Athena's analysis
                    regime = athena_context.get('regime', 'Unknown')
                    regime_conf = athena_context.get('regime_confidence', 0)
                    summary = athena_context.get('summary', 'No summary available')
                    
                    athena_summary = f"Market Regime: {regime} (Confidence: {regime_conf:.2%})\n"
                    athena_summary += f"Analysis: {summary[:200]}..."
                    
                    self.display_result("Athena Market Intelligence", athena_summary)
                else:
                    self.display_error(f"Athena analysis failed: {athena_context['error']}")
                    return athena_context
                    
            except Exception as e:
                self.display_error(f"Failed to get Athena analysis: {str(e)}")
                return {"error": str(e)}
        else:
            # Demo Athena context
            athena_context = {
                "regime": "TRENDING_UP",
                "regime_confidence": 0.82,
                "summary": "Strong bullish momentum with healthy volume profile",
                "patterns": [{"name": "Ascending Triangle", "confidence": 0.75}]
            }
            self.display_result("Demo Athena Analysis", 
                              f"Regime: {athena_context['regime']} ({athena_context['regime_confidence']:.2%})")
        
        # Step 4: Generate Apollo signals
        self.display_step(4, total_steps, "⚡ Apollo generating signals from analysis...")
        
        if self.agent_connector:
            try:
                result = await self.agent_connector.generate_signals_from_athena(symbol, athena_context)
                
                if 'error' not in result:
                    signals = result.get('signals', [])
                    
                    if signals:
                        self.display_result(
                            f"Apollo Generated {len(signals)} Signal(s)",
                            self._format_multiple_signals(signals)
                        )
                        
                        # Show detailed view of top signal
                        if self.confirm_action("Would you like detailed analysis of the top signal?"):
                            top_signal = signals[0]
                            self.display_result("Top Signal Details", self._format_signal_detailed(top_signal))
                    else:
                        print(f"\n{Colors.YELLOW}⚠ No signals generated - market conditions not favorable{Colors.RESET}\n")
                else:
                    self.display_error(f"Signal generation failed: {result['error']}")
                
                return result
                
            except Exception as e:
                self.display_error(f"Apollo signal generation failed: {str(e)}")
                return {"error": str(e)}
        else:
            # Demo signals
            demo_result = {
                "symbol": symbol,
                "signals": [
                    {
                        "symbol": symbol,
                        "direction": "BUY",
                        "entry_price": 45000.0,
                        "stop_loss": 44500.0,
                        "take_profit": 46500.0,
                        "confidence": 0.85,
                        "pattern": "ascending_triangle_breakout",
                        "reasoning": "Athena detected strong bullish regime with ascending triangle pattern. Breakout confirmed with high volume."
                    }
                ],
                "count": 1
            }
            self.display_result("Demo Signals", self._format_multiple_signals(demo_result['signals']))
            return demo_result
        
        # Step 5: Ask about storage
        if self.confirm_action("Would you like to see how this signal performs against historical data?"):
            print(f"\n{Colors.CYAN}📊 Historical validation would be performed here...{Colors.RESET}\n")
    
    async def run_multi_signal_flow(self) -> Dict[str, Any]:
        """Interactive flow for generating signals across multiple symbols"""
        self.display_header("APOLLO - Multi-Symbol Signal Generation")
        
        total_steps = 3
        
        # Step 1: Get symbols
        self.display_step(1, total_steps, "Select Trading Pairs")
        symbols_input = self.get_input("Enter symbols (comma-separated)", "BTCUSD,ETHUSD,SOLUSD")
        symbols = [s.strip() for s in symbols_input.split(',')]
        
        # Step 2: Confirm
        self.display_step(2, total_steps, "Configuration")
        print(f"Generating signals for {len(symbols)} symbols: {', '.join(symbols)}")
        
        if not self.confirm_action("Proceed with multi-symbol analysis?"):
            return {"cancelled": True}
        
        # Step 3: Generate signals
        self.display_step(3, total_steps, f"Analyzing {len(symbols)} symbols...")
        
        if self.agent_connector:
            try:
                result = await self.agent_connector.batch_generate_signals(symbols)
                
                if 'error' not in result:
                    # Display summary
                    summary = f"Analyzed {result['total_symbols']} symbols\n"
                    summary += f"Generated {result['total_signals']} total signals\n\n"
                    
                    # Show signals per symbol
                    for symbol in symbols:
                        symbol_result = result['results'].get(symbol, {})
                        signal_count = symbol_result.get('count', 0)
                        summary += f"  {symbol}: {signal_count} signal(s)\n"
                    
                    self.display_result("Multi-Symbol Analysis Complete", summary)
                    
                    # Offer detailed view
                    if result['total_signals'] > 0 and self.confirm_action("Show detailed signals?"):
                        for symbol in symbols:
                            symbol_result = result['results'].get(symbol, {})
                            signals = symbol_result.get('signals', [])
                            if signals:
                                print(f"\n{Colors.BOLD}=== {symbol} ==={Colors.RESET}")
                                print(self._format_multiple_signals(signals))
                else:
                    self.display_error(f"Multi-signal generation failed: {result['error']}")
                
                return result
                
            except Exception as e:
                self.display_error(f"Multi-signal generation failed: {str(e)}")
                return {"error": str(e)}
        else:
            # Demo mode
            demo_result = {
                "total_symbols": len(symbols),
                "total_signals": len(symbols),
                "results": {symbol: {"count": 1, "signals": [{"symbol": symbol, "direction": "BUY", "confidence": 0.75}]} 
                           for symbol in symbols}
            }
            self.display_result("Demo Multi-Signal", f"Generated {len(symbols)} signals")
            return demo_result
    
    async def run_signal_validation_flow(self) -> Dict[str, Any]:
        """Interactive flow for validating a trading signal"""
        self.display_header("APOLLO - Signal Validation")
        
        total_steps = 3
        
        # Step 1: Get signal details
        self.display_step(1, total_steps, "Enter Signal Details")
        symbol = self.get_input("Symbol", "BTCUSD")
        direction = self.get_input("Direction (BUY/SELL)", "BUY")
        entry = float(self.get_input("Entry Price", "45000"))
        stop = float(self.get_input("Stop Loss", "44500"))
        target = float(self.get_input("Take Profit", "46500"))
        
        signal = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry,
            "stop_loss": stop,
            "take_profit": target
        }
        
        # Step 2: Display signal summary
        self.display_step(2, total_steps, "Signal Summary")
        print(self._format_signal(signal))
        
        # Step 3: Validate
        self.display_step(3, total_steps, "Validating against historical data...")
        
        if self.agent_connector:
            try:
                validation = await self.agent_connector.validate_signal(signal)
                
                if 'error' not in validation:
                    output = []
                    output.append(f"Valid: {'✓ Yes' if validation.get('valid') else '✗ No'}")
                    if 'historical_win_rate' in validation:
                        output.append(f"Historical Win Rate: {validation['historical_win_rate']:.2%}")
                    if 'sample_size' in validation:
                        output.append(f"Sample Size: {validation['sample_size']} trades")
                    if 'avg_reward_risk' in validation:
                        output.append(f"Avg R:R Ratio: 1:{validation['avg_reward_risk']:.2f}")
                    output.append(f"\n{validation.get('message', '')}")
                    
                    self.display_result("Validation Results", '\n'.join(output))
                else:
                    self.display_error(f"Validation failed: {validation['error']}")
                
                return validation
                
            except Exception as e:
                self.display_error(f"Validation failed: {str(e)}")
                return {"error": str(e)}
        else:
            # Demo validation
            demo_validation = {
                "valid": True,
                "historical_win_rate": 0.72,
                "sample_size": 30,
                "message": "Signal shows strong historical performance"
            }
            self.display_result("Demo Validation", "Signal validated successfully")
            return demo_validation
    
    async def run_memory_signals_flow(self) -> Dict[str, Any]:
        """Interactive flow for retrieving stored signals from memory"""
        self.display_header("APOLLO - Signal Memory Retrieval")
        
        total_steps = 2
        
        # Step 1: Get filter parameters
        self.display_step(1, total_steps, "Filter Configuration")
        filter_symbol = self.get_input("Filter by symbol (leave empty for all)", "")
        limit = int(self.get_input("Maximum signals to retrieve", "10"))
        
        # Step 2: Retrieve signals
        self.display_step(2, total_steps, "Retrieving signals from memory...")
        
        if self.agent_connector:
            try:
                result = await self.agent_connector.get_stored_signals(
                    symbol=filter_symbol if filter_symbol else None,
                    limit=limit
                )
                
                if 'error' not in result:
                    signals = result.get('signals', [])
                    
                    if signals:
                        self.display_result(
                            f"Retrieved {len(signals)} Signal(s)",
                            self._format_multiple_signals(signals)
                        )
                    else:
                        print(f"\n{Colors.YELLOW}No stored signals found{Colors.RESET}\n")
                else:
                    self.display_error(f"Retrieval failed: {result['error']}")
                
                return result
                
            except Exception as e:
                self.display_error(f"Memory retrieval failed: {str(e)}")
                return {"error": str(e)}
        else:
            # Demo mode
            print(f"\n{Colors.CYAN}Demo: Would retrieve stored signals from memory{Colors.RESET}\n")
            return {"signals": [], "count": 0}
    
    def _parse_interval(self, interval_str: str) -> int:
        """Convert interval string to seconds"""
        mapping = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        return mapping.get(interval_str.lower(), 3600)
    
    def _format_signal(self, signal: Dict) -> str:
        """Format trading signal for display"""
        output = []
        output.append(f"Symbol: {signal.get('symbol', 'N/A')}")
        output.append(f"Direction: {signal.get('direction', 'N/A')}")
        output.append(f"Entry Price: ${signal.get('entry_price', 0):,.2f}")
        output.append(f"Stop Loss: ${signal.get('stop_loss', 0):,.2f}")
        output.append(f"Take Profit: ${signal.get('take_profit', 0):,.2f}")
        output.append(f"Confidence: {signal.get('confidence', 0):.2%}")
        
        if 'reasoning' in signal and signal['reasoning']:
            output.append(f"\nReasoning:\n{signal['reasoning'][:300]}...")
        
        return '\n'.join(output)
    
    def _format_signal_detailed(self, signal: Dict) -> str:
        """Format signal with full details"""
        output = []
        output.append(f"Symbol: {signal.get('symbol', 'N/A')}")
        output.append(f"Direction: {signal.get('direction', 'N/A')}")
        output.append(f"Pattern: {signal.get('pattern', 'N/A')}")
        output.append(f"Entry Price: ${signal.get('entry_price', 0):,.2f}")
        output.append(f"Stop Loss: ${signal.get('stop_loss', 0):,.2f}")
        output.append(f"Take Profit: ${signal.get('take_profit', 0):,.2f}")
        output.append(f"Confidence: {signal.get('confidence', 0):.2%}")
        
        # Calculate R:R
        entry = signal.get('entry_price', 0)
        stop = signal.get('stop_loss', 0)
        target = signal.get('take_profit', 0)
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = reward / risk if risk > 0 else 0
        output.append(f"Risk/Reward: 1:{rr:.2f}")
        
        if 'reasoning' in signal and signal['reasoning']:
            output.append(f"\n📊 Reasoning:\n{signal['reasoning']}")
        
        if 'invalidation_criteria' in signal:
            criteria = signal['invalidation_criteria']
            if isinstance(criteria, list):
                output.append(f"\n🚫 Invalidation Criteria:")
                for criterion in criteria:
                    output.append(f"  • {criterion}")
        
        return '\n'.join(output)
    
    def _format_multiple_signals(self, signals: List[Dict]) -> str:
        """Format multiple signals for display"""
        if not signals:
            return "No signals available"
        
        output = []
        for i, signal in enumerate(signals[:5], 1):  # Show top 5
            direction = signal.get('direction', 'N/A')
            entry = signal.get('entry_price', 0)
            conf = signal.get('confidence', 0)
            pattern = signal.get('pattern', 'N/A')
            
            output.append(f"{i}. {direction} @ ${entry:,.2f} | Confidence: {conf:.2%} | Pattern: {pattern}")
        
        if len(signals) > 5:
            output.append(f"\n... and {len(signals) - 5} more signals")
        
        return '\n'.join(output)
    
    def _format_risk_reward(self, signal: Dict) -> str:
        """Format risk/reward analysis"""
        entry = signal.get('entry_price', 0)
        stop = signal.get('stop_loss', 0)
        target = signal.get('take_profit', 0)
        
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr_ratio = reward / risk if risk > 0 else 0
        
        output = []
        output.append(f"Risk: ${risk:,.2f} per unit")
        output.append(f"Reward: ${reward:,.2f} per unit")
        output.append(f"Risk/Reward Ratio: 1:{rr_ratio:.2f}")
        
        return '\n'.join(output)
    
    def _format_invalidation(self, signal: Dict) -> str:
        """Format invalidation criteria"""
        criteria = signal.get('invalidation_criteria', [
            "Price closes below stop loss",
            "Volume divergence occurs",
            "Market regime changes unexpectedly"
        ])
        
        output = ["Signal is invalidated if:"]
        for i, criterion in enumerate(criteria, 1):
            output.append(f"  {i}. {criterion}")
        
        return '\n'.join(output)


class ChronosFlow(AgentFlow):
    """Interactive flow for Chronos - Risk Management Agent"""
    
    def __init__(self, agent_connector=None):
        super().__init__("Chronos", "⏱️", agent_connector)
    
    async def run_risk_assessment_flow(self) -> Dict[str, Any]:
        """Interactive flow for portfolio risk assessment"""
        self.display_header("CHRONOS - Risk Assessment")
        
        total_steps = 3
        
        # Step 1: Get portfolio information
        self.display_step(1, total_steps, "Portfolio Information")
        print("Fetching current portfolio data...")
        
        # Step 2: Risk metrics selection
        self.display_step(2, total_steps, "Select Risk Metrics")
        print("Available metrics:")
        print("  1) Position Sizing Analysis")
        print("  2) Drawdown Risk Assessment")
        print("  3) Value at Risk (VaR)")
        print("  4) Correlation Analysis")
        print("  5) All Metrics")
        
        metric_choice = self.get_input("Select metrics (1-5)", "5")
        
        # Step 3: Run assessment
        self.display_step(3, total_steps, "Analyzing Risk Exposure...")
        
        if self.agent_connector:
            try:
                result = await self.agent_connector.assess_risk(metric_choice)
                self.display_result("Risk Assessment", self._format_risk_assessment(result))
                
                # Check for warnings
                if result.get('warnings'):
                    print(f"\n{Colors.YELLOW}⚠ Warnings:{Colors.RESET}")
                    for warning in result['warnings']:
                        print(f"  • {warning}")
                
                return result
            except Exception as e:
                self.display_error(f"Risk assessment failed: {str(e)}")
                return {"error": str(e)}
        else:
            # Demo risk assessment
            demo_result = {
                "total_exposure": 0.45,
                "max_drawdown": 0.12,
                "var_95": 1250.0,
                "warnings": ["Portfolio concentration above 40%", "Correlation risk detected"]
            }
            self.display_result("Demo Risk Assessment", self._format_risk_assessment(demo_result))
            return demo_result
    
    async def run_position_sizing_flow(self, signal: Dict) -> Dict[str, Any]:
        """Interactive flow for position sizing recommendation"""
        self.display_header("CHRONOS - Position Sizing")
        
        total_steps = 4
        
        # Step 1: Signal information
        self.display_step(1, total_steps, "Trading Signal")
        print(f"Symbol: {signal.get('symbol', 'N/A')}")
        print(f"Direction: {signal.get('direction', 'N/A')}")
        print(f"Entry: ${signal.get('entry_price', 0):,.2f}")
        print(f"Stop Loss: ${signal.get('stop_loss', 0):,.2f}")
        
        # Step 2: Account information
        self.display_step(2, total_steps, "Account Information")
        account_balance = float(self.get_input("Account balance ($)", "10000"))
        
        # Step 3: Risk tolerance
        self.display_step(3, total_steps, "Risk Tolerance")
        risk_per_trade = float(self.get_input("Risk per trade (%)", "2")) / 100
        
        # Step 4: Calculate position size
        self.display_step(4, total_steps, "Calculating Optimal Position Size...")
        
        if self.agent_connector:
            try:
                result = await self.agent_connector.calculate_position_size(
                    signal, account_balance, risk_per_trade
                )
                self.display_result("Position Sizing", self._format_position_size(result))
                return result
            except Exception as e:
                self.display_error(f"Position sizing failed: {str(e)}")
                return {"error": str(e)}
        else:
            # Demo calculation
            entry = signal.get('entry_price', 0)
            stop = signal.get('stop_loss', 0)
            risk_amount = account_balance * risk_per_trade
            risk_per_unit = abs(entry - stop)
            position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
            
            demo_result = {
                "position_size": position_size,
                "position_value": position_size * entry,
                "risk_amount": risk_amount,
                "position_pct": (position_size * entry) / account_balance * 100
            }
            self.display_result("Demo Position Sizing", self._format_position_size(demo_result))
            return demo_result
    
    def _format_risk_assessment(self, result: Dict) -> str:
        """Format risk assessment results"""
        output = []
        output.append(f"Total Exposure: {result.get('total_exposure', 0):.2%}")
        output.append(f"Maximum Drawdown: {result.get('max_drawdown', 0):.2%}")
        output.append(f"Value at Risk (95%): ${result.get('var_95', 0):,.2f}")
        
        return '\n'.join(output)
    
    def _format_position_size(self, result: Dict) -> str:
        """Format position sizing recommendation"""
        output = []
        output.append(f"Recommended Position Size: {result.get('position_size', 0):.4f} units")
        output.append(f"Position Value: ${result.get('position_value', 0):,.2f}")
        output.append(f"Risk Amount: ${result.get('risk_amount', 0):,.2f}")
        output.append(f"Portfolio Allocation: {result.get('position_pct', 0):.2f}%")
        
        return '\n'.join(output)


class DaedalusFlow(AgentFlow):
    """Interactive flow for Daedalus - Strategy Simulation Agent"""
    
    def __init__(self, agent_connector=None):
        super().__init__("Daedalus", "🏛️", agent_connector)
    
    async def run_backtest_flow(self) -> Dict[str, Any]:
        """Interactive flow for strategy backtesting"""
        self.display_header("DAEDALUS - Strategy Backtesting")
        
        total_steps = 5
        
        # Step 1: Strategy selection
        self.display_step(1, total_steps, "Select Strategy")
        print("Available strategies:")
        print("  1) Mean Reversion")
        print("  2) Trend Following")
        print("  3) Breakout")
        print("  4) Custom Strategy")
        
        strategy = self.get_input("Select strategy (1-4)", "2")
        
        # Step 2: Symbol and timeframe
        self.display_step(2, total_steps, "Trading Pair and Timeframe")
        symbol = self.get_input("Enter symbol", "BTCUSD")
        timeframe = self.get_input("Enter timeframe (1h, 4h, 1d)", "1h")
        
        # Step 3: Backtest period
        self.display_step(3, total_steps, "Backtest Period")
        lookback_days = int(self.get_input("Number of days to backtest", "30"))
        
        # Step 4: Initial capital
        self.display_step(4, total_steps, "Initial Capital")
        capital = float(self.get_input("Starting capital ($)", "10000"))
        
        # Step 5: Run backtest
        self.display_step(5, total_steps, "Running Backtest Simulation...")
        
        if self.agent_connector:
            try:
                result = await self.agent_connector.run_backtest(
                    strategy, symbol, timeframe, lookback_days, capital
                )
                
                if "error" in result:
                    self.display_error(f"Backtest failed: {result['error']}")
                    return result
                
                self.display_result("Backtest Results", self._format_backtest_results(result))
                return result
            except Exception as e:
                self.display_error(f"Backtest failed: {str(e)}")
                return {"error": str(e)}
        else:
            self.display_error("Agent connector not initialized")
            return {"error": "Agent connector not available"}
    
    async def run_optimization_flow(self) -> Dict[str, Any]:
        """Interactive flow for strategy parameter optimization"""
        self.display_header("DAEDALUS - Strategy Optimization")
        
        total_steps = 5
        
        # Step 1: Strategy selection
        self.display_step(1, total_steps, "Select Strategy to Optimize")
        print("Available strategies:")
        print("  1) Mean Reversion")
        print("  2) Trend Following")
        print("  3) Breakout")
        
        strategy = self.get_input("Select strategy (1-3)", "2")
        
        # Step 2: Symbol and timeframe
        self.display_step(2, total_steps, "Trading Pair and Timeframe")
        symbol = self.get_input("Enter symbol", "BTCUSD")
        timeframe = self.get_input("Enter timeframe (1h, 4h, 1d)", "1h")
        
        # Step 3: Optimization period
        self.display_step(3, total_steps, "Optimization Period")
        lookback_days = int(self.get_input("Number of days for optimization", "90"))
        
        # Step 4: Optimization method
        self.display_step(4, total_steps, "Optimization Method")
        print("Available methods:")
        print("  1) Grid Search")
        print("  2) Random Search")
        print("  3) Genetic Algorithm")
        
        method_choice = self.get_input("Select method (1-3)", "3")
        method_map = {"1": "grid", "2": "random", "3": "genetic"}
        method = method_map.get(method_choice, "genetic")
        
        # Define parameter space based on strategy
        param_space = self._get_param_space(strategy)
        
        # Step 5: Run optimization
        self.display_step(5, total_steps, f"Running {method} optimization...")
        
        if self.agent_connector:
            try:
                result = await self.agent_connector.optimize_strategy(
                    strategy, symbol, timeframe, lookback_days, param_space, method
                )
                
                if "error" in result:
                    self.display_error(f"Optimization failed: {result['error']}")
                    return result
                
                self.display_result("Optimization Results", self._format_optimization_results(result))
                return result
            except Exception as e:
                self.display_error(f"Optimization failed: {str(e)}")
                return {"error": str(e)}
        else:
            self.display_error("Agent connector not initialized")
            return {"error": "Agent connector not available"}
    
    async def run_monte_carlo_flow(self) -> Dict[str, Any]:
        """Interactive flow for Monte Carlo simulation"""
        self.display_header("DAEDALUS - Monte Carlo Forecast")
        
        total_steps = 3
        
        # Step 1: Get past simulations
        self.display_step(1, total_steps, "Select Strategy Result")
        
        if self.agent_connector:
            try:
                past_results = await self.agent_connector.get_past_simulations(limit=10)
                
                if "error" in past_results:
                    self.display_error(f"Failed to fetch past results: {past_results['error']}")
                    return past_results
                
                if not past_results.get("results"):
                    self.display_error("No past simulation results found. Run a backtest first.")
                    return {"error": "No past results available"}
                
                print("\nAvailable strategy results:")
                for i, result in enumerate(past_results["results"], 1):
                    print(f"  {i}) {result['strategy_name']} - Sharpe: {result['sharpe_ratio']:.2f}, Return: {result['total_return']:.2%}")
                
                choice = int(self.get_input(f"Select result (1-{len(past_results['results'])})", "1"))
                selected_result = past_results["results"][choice - 1]
                strategy_id = selected_result["strategy_id"]
                
                # Step 2: Forecast parameters
                self.display_step(2, total_steps, "Forecast Parameters")
                n_days = int(self.get_input("Forecast period (days)", "252"))
                n_paths = int(self.get_input("Number of Monte Carlo paths", "10000"))
                
                # Step 3: Run Monte Carlo
                self.display_step(3, total_steps, "Running Monte Carlo Simulation...")
                
                forecast = await self.agent_connector.monte_carlo_forecast(
                    strategy_id, n_days, n_paths
                )
                
                if "error" in forecast:
                    self.display_error(f"Monte Carlo forecast failed: {forecast['error']}")
                    return forecast
                
                self.display_result("Monte Carlo Forecast", self._format_monte_carlo_results(forecast))
                return forecast
                
            except Exception as e:
                self.display_error(f"Monte Carlo forecast failed: {str(e)}")
                return {"error": str(e)}
        else:
            self.display_error("Agent connector not initialized")
            return {"error": "Agent connector not available"}
    
    async def run_past_results_flow(self) -> Dict[str, Any]:
        """Interactive flow for viewing past simulation results"""
        self.display_header("DAEDALUS - Past Simulation Results")
        
        total_steps = 2
        
        # Step 1: Sort preference
        self.display_step(1, total_steps, "Sort Preference")
        print("Sort by:")
        print("  1) Sharpe Ratio")
        print("  2) Total Return")
        print("  3) Win Rate")
        
        sort_choice = self.get_input("Select sort (1-3)", "1")
        sort_map = {"1": "sharpe", "2": "profit", "3": "win_rate"}
        sort_by = sort_map.get(sort_choice, "sharpe")
        
        limit = int(self.get_input("Number of results to show", "10"))
        
        # Step 2: Fetch results
        self.display_step(2, total_steps, "Fetching past results...")
        
        if self.agent_connector:
            try:
                results = await self.agent_connector.get_past_simulations(limit, sort_by)
                
                if "error" in results:
                    self.display_error(f"Failed to fetch results: {results['error']}")
                    return results
                
                self.display_result("Past Simulation Results", self._format_past_results(results))
                return results
            except Exception as e:
                self.display_error(f"Failed to fetch results: {str(e)}")
                return {"error": str(e)}
        else:
            self.display_error("Agent connector not initialized")
            return {"error": "Agent connector not available"}
    
    def _get_param_space(self, strategy: str) -> Dict[str, tuple]:
        """Get parameter space for optimization based on strategy type"""
        param_spaces = {
            "1": {  # Mean Reversion
                "rsi_period": (10, 20),
                "upper_threshold": (65, 80),
                "lower_threshold": (20, 35)
            },
            "2": {  # Trend Following
                "fast": (5, 20),
                "slow": (30, 100)
            },
            "3": {  # Breakout
                "lookback": (10, 30),
                "atr_period": (10, 20)
            }
        }
        return param_spaces.get(strategy, param_spaces["2"])
    
    def _format_backtest_results(self, result: Dict) -> str:
        """Format backtest results"""
        output = []
        output.append(f"Strategy: {result.get('strategy', 'N/A')}")
        output.append(f"Strategy ID: {result.get('strategy_id', 'N/A')}")
        output.append("")
        output.append("=== Performance Metrics ===")
        output.append(f"Total Trades: {result.get('total_trades', 0)}")
        output.append(f"Winning Trades: {result.get('winning_trades', 0)}")
        output.append(f"Losing Trades: {result.get('losing_trades', 0)}")
        output.append(f"Win Rate: {result.get('win_rate', 0):.2%}")
        output.append("")
        output.append(f"Total Return: {result.get('total_return', 0):.2%}")
        output.append(f"Annual Return: {result.get('annual_return', 0):.2%}")
        output.append(f"Annual Volatility: {result.get('annual_volatility', 0):.2%}")
        output.append("")
        output.append("=== Risk Metrics ===")
        output.append(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
        output.append(f"Sortino Ratio: {result.get('sortino_ratio', 0):.2f}")
        output.append(f"Calmar Ratio: {result.get('calmar_ratio', 0):.2f}")
        output.append(f"Max Drawdown: {result.get('max_drawdown', 0):.2%}")
        output.append("")
        output.append("=== Trade Metrics ===")
        output.append(f"Profit Factor: {result.get('profit_factor', 0):.2f}")
        output.append(f"Avg Trade Duration: {result.get('avg_trade_duration', 0)}")
        output.append(f"Consistency Score: {result.get('consistency_score', 0):.2f}")
        
        return '\n'.join(output)
    
    def _format_optimization_results(self, result: Dict) -> str:
        """Format optimization results"""
        output = []
        output.append(f"Task ID: {result.get('task_id', 'N/A')}")
        output.append(f"Status: {result.get('status', 'N/A')}")
        output.append(f"Method: {result.get('method', 'N/A')}")
        output.append(f"Optimization Metric: {result.get('metric', 'N/A')}")
        output.append("")
        output.append(f"Best Score: {result.get('best_score', 0):.4f}")
        output.append("")
        output.append("=== Best Parameters ===")
        
        best_params = result.get('best_params', {})
        for param, value in best_params.items():
            output.append(f"  {param}: {value}")
        
        return '\n'.join(output)
    
    def _format_monte_carlo_results(self, result: Dict) -> str:
        """Format Monte Carlo forecast results"""
        output = []
        output.append(f"Forecast Period: {result.get('n_days', 0)} days")
        output.append(f"Monte Carlo Paths: {result.get('n_paths', 0):,}")
        output.append("")
        output.append("=== Forecast Statistics ===")
        output.append(f"Expected Value: ${result.get('expected_value', 0):,.2f}")
        output.append(f"Median Value: ${result.get('median_value', 0):,.2f}")
        output.append("")
        output.append("=== Confidence Intervals ===")
        output.append(f"5th Percentile: ${result.get('percentile_5', 0):,.2f}")
        output.append(f"95th Percentile: ${result.get('percentile_95', 0):,.2f}")
        output.append("")
        output.append("=== Risk Metrics ===")
        output.append(f"Value at Risk (95%): ${result.get('var_95', 0):,.2f}")
        output.append(f"Conditional VaR (95%): ${result.get('cvar_95', 0):,.2f}")
        output.append(f"Probability of Profit: {result.get('prob_profit', 0):.2%}")
        
        return '\n'.join(output)
    
    def _format_past_results(self, results: Dict) -> str:
        """Format past simulation results"""
        output = []
        output.append(f"Total Results: {results.get('count', 0)}")
        output.append(f"Sorted By: {results.get('sort_by', 'N/A')}")
        output.append("")
        
        for i, result in enumerate(results.get('results', []), 1):
            output.append(f"{i}. {result.get('strategy_name', 'N/A')}")
            output.append(f"   Strategy ID: {result.get('strategy_id', 'N/A')}")
            output.append(f"   Total Return: {result.get('total_return', 0):.2%}")
            output.append(f"   Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
            output.append(f"   Max Drawdown: {result.get('max_drawdown', 0):.2%}")
            output.append(f"   Win Rate: {result.get('win_rate', 0):.2%}")
            output.append(f"   Total Trades: {result.get('total_trades', 0)}")
            output.append(f"   Profit Factor: {result.get('profit_factor', 0):.2f}")
            output.append("")
        
        return '\n'.join(output)


class HermesFlow(AgentFlow):
    """Interactive flow for Hermes - Consensus Engine Agent"""
    
    def __init__(self, agent_connector=None):
        super().__init__("Hermes", "🕊️", agent_connector)
    
    async def run_consensus_flow(self) -> Dict[str, Any]:
        """Interactive flow for multi-agent consensus"""
        self.display_header("HERMES - Consensus Engine")
        
        total_steps = 3
        
        # Step 1: Trading decision context
        self.display_step(1, total_steps, "Decision Context")
        symbol = self.get_input("Enter symbol for decision", "BTCUSD")
        
        # Step 2: Gather agent inputs
        self.display_step(2, total_steps, "Gathering Agent Opinions")
        print("Hermes is consulting all agents...")
        print("  • Athena: Analyzing market intelligence...")
        print("  • Apollo: Generating signal consensus...")
        print("  • Chronos: Assessing risk factors...")
        print("  • Daedalus: Validating against historical patterns...")
        
        # Step 3: Generate consensus
        self.display_step(3, total_steps, "Generating Consensus Decision...")
        
        if self.agent_connector:
            try:
                result = await self.agent_connector.generate_consensus(symbol)
                self.display_result("Consensus Decision", self._format_consensus(result))
                return result
            except Exception as e:
                self.display_error(f"Consensus generation failed: {str(e)}")
                return {"error": str(e)}
        else:
            # Demo consensus
            demo_result = {
                "decision": "BUY",
                "confidence": 0.72,
                "agent_votes": {
                    "athena": {"vote": "BUY", "confidence": 0.8},
                    "apollo": {"vote": "BUY", "confidence": 0.75},
                    "chronos": {"vote": "HOLD", "confidence": 0.6},
                    "daedalus": {"vote": "BUY", "confidence": 0.73}
                }
            }
            self.display_result("Demo Consensus", self._format_consensus(demo_result))
            return demo_result
    
    def _format_consensus(self, result: Dict) -> str:
        """Format consensus decision"""
        output = []
        output.append(f"Decision: {result.get('decision', 'N/A')}")
        output.append(f"Confidence: {result.get('confidence', 0):.2%}")
        output.append("\nAgent Votes:")
        
        agent_votes = result.get('agent_votes', {})
        for agent, vote_data in agent_votes.items():
            vote = vote_data.get('vote', 'N/A')
            conf = vote_data.get('confidence', 0)
            output.append(f"  • {agent.capitalize()}: {vote} ({conf:.2%})")
        
        return '\n'.join(output)


class AgentFlowManager:
    """Manages all agent flows"""
    
    def __init__(self, connector_manager=None):
        """
        Initialize flow manager
        
        Args:
            connector_manager: Optional AgentConnectorManager instance
        """
        self.flows = {
            'athena': AthenaFlow(),
            'apollo': ApolloFlow(),
            'chronos': ChronosFlow(),
            'daedalus': DaedalusFlow(),
            'hermes': HermesFlow()
        }
        
        # Set connectors if provided
        if connector_manager:
            self.set_connectors(connector_manager)
    
    def set_connectors(self, connector_manager):
        """
        Set agent connectors for all flows
        
        Args:
            connector_manager: AgentConnectorManager instance
        """
        self.flows['athena'].agent_connector = connector_manager.get_connector('athena')
        self.flows['apollo'].agent_connector = connector_manager.get_connector('apollo')
        self.flows['chronos'].agent_connector = connector_manager.get_connector('chronos')
        self.flows['daedalus'].agent_connector = connector_manager.get_connector('daedalus')
        self.flows['hermes'].agent_connector = connector_manager.get_connector('hermes')
    
    def get_flow(self, agent_name: str) -> Optional[AgentFlow]:
        """Get flow handler for an agent"""
        return self.flows.get(agent_name.lower())
    
    async def run_flow(self, agent_name: str, flow_type: str, **kwargs) -> Dict[str, Any]:
        """Run a specific flow for an agent"""
        flow = self.get_flow(agent_name)
        if not flow:
            return {"error": f"No flow found for agent: {agent_name}"}
        
        # Map flow types to method names (strings)
        flow_method_names = {
            'athena': {
                'market_analysis': 'run_market_analysis_flow',
                'regime_detection': 'run_regime_detection_flow',
                'pattern_detection': 'run_pattern_detection_flow',
                'memory_context': 'run_memory_context_flow',
                'multi_symbol': 'run_multi_symbol_flow',
                'insights': 'run_insights_flow',
                'athena_apollo_integrated': 'run_athena_apollo_integrated_flow'  # New integrated flow!
            },
            'apollo': {
                'signal_generation': 'run_signal_generation_flow',
                'athena_based_signal': 'run_athena_based_signal_flow',  # New: signals from Athena
                'multi_signal': 'run_multi_signal_flow',  # New: batch signals
                'signal_validation': 'run_signal_validation_flow',  # New: validate signals
                'memory_signals': 'run_memory_signals_flow'  # New: retrieve stored signals
            },
            'chronos': {
                'risk_assessment': 'run_risk_assessment_flow',
                'position_sizing': 'run_position_sizing_flow'
            },
            'daedalus': {
                'backtest': 'run_backtest_flow',
                'optimize': 'run_optimization_flow',
                'monte_carlo': 'run_monte_carlo_flow',
                'past_results': 'run_past_results_flow'
            },
            'hermes': {
                'consensus': 'run_consensus_flow'
            }
        }
        
        agent_methods = flow_method_names.get(agent_name.lower(), {})
        method_name = agent_methods.get(flow_type)
        
        if method_name and hasattr(flow, method_name):
            method = getattr(flow, method_name)
            return await method(**kwargs)
        else:
            return {"error": f"Flow type '{flow_type}' not found for {agent_name}"}
