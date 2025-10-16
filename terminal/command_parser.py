"""
Command Parser for LumosTrade Terminal
Handles special commands and user input parsing
"""

from typing import Tuple, Optional, Dict, Any
from enum import Enum


class Colors:
    """ANSI color codes for terminal styling"""
    AQUA = '\033[38;2;15;240;252m'      # #0FF0FC
    MAGENTA = '\033[38;2;255;0;128m'    # #FF0080
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class CommandType(Enum):
    """Types of commands supported by the terminal"""
    EXIT = "exit"
    HELP = "help"
    STATUS = "status"
    CHART = "chart"
    LIVE_CHART = "live_chart"
    AGENT_QUERY = "agent_query"
    AGENT_FLOW = "agent_flow"  # Interactive agent flow
    UNKNOWN = "unknown"


class CommandParser:
    """Parse user commands and determine routing"""
    
    def __init__(self):
        self.special_commands = {
            '/exit': CommandType.EXIT,
            '/quit': CommandType.EXIT,
            '/help': CommandType.HELP,
            '/h': CommandType.HELP,
            '/status': CommandType.STATUS,
            '/s': CommandType.STATUS,
            '/chart': CommandType.CHART,
            '/plot': CommandType.CHART,
            '/live': CommandType.LIVE_CHART,
            '/livechart': CommandType.LIVE_CHART,
        }
        
        # Agent flow commands
        self.flow_commands = {
            '/athena': ('athena', 'market_analysis'),
            '/athena-analysis': ('athena', 'market_analysis'),
            '/athena-regime': ('athena', 'regime_detection'),
            '/athena-patterns': ('athena', 'pattern_detection'),
            '/athena-memory': ('athena', 'memory_context'),
            '/athena-multi': ('athena', 'multi_symbol'),
            '/athena-insights': ('athena', 'insights'),
            '/apollo': ('apollo', 'signal_generation'),
            '/apollo-signal': ('apollo', 'signal_generation'),
            '/chronos': ('chronos', 'risk_assessment'),
            '/chronos-risk': ('chronos', 'risk_assessment'),
            '/chronos-size': ('chronos', 'position_sizing'),
            '/daedalus': ('daedalus', 'backtest'),
            '/daedalus-backtest': ('daedalus', 'backtest'),
            '/hermes': ('hermes', 'consensus'),
            '/hermes-consensus': ('hermes', 'consensus'),
        }
        
        self.agent_keywords = {
            'athena': ['athena', 'market', 'analysis', 'intelligence', 'trend', 'pattern', 'regime', 'memory', 'insights'],
            'apollo': ['apollo', 'signal', 'trading', 'opportunity'],
            'chronos': ['chronos', 'risk', 'position', 'balance', 'portfolio'],
            'daedalus': ['daedalus', 'simulate', 'backtest', 'strategy', 'scenario'],
            'hermes': ['hermes', 'consensus', 'vote', 'decision'],
        }
    
    def parse(self, user_input: str) -> Tuple[CommandType, Dict[str, Any]]:
        """
        Parse user input and return command type and metadata
        
        Args:
            user_input: Raw user input string
            
        Returns:
            Tuple of (CommandType, metadata dict)
        """
        if not user_input or not user_input.strip():
            return CommandType.UNKNOWN, {'error': 'Empty input'}
        
        user_input = user_input.strip()
        
        # Check for agent flow commands first
        for flow_cmd, (agent, flow_type) in self.flow_commands.items():
            if user_input.lower().startswith(flow_cmd):
                return CommandType.AGENT_FLOW, {
                    'agent': agent,
                    'flow_type': flow_type,
                    'command': user_input
                }
        
        # Check for live chart command with parameters
        if user_input.lower().startswith(('/live', '/livechart')):
            return self._parse_live_chart_command(user_input)
        
        # Check for chart command with parameters
        if user_input.lower().startswith(('/chart', '/plot')):
            return self._parse_chart_command(user_input)
        
        # Check for other special commands
        if user_input.lower() in self.special_commands:
            cmd_type = self.special_commands[user_input.lower()]
            return cmd_type, {'command': user_input}
        
        # Parse agent query
        detected_agent = self._detect_agent(user_input)
        
        return CommandType.AGENT_QUERY, {
            'query': user_input,
            'suggested_agent': detected_agent
        }
    
    def _detect_agent(self, query: str) -> Optional[str]:
        """
        Detect which agent the query is likely for based on keywords
        
        Args:
            query: User query string
            
        Returns:
            Agent name or None if unclear
        """
        query_lower = query.lower()
        
        # Check each agent's keywords
        agent_scores = {}
        for agent, keywords in self.agent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                agent_scores[agent] = score
        
        if not agent_scores:
            return None
        
        # Return agent with highest score
        return max(agent_scores, key=agent_scores.get)
    
    def _parse_chart_command(self, user_input: str) -> Tuple[CommandType, Dict[str, Any]]:
        """
        Parse chart command and extract parameters
        
        Args:
            user_input: Chart command string (e.g., '/chart BTCUSD 1m')
            
        Returns:
            Tuple of (CommandType.CHART, metadata dict)
        """
        parts = user_input.split()
        
        metadata = {
            'command': 'chart',
            'symbol': None,
            'interval': 60,  # Default interval in seconds (1m)
            'interval_str': '1m',  # Keep string for display
        }
        
        # Extract symbol (second parameter)
        if len(parts) >= 2:
            metadata['symbol'] = parts[1].upper()
        
        # Extract interval (third parameter)
        if len(parts) >= 3:
            interval_str = parts[2].lower()
            # Validate interval and convert to seconds
            interval_map = {
                '1m': 60,
                '2m': 120,
                '5m': 300,
                '15m': 900,
                '1h': 3600
            }
            
            if interval_str in interval_map:
                metadata['interval'] = interval_map[interval_str]
                metadata['interval_str'] = interval_str
            else:
                metadata['error'] = f"Invalid interval '{interval_str}'. Valid: {', '.join(interval_map.keys())}"
        
        return CommandType.CHART, metadata
    
    def _parse_live_chart_command(self, user_input: str) -> Tuple[CommandType, Dict[str, Any]]:
        """
        Parse live chart command and extract parameters
        
        Args:
            user_input: Live chart command string (e.g., '/live BTCUSD 1m 2.0')
            
        Returns:
            Tuple of (CommandType.LIVE_CHART, metadata dict)
        """
        parts = user_input.split()
        
        metadata = {
            'command': 'live_chart',
            'symbol': None,
            'symbols': None,  # For multi-symbol
            'interval': 60,  # Default interval in seconds (1m)
            'interval_str': '1m',
            'refresh_rate': 1.0,  # Default refresh rate in seconds
        }
        
        # Extract symbol(s) (second parameter)
        if len(parts) >= 2:
            symbol_input = parts[1].upper()
            # Check if multi-symbol (comma-separated)
            if ',' in symbol_input:
                metadata['symbols'] = [s.strip() for s in symbol_input.split(',')]
            else:
                metadata['symbol'] = symbol_input
        
        # Extract interval (third parameter)
        if len(parts) >= 3:
            interval_str = parts[2].lower()
            # Validate interval and convert to seconds
            interval_map = {
                '1m': 60,
                '2m': 120,
                '5m': 300,
                '15m': 900,
                '1h': 3600
            }
            
            if interval_str in interval_map:
                metadata['interval'] = interval_map[interval_str]
                metadata['interval_str'] = interval_str
            else:
                metadata['error'] = f"Invalid interval '{interval_str}'. Valid: {', '.join(interval_map.keys())}"
        
        # Extract refresh rate (fourth parameter)
        if len(parts) >= 4:
            try:
                refresh_rate = float(parts[3])
                if 0.1 <= refresh_rate <= 60:
                    metadata['refresh_rate'] = refresh_rate
                else:
                    metadata['error'] = "Refresh rate must be between 0.1 and 60 seconds"
            except ValueError:
                metadata['error'] = f"Invalid refresh rate '{parts[3]}'. Must be a number (e.g., 1.0, 2.5)"
        
        return CommandType.LIVE_CHART, metadata
    
    def get_help_text(self) -> str:
        """Return help text for available commands"""
        return f"""
{Colors.AQUA}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}
{Colors.AQUA}║{Colors.RESET}                   {Colors.BOLD}{Colors.MAGENTA}LUMOSTRADE TERMINAL HELP{Colors.RESET}                    {Colors.AQUA}║{Colors.RESET}
{Colors.AQUA}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.BOLD}{Colors.CYAN}SPECIAL COMMANDS:{Colors.RESET}
  {Colors.AQUA}/help, /h{Colors.RESET}      - Show this help message
  {Colors.AQUA}/status, /s{Colors.RESET}    - Show active agents and system status
  
  {Colors.AQUA}/chart, /plot{Colors.RESET}  - Display candlestick chart (snapshot)
                   {Colors.GRAY}Usage: /chart <SYMBOL> [interval]{Colors.RESET}
                   {Colors.GRAY}Example: /chart BTCUSD 1m{Colors.RESET}
                   {Colors.GRAY}Intervals: 1m, 2m, 5m, 15m, 1h (default: 1m){Colors.RESET}
  
  {Colors.AQUA}/live{Colors.RESET}          - Display LIVE auto-refreshing chart
                   {Colors.GRAY}Usage: /live <SYMBOL> [interval] [refresh_rate]{Colors.RESET}
                   {Colors.GRAY}Example: /live BTCUSD 1m 1.0{Colors.RESET}
                   {Colors.GRAY}Multi-symbol: /live BTCUSD,ETHUSD,SOLUSD 5m 2.0{Colors.RESET}
                   {Colors.GRAY}Refresh rate: 0.1-60 seconds (default: 1.0){Colors.RESET}
                   {Colors.GRAY}Press Ctrl+C to exit live mode{Colors.RESET}
  
  {Colors.AQUA}/exit, /quit{Colors.RESET}   - Exit the terminal

{Colors.BOLD}{Colors.CYAN}INTERACTIVE AGENT FLOWS:{Colors.RESET}
  {Colors.MAGENTA}/athena{Colors.RESET}            - Start Athena market analysis flow
  {Colors.MAGENTA}/athena-analysis{Colors.RESET}  - Comprehensive market analysis
  {Colors.MAGENTA}/athena-regime{Colors.RESET}    - Market regime detection
  
  {Colors.AQUA}/apollo{Colors.RESET}            - Start Apollo signal generation flow
  {Colors.AQUA}/apollo-signal{Colors.RESET}    - Generate trading signals
  
  {Colors.MAGENTA}/chronos{Colors.RESET}           - Start Chronos risk assessment flow
  {Colors.MAGENTA}/chronos-risk{Colors.RESET}     - Portfolio risk analysis
  {Colors.MAGENTA}/chronos-size{Colors.RESET}     - Position sizing calculator
  
  {Colors.AQUA}/daedalus{Colors.RESET}          - Start Daedalus backtesting flow
  {Colors.AQUA}/daedalus-backtest{Colors.RESET} - Run strategy backtest
  
  {Colors.MAGENTA}/hermes{Colors.RESET}           - Start Hermes consensus flow
  {Colors.MAGENTA}/hermes-consensus{Colors.RESET} - Multi-agent consensus decision

{Colors.BOLD}{Colors.CYAN}AVAILABLE AGENTS:{Colors.RESET}
  {Colors.MAGENTA}🧭 Athena{Colors.RESET}      - Market Intelligence & Analysis
                   {Colors.GRAY}Keywords: market, analysis, trend, intelligence{Colors.RESET}
                   
  {Colors.AQUA}⚡ Apollo{Colors.RESET}      - Signal Generation & Trading Opportunities
                   {Colors.GRAY}Keywords: signal, trading, opportunity{Colors.RESET}
                   
  {Colors.MAGENTA}⏱️  Chronos{Colors.RESET}     - Risk Management & Portfolio Balance
                   {Colors.GRAY}Keywords: risk, position, balance, portfolio{Colors.RESET}
                   
  {Colors.AQUA}🏛️  Daedalus{Colors.RESET}    - Strategy Simulation & Backtesting
                   {Colors.GRAY}Keywords: simulate, backtest, strategy, scenario{Colors.RESET}
                   
  {Colors.MAGENTA}🕊️  Hermes{Colors.RESET}      - Consensus & Decision Making
                   {Colors.GRAY}Keywords: consensus, vote, decision{Colors.RESET}

{Colors.BOLD}{Colors.CYAN}USAGE:{Colors.RESET}
  Simply type your question or command naturally. The system will
  route it to the appropriate agent based on context.
  
  {Colors.BOLD}Examples:{Colors.RESET}
    {Colors.GREEN}>{Colors.RESET} What's the market trend for BTC?
    {Colors.GREEN}>{Colors.RESET} Show me trading signals for ETH
    {Colors.GREEN}>{Colors.RESET} What's my current risk exposure?
    {Colors.GREEN}>{Colors.RESET} Run a backtest on my strategy
    {Colors.GREEN}>{Colors.RESET} /chart BTCUSD 5m
    {Colors.GREEN}>{Colors.RESET} /athena-analysis  {Colors.GRAY}# Interactive flow{Colors.RESET}
    
{Colors.AQUA}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
