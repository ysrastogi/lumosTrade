"""
Response Formatter for LumosTrade Terminal
Formats agent responses for clean terminal display
"""

from typing import Dict, Any, Optional
from datetime import datetime


class Colors:
    """ANSI color codes for terminal styling"""
    # Accent colors
    AQUA = '\033[38;2;15;240;252m'      # #0FF0FC
    MAGENTA = '\033[38;2;255;0;128m'    # #FF0080
    
    # Standard colors
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    
    @staticmethod
    def gradient_text(text: str, color1: str, color2: str) -> str:
        """Create a simple gradient effect (alternating colors)"""
        result = []
        for i, char in enumerate(text):
            color = color1 if i % 2 == 0 else color2
            result.append(f"{color}{char}")
        result.append(Colors.RESET)
        return ''.join(result)


class ResponseFormatter:
    """Format agent responses for terminal display"""
    
    def __init__(self):
        self.agent_icons = {
            'athena': '🧭',
            'apollo': '⚡',
            'chronos': '⏱️',
            'daedalus': '🏛️',
            'hermes': '🕊️',
            'system': '⚙️'
        }
        
        self.agent_colors = {
            'athena': 'ATHENA',
            'apollo': 'APOLLO',
            'chronos': 'CHRONOS',
            'daedalus': 'DAEDALUS',
            'hermes': 'HERMES',
            'system': 'SYSTEM'
        }
    
    def format_response(self, agent: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Format agent response with header and content
        
        Args:
            agent: Agent name (lowercase)
            message: Response message
            metadata: Optional metadata to display
            
        Returns:
            Formatted response string
        """
        icon = self.agent_icons.get(agent.lower(), '🤖')
        agent_name = self.agent_colors.get(agent.lower(), agent.upper())
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding for agents
        agent_colors = {
            'athena': Colors.MAGENTA,
            'apollo': Colors.AQUA,
            'chronos': Colors.MAGENTA,
            'daedalus': Colors.AQUA,
            'hermes': Colors.MAGENTA,
            'system': Colors.CYAN
        }
        
        agent_color = agent_colors.get(agent.lower(), Colors.WHITE)
        
        # Build header with color
        header = f"\n{agent_color}{icon} {Colors.BOLD}{agent_name}{Colors.RESET} {Colors.GRAY}[{timestamp}]{Colors.RESET}"
        separator = f"{agent_color}{'─' * 60}{Colors.RESET}"
        
        # Build response
        lines = [
            header,
            separator,
            message,
            separator
        ]
        
        # Add metadata if present
        if metadata:
            lines.append(f"\n{Colors.DIM}[Metadata]{Colors.RESET}")
            for key, value in metadata.items():
                lines.append(f"{Colors.GRAY}  {key}: {Colors.CYAN}{value}{Colors.RESET}")
            lines.append(separator)
        
        return "\n".join(lines)
    
    def format_error(self, error_message: str) -> str:
        """Format error message"""
        return f"""
{Colors.RED}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}
{Colors.RED}║{Colors.RESET}                          {Colors.BOLD}{Colors.RED}ERROR{Colors.RESET}                                {Colors.RED}║{Colors.RESET}
{Colors.RED}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
{Colors.YELLOW}{error_message}{Colors.RESET}
"""
    
    def format_status(self, status_data: Dict[str, Any]) -> str:
        """Format system status display"""
        agents = status_data.get('agents', {})
        uptime = status_data.get('uptime', 'Unknown')
        
        lines = [
            f"\n{Colors.AQUA}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}",
            f"{Colors.AQUA}║{Colors.RESET}                    {Colors.BOLD}{Colors.MAGENTA}SYSTEM STATUS{Colors.RESET}                              {Colors.AQUA}║{Colors.RESET}",
            f"{Colors.AQUA}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}",
            f"\n{Colors.CYAN}Uptime:{Colors.RESET} {Colors.WHITE}{uptime}{Colors.RESET}",
            f"\n{Colors.BOLD}{Colors.AQUA}AGENT STATUS:{Colors.RESET}"
        ]
        
        agent_color_map = {
            'athena': Colors.MAGENTA,
            'apollo': Colors.AQUA,
            'chronos': Colors.MAGENTA,
            'daedalus': Colors.AQUA,
            'hermes': Colors.MAGENTA
        }
        
        for agent_name, agent_info in agents.items():
            icon = self.agent_icons.get(agent_name.lower(), '🤖')
            agent_color = agent_color_map.get(agent_name.lower(), Colors.WHITE)
            
            if agent_info.get('active'):
                status = f"{Colors.GREEN}✓ Active{Colors.RESET}"
            else:
                status = f"{Colors.RED}✗ Inactive{Colors.RESET}"
            
            lines.append(f"  {icon} {agent_color}{Colors.BOLD}{agent_name.upper():<12}{Colors.RESET} {status}")
            
            if agent_info.get('description'):
                lines.append(f"     {Colors.GRAY}└─ {agent_info['description']}{Colors.RESET}")
        
        lines.append(f"\n{Colors.AQUA}{'─' * 60}{Colors.RESET}")
        return "\n".join(lines)
    
    def format_welcome(self) -> str:
        """Format welcome banner with ASCII art and colors"""
        aqua = Colors.AQUA
        magenta = Colors.MAGENTA
        cyan = Colors.CYAN
        white = Colors.WHITE
        bold = Colors.BOLD
        reset = Colors.RESET
        
        banner = f"""
{aqua}╔══════════════════════════════════════════════════════════════════════════╗{reset}
{aqua}║{reset}                                                                          {aqua}║{reset}
{aqua}║{reset}   {magenta}██╗     ██╗   ██╗███╗   ███╗ ██████╗ ███████╗████████╗██████╗  █████╗ ██████╗ ███████╗{reset}   {aqua}║{reset}
{aqua}║{reset}   {magenta}██║     ██║   ██║████╗ ████║██╔═══██╗██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝{reset}   {aqua}║{reset}
{aqua}║{reset}   {aqua}██║     ██║   ██║██╔████╔██║██║   ██║███████╗   ██║   ██████╔╝███████║██║  ██║█████╗{reset}     {aqua}║{reset}
{aqua}║{reset}   {aqua}██║     ██║   ██║██║╚██╔╝██║██║   ██║╚════██║   ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝{reset}     {aqua}║{reset}
{aqua}║{reset}   {magenta}███████╗╚██████╔╝██║ ╚═╝ ██║╚██████╔╝███████║   ██║   ██║  ██║██║  ██║██████╔╝███████╗{reset}   {aqua}║{reset}
{aqua}║{reset}   {magenta}╚══════╝ ╚═════╝ ╚═╝     ╚═╝ ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝{reset}   {aqua}║{reset}
{aqua}║{reset}                                                                          {aqua}║{reset}
{aqua}║{reset}            {cyan}🧠 A Cognitive Trading Terminal — Where AI Agents{reset}             {aqua}║{reset}
{aqua}║{reset}                  {cyan}Observe, Reason, and Trade Together{reset}                    {aqua}║{reset}
{aqua}║{reset}                                                                          {aqua}║{reset}
{aqua}╚══════════════════════════════════════════════════════════════════════════╝{reset}

{white}Welcome to {bold}{magenta}LumosTrade Terminal{reset}{white}!{reset}
{white}Type {cyan}/help{reset} for available commands or start asking questions.

{bold}{aqua}Available Agents:{reset}
  {magenta}🧭 ATHENA{reset}   - Market Intelligence & Analysis
  {aqua}⚡ APOLLO{reset}   - Signal Generation  
  {magenta}⏱️  CHRONOS{reset}  - Risk Management
  {aqua}🏛️  DAEDALUS{reset} - Strategy Simulation
  {magenta}🕊️  HERMES{reset}   - Consensus Engine

{white}Type your query or command below:{reset}
"""
        return banner
    
    def format_exit(self) -> str:
        """Format exit message"""
        return f"""
{Colors.AQUA}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}
{Colors.AQUA}║{Colors.RESET}         {Colors.MAGENTA}Thank you for using LumosTrade Terminal!{Colors.RESET}              {Colors.AQUA}║{Colors.RESET}
{Colors.AQUA}║{Colors.RESET}                    {Colors.CYAN}Trading wisely! 🌟{Colors.RESET}                         {Colors.AQUA}║{Colors.RESET}
{Colors.AQUA}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
