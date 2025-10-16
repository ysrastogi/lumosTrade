"""
LumosTrade Terminal CLI
Main command-line interface for the multi-agent trading system
"""

import sys
from typing import Optional
from terminal.agent_manager import AgentManager
from terminal.orchestrator import AgentOrchestrator
from terminal.command_parser import CommandParser, CommandType
from terminal.formatter import ResponseFormatter
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal styling"""
    AQUA = '\033[38;2;15;240;252m'      # #0FF0FC
    MAGENTA = '\033[38;2;255;0;128m'    # #FF0080
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class LumosTerminal:
    """Main terminal interface for LumosTrade"""
    
    def __init__(self):
        """Initialize the terminal with all components"""
        self.agent_manager = AgentManager()
        self.orchestrator = AgentOrchestrator(self.agent_manager)
        self.parser = CommandParser()
        self.formatter = ResponseFormatter()
        self.running = False
        
        logger.info("LumosTrade Terminal initialized")
    
    def start(self):
        """Start the terminal loop"""
        self.running = True
        
        # Display welcome message
        print(self.formatter.format_welcome())
        
        # Main loop
        while self.running:
            try:
                # Get user input with colored prompt
                prompt = f"\n{Colors.MAGENTA}▶{Colors.RESET} "
                user_input = input(prompt)
                
                # Skip empty input
                if not user_input.strip():
                    continue
                
                # Parse command
                command_type, metadata = self.parser.parse(user_input)
                
                # Route to orchestrator
                agent_name, response, response_meta = self.orchestrator.route(command_type, metadata)
                
                # Handle special commands
                if response == 'EXIT_COMMAND':
                    self._handle_exit()
                    break
                
                elif response == 'HELP_COMMAND':
                    self._handle_help()
                    continue
                
                elif response == 'STATUS_COMMAND':
                    self._handle_status()
                    continue
                
                # Display response
                formatted_response = self.formatter.format_response(
                    agent=agent_name,
                    message=response,
                    metadata=response_meta
                )
                print(formatted_response)
                
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                print("\n")
                self._handle_exit()
                break
            
            except Exception as e:
                # Handle errors gracefully
                logger.error(f"Error in terminal loop: {e}", exc_info=True)
                error_msg = self.formatter.format_error(
                    f"An error occurred: {str(e)}\nPlease try again or type /help for assistance."
                )
                print(error_msg)
    
    def _handle_exit(self):
        """Handle exit command"""
        self.running = False
        print(self.formatter.format_exit())
        logger.info("Terminal session ended")
    
    def _handle_help(self):
        """Handle help command"""
        help_text = self.parser.get_help_text()
        print(help_text)
    
    def _handle_status(self):
        """Handle status command"""
        status_data = {
            'agents': self.agent_manager.get_all_agents(),
            'uptime': self.agent_manager.get_uptime()
        }
        status_display = self.formatter.format_status(status_data)
        print(status_display)
    
    def stop(self):
        """Stop the terminal"""
        self.running = False


def main():
    """Main entry point for the terminal"""
    terminal = LumosTerminal()
    try:
        terminal.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
