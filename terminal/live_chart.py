"""
Live Auto-Refreshing Chart for LumosTrade Terminal
Displays real-time candlestick charts with automatic updates
"""

import time
import sys
import os
from typing import Optional, Callable, List
from datetime import datetime
import threading
import logging

logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal styling"""
    AQUA = '\033[38;2;15;240;252m'
    MAGENTA = '\033[38;2;255;0;128m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    CLEAR_SCREEN = '\033[2J'
    MOVE_CURSOR_HOME = '\033[H'


class LiveChart:
    """
    Live auto-refreshing candlestick chart
    
    Features:
    - Real-time updates from cache
    - Configurable refresh rate
    - Clean terminal clearing
    - Graceful exit (Ctrl+C)
    - Statistics tracking
    """
    
    def __init__(self, symbol: str, interval: int = 60, refresh_rate: float = 1.0):
        """
        Initialize live chart
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            interval: Time interval in seconds (60=1m, 120=2m, 300=5m, 900=15m, 3600=1h)
            refresh_rate: Refresh rate in seconds (default: 1.0)
        """
        self.symbol = symbol.upper()
        self.interval = interval
        self.refresh_rate = refresh_rate
        self.is_running = False
        self.update_count = 0
        self.start_time = None
        self.last_update_time = None
        self.last_price = None
        
        # Interval display mapping
        self.interval_map = {
            60: '1m',
            120: '2m',
            300: '5m',
            900: '15m',
            3600: '1h'
        }
        self.interval_str = self.interval_map.get(interval, f"{interval}s")
    
    def start(self):
        """
        Start the live chart display
        
        This will:
        1. Clear the terminal
        2. Fetch and display chart
        3. Update at specified refresh rate
        4. Continue until user presses Ctrl+C
        """
        self.is_running = True
        self.start_time = datetime.now()
        
        print(f"{Colors.CLEAR_SCREEN}{Colors.MOVE_CURSOR_HOME}")
        print(f"{Colors.AQUA}{'═' * 80}{Colors.RESET}")
        print(f"{Colors.AQUA}{Colors.BOLD}📊 LIVE CHART MODE{Colors.RESET}")
        print(f"{Colors.AQUA}{'═' * 80}{Colors.RESET}")
        print(f"{Colors.WHITE}Symbol: {Colors.MAGENTA}{self.symbol}{Colors.RESET}")
        print(f"{Colors.WHITE}Interval: {Colors.AQUA}{self.interval_str}{Colors.RESET}")
        print(f"{Colors.WHITE}Refresh Rate: {Colors.AQUA}{self.refresh_rate}s{Colors.RESET}")
        print(f"{Colors.GRAY}Press Ctrl+C to exit...{Colors.RESET}")
        print(f"{Colors.AQUA}{'═' * 80}{Colors.RESET}\n")
        
        time.sleep(2)  # Brief pause to show info
        
        try:
            while self.is_running:
                self._update_display()
                time.sleep(self.refresh_rate)
                
        except KeyboardInterrupt:
            self._handle_exit()
    
    def stop(self):
        """Stop the live chart"""
        self.is_running = False
    
    def _update_display(self):
        """Update the chart display"""
        try:
            # Clear screen and move cursor to home
            print(f"{Colors.CLEAR_SCREEN}{Colors.MOVE_CURSOR_HOME}", end='')
            
            # Get current chart
            from terminal.chart import create_chart
            chart_output = create_chart(self.symbol, self.interval)
            
            # Display chart
            print(chart_output)
            
            # Display live statistics
            self._display_stats()
            
            # Update counters
            self.update_count += 1
            self.last_update_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating live chart: {e}", exc_info=True)
            print(f"{Colors.RED}❌ Error updating chart: {str(e)}{Colors.RESET}")
    
    def _display_stats(self):
        """Display live chart statistics"""
        if not self.start_time:
            return
        
        uptime = datetime.now() - self.start_time
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        
        last_update = "Never"
        if self.last_update_time:
            last_update = self.last_update_time.strftime("%H:%M:%S")
        
        print(f"\n{Colors.AQUA}{'─' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}📊 Live Chart Statistics{Colors.RESET}")
        print(f"{Colors.AQUA}{'─' * 80}{Colors.RESET}")
        print(f"{Colors.WHITE}Updates: {Colors.AQUA}{self.update_count}{Colors.RESET} | "
              f"{Colors.WHITE}Uptime: {Colors.AQUA}{uptime_str}{Colors.RESET} | "
              f"{Colors.WHITE}Last Update: {Colors.AQUA}{last_update}{Colors.RESET}")
        print(f"{Colors.GRAY}Refreshing every {self.refresh_rate}s... Press Ctrl+C to exit{Colors.RESET}")
        print(f"{Colors.AQUA}{'─' * 80}{Colors.RESET}")
    
    def _handle_exit(self):
        """Handle graceful exit"""
        print(f"\n\n{Colors.AQUA}{'═' * 80}{Colors.RESET}")
        print(f"{Colors.YELLOW}👋 Exiting Live Chart Mode{Colors.RESET}")
        print(f"{Colors.AQUA}{'═' * 80}{Colors.RESET}")
        
        if self.start_time:
            uptime = datetime.now() - self.start_time
            uptime_str = str(uptime).split('.')[0]
            
            print(f"{Colors.WHITE}Session Statistics:{Colors.RESET}")
            print(f"  • Total Updates: {Colors.AQUA}{self.update_count}{Colors.RESET}")
            print(f"  • Session Duration: {Colors.AQUA}{uptime_str}{Colors.RESET}")
            print(f"  • Symbol: {Colors.MAGENTA}{self.symbol}{Colors.RESET}")
            print(f"  • Interval: {Colors.AQUA}{self.interval_str}{Colors.RESET}")
        
        print(f"{Colors.AQUA}{'═' * 80}{Colors.RESET}\n")


class LiveMultiChart:
    """
    Live auto-refreshing multi-symbol comparison chart
    
    Similar to LiveChart but displays multiple symbols in a comparison view
    """
    
    def __init__(self, symbols: List[str], interval: int = 60, refresh_rate: float = 2.0):
        """
        Initialize live multi-symbol chart
        
        Args:
            symbols: List of trading symbols
            interval: Time interval in seconds
            refresh_rate: Refresh rate in seconds (default: 2.0 for multi-symbol)
        """
        self.symbols = [s.upper() for s in symbols]
        self.interval = interval
        self.refresh_rate = refresh_rate
        self.is_running = False
        self.update_count = 0
        self.start_time = None
        self.last_update_time = None
        
        # Interval display mapping
        self.interval_map = {
            60: '1m',
            120: '2m',
            300: '5m',
            900: '15m',
            3600: '1h'
        }
        self.interval_str = self.interval_map.get(interval, f"{interval}s")
    
    def start(self):
        """Start the live multi-symbol chart display"""
        self.is_running = True
        self.start_time = datetime.now()
        
        print(f"{Colors.CLEAR_SCREEN}{Colors.MOVE_CURSOR_HOME}")
        print(f"{Colors.AQUA}{'═' * 80}{Colors.RESET}")
        print(f"{Colors.AQUA}{Colors.BOLD}📊 LIVE MULTI-SYMBOL CHART MODE{Colors.RESET}")
        print(f"{Colors.AQUA}{'═' * 80}{Colors.RESET}")
        print(f"{Colors.WHITE}Symbols: {Colors.MAGENTA}{', '.join(self.symbols)}{Colors.RESET}")
        print(f"{Colors.WHITE}Interval: {Colors.AQUA}{self.interval_str}{Colors.RESET}")
        print(f"{Colors.WHITE}Refresh Rate: {Colors.AQUA}{self.refresh_rate}s{Colors.RESET}")
        print(f"{Colors.GRAY}Press Ctrl+C to exit...{Colors.RESET}")
        print(f"{Colors.AQUA}{'═' * 80}{Colors.RESET}\n")
        
        time.sleep(2)
        
        try:
            while self.is_running:
                self._update_display()
                time.sleep(self.refresh_rate)
                
        except KeyboardInterrupt:
            self._handle_exit()
    
    def stop(self):
        """Stop the live chart"""
        self.is_running = False
    
    def _update_display(self):
        """Update the multi-symbol chart display"""
        try:
            # Clear screen
            print(f"{Colors.CLEAR_SCREEN}{Colors.MOVE_CURSOR_HOME}", end='')
            
            # Get charts for each symbol
            from terminal.chart import create_chart, create_multi_symbol_chart
            
            # Use multi-symbol chart for comparison
            chart_output = create_multi_symbol_chart(self.symbols, self.interval)
            
            # Display chart
            print(chart_output)
            
            # Display individual charts (optional, can be toggled)
            # for symbol in self.symbols:
            #     chart = create_chart(symbol, self.interval)
            #     print(chart)
            #     print()
            
            # Display stats
            self._display_stats()
            
            # Update counters
            self.update_count += 1
            self.last_update_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating live multi-chart: {e}", exc_info=True)
            print(f"{Colors.RED}❌ Error updating chart: {str(e)}{Colors.RESET}")
    
    def _display_stats(self):
        """Display live chart statistics"""
        if not self.start_time:
            return
        
        uptime = datetime.now() - self.start_time
        uptime_str = str(uptime).split('.')[0]
        
        last_update = "Never"
        if self.last_update_time:
            last_update = self.last_update_time.strftime("%H:%M:%S")
        
        print(f"\n{Colors.AQUA}{'─' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}📊 Live Multi-Chart Statistics{Colors.RESET}")
        print(f"{Colors.AQUA}{'─' * 80}{Colors.RESET}")
        print(f"{Colors.WHITE}Updates: {Colors.AQUA}{self.update_count}{Colors.RESET} | "
              f"{Colors.WHITE}Uptime: {Colors.AQUA}{uptime_str}{Colors.RESET} | "
              f"{Colors.WHITE}Last Update: {Colors.AQUA}{last_update}{Colors.RESET}")
        print(f"{Colors.GRAY}Refreshing every {self.refresh_rate}s... Press Ctrl+C to exit{Colors.RESET}")
        print(f"{Colors.AQUA}{'─' * 80}{Colors.RESET}")
    
    def _handle_exit(self):
        """Handle graceful exit"""
        print(f"\n\n{Colors.AQUA}{'═' * 80}{Colors.RESET}")
        print(f"{Colors.YELLOW}👋 Exiting Live Multi-Chart Mode{Colors.RESET}")
        print(f"{Colors.AQUA}{'═' * 80}{Colors.RESET}")
        
        if self.start_time:
            uptime = datetime.now() - self.start_time
            uptime_str = str(uptime).split('.')[0]
            
            print(f"{Colors.WHITE}Session Statistics:{Colors.RESET}")
            print(f"  • Total Updates: {Colors.AQUA}{self.update_count}{Colors.RESET}")
            print(f"  • Session Duration: {Colors.AQUA}{uptime_str}{Colors.RESET}")
            print(f"  • Symbols: {Colors.MAGENTA}{', '.join(self.symbols)}{Colors.RESET}")
            print(f"  • Interval: {Colors.AQUA}{self.interval_str}{Colors.RESET}")
        
        print(f"{Colors.AQUA}{'═' * 80}{Colors.RESET}\n")


def start_live_chart(symbol: str, interval: int = 60, refresh_rate: float = 1.0):
    """
    Convenience function to start a live chart
    
    Args:
        symbol: Trading symbol
        interval: Time interval in seconds
        refresh_rate: Refresh rate in seconds
    """
    chart = LiveChart(symbol, interval, refresh_rate)
    chart.start()


def start_live_multi_chart(symbols: List[str], interval: int = 60, refresh_rate: float = 2.0):
    """
    Convenience function to start a live multi-symbol chart
    
    Args:
        symbols: List of trading symbols
        interval: Time interval in seconds
        refresh_rate: Refresh rate in seconds
    """
    chart = LiveMultiChart(symbols, interval, refresh_rate)
    chart.start()


if __name__ == "__main__":
    """CLI interface for testing live charts"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single symbol:  python live_chart.py <SYMBOL> [interval_seconds] [refresh_rate]")
        print("  Multi-symbol:   python live_chart.py <SYM1,SYM2,SYM3> [interval_seconds] [refresh_rate]")
        print()
        print("Examples:")
        print("  python live_chart.py BTCUSD")
        print("  python live_chart.py BTCUSD 300 2.0")
        print("  python live_chart.py BTCUSD,ETHUSD,SOLUSD 60 1.5")
        sys.exit(1)
    
    symbols_input = sys.argv[1]
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    refresh_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    
    # Check if multi-symbol
    if ',' in symbols_input:
        symbols = [s.strip() for s in symbols_input.split(',')]
        start_live_multi_chart(symbols, interval, refresh_rate)
    else:
        start_live_chart(symbols_input, interval, refresh_rate)
