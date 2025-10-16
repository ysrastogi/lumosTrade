"""
Terminal-based Candlestick Chart Visualizer for LumosTrade
Uses ASCII art for charting directly in the terminal
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
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


class CandlestickChart:
    """Terminal-based candlestick chart renderer"""
    
    def __init__(self):
        """Initialize the chart renderer"""
        self.supported_intervals = ["1m", "2m", "5m", "15m", "1h"]
        # Map intervals to seconds for cache lookup
        self.interval_to_seconds = {
            "1m": 60,
            "2m": 120,
            "5m": 300,
            "15m": 900,
            "1h": 3600
        }
        # Reverse map for display
        self.seconds_to_interval = {v: k for k, v in self.interval_to_seconds.items()}
        
    def render(self, symbol: str, market_data: Dict[str, Any], 
               interval: Any = "1m", width: int = 100, height: int = 20) -> str:
        """
        Render a candlestick chart for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            market_data: Market data from get_market_data()
            interval: Time interval (1m, 2m, 5m, 15m, 1h) or seconds (60, 120, 300, 900, 3600)
            width: Chart width in characters
            height: Chart height in characters
            
        Returns:
            Formatted chart string ready for terminal display
        """
        try:
            # Convert interval to string format if it's an integer (seconds)
            if isinstance(interval, int):
                interval_str = self.seconds_to_interval.get(interval, f"{interval}s")
            else:
                interval_str = interval
                
            # Validate interval
            if interval_str not in self.supported_intervals:
                return f"{Colors.RED}❌ Unsupported interval '{interval_str}'. Use: {', '.join(self.supported_intervals)}{Colors.RESET}"
            
            # Extract symbol data
            if "symbols" not in market_data or symbol not in market_data["symbols"]:
                return f"{Colors.RED}❌ Symbol '{symbol}' not found in market data{Colors.RESET}"
            
            symbol_data = market_data["symbols"][symbol]
            
            # Get OHLC data for the requested interval (check both formats)
            ohlc_data = None
            if "ohlc" in symbol_data:
                # Try both string and integer formats
                ohlc_data = symbol_data["ohlc"].get(interval_str) or symbol_data["ohlc"].get(interval)
            
            if not ohlc_data:
                return f"{Colors.YELLOW}⚠️  No {interval_str} OHLC data available for {symbol}{Colors.RESET}"
            
            # Create the chart
            chart_output = self._create_candlestick_chart(
                symbol=symbol,
                ohlc_data=ohlc_data,
                metrics=symbol_data,
                interval=interval_str,
                width=width,
                height=height
            )
            
            return chart_output
            
        except Exception as e:
            logger.error(f"Error rendering chart for {symbol}: {e}", exc_info=True)
            return f"{Colors.RED}❌ Failed to render chart: {str(e)}{Colors.RESET}"
    
    def _create_candlestick_chart(self, symbol: str, ohlc_data: Dict[str, Any],
                                  metrics: Dict[str, Any], interval: str,
                                  width: int, height: int) -> str:
        """
        Create the actual candlestick chart using plotext
        
        Args:
            symbol: Trading symbol
            ohlc_data: OHLC data dictionary
            metrics: Symbol metrics
            interval: Time interval
            width: Chart width
            height: Chart height
            
        Returns:
            Rendered chart as string
        """
        # Note: We're not using plotext's plotting features due to API complexity
        # Instead, we'll create a clean ASCII representation
        
        # Extract OHLC values
        open_price = float(ohlc_data.get('open', 0))
        high_price = float(ohlc_data.get('high', 0))
        low_price = float(ohlc_data.get('low', 0))
        close_price = float(ohlc_data.get('close', 0))
        volume = float(ohlc_data.get('volume', 0))
        
        # Determine candle color
        is_bullish = close_price >= open_price
        
        # Build comprehensive output
        output_lines = []
        
        # Header
        output_lines.append(f"\n{Colors.AQUA}{'═' * width}{Colors.RESET}")
        output_lines.append(f"{Colors.AQUA}{Colors.BOLD}📊 {symbol} - {interval.upper()} Chart{Colors.RESET}")
        output_lines.append(f"{Colors.AQUA}{'═' * width}{Colors.RESET}\n")
        
        # Price summary box
        price_change = close_price - open_price
        price_change_pct = (price_change / open_price * 100) if open_price > 0 else 0
        change_color = Colors.GREEN if price_change >= 0 else Colors.RED
        change_arrow = "▲" if price_change >= 0 else "▼"
        
        output_lines.append(f"{Colors.WHITE}┌─ Price Summary ─────────────────────────────────────┐{Colors.RESET}")
        output_lines.append(f"{Colors.WHITE}│{Colors.RESET} Open:   {Colors.AQUA}{open_price:>12,.2f}{Colors.RESET} {Colors.WHITE}│{Colors.RESET}")
        output_lines.append(f"{Colors.WHITE}│{Colors.RESET} High:   {Colors.GREEN}{high_price:>12,.2f}{Colors.RESET} {Colors.WHITE}│{Colors.RESET}")
        output_lines.append(f"{Colors.WHITE}│{Colors.RESET} Low:    {Colors.RED}{low_price:>12,.2f}{Colors.RESET} {Colors.WHITE}│{Colors.RESET}")
        output_lines.append(f"{Colors.WHITE}│{Colors.RESET} Close:  {change_color}{Colors.BOLD}{close_price:>12,.2f}{Colors.RESET} {Colors.WHITE}│{Colors.RESET}")
        output_lines.append(f"{Colors.WHITE}│{Colors.RESET} Change: {change_color}{change_arrow} {abs(price_change):>10,.2f} ({price_change_pct:+.2f}%){Colors.RESET} {Colors.WHITE}│{Colors.RESET}")
        output_lines.append(f"{Colors.WHITE}│{Colors.RESET} Volume: {Colors.YELLOW}{volume:>12,.0f}{Colors.RESET} {Colors.WHITE}│{Colors.RESET}")
        output_lines.append(f"{Colors.WHITE}└──────────────────────────────────────────────────────┘{Colors.RESET}\n")
        
        # Simplified ASCII candlestick representation
        output_lines.append(self._create_ascii_candle(
            open_price, high_price, low_price, close_price, width=50
        ))
        
        # Additional metrics if available
        if metrics:
            output_lines.append(f"\n{Colors.WHITE}┌─ Technical Indicators ───────────────────────────────┐{Colors.RESET}")
            
            # Directional bias
            bias = metrics.get('directional_bias', 'neutral')
            bias_color = Colors.GREEN if bias == 'bull' else (Colors.RED if bias == 'bear' else Colors.YELLOW)
            output_lines.append(f"{Colors.WHITE}│{Colors.RESET} Bias:        {bias_color}{bias.upper():>10}{Colors.RESET} {Colors.WHITE}│{Colors.RESET}")
            
            # Volatility
            volatility = metrics.get('volatility', 0)
            vol_level = "HIGH" if volatility > 2 else ("MEDIUM" if volatility > 1 else "LOW")
            vol_color = Colors.RED if volatility > 2 else (Colors.YELLOW if volatility > 1 else Colors.GREEN)
            output_lines.append(f"{Colors.WHITE}│{Colors.RESET} Volatility:  {vol_color}{vol_level:>10}{Colors.RESET} ({volatility:.2f}%) {Colors.WHITE}│{Colors.RESET}")
            
            # Price changes across timeframes
            if 'price_change_1m' in metrics:
                pc_1m = metrics['price_change_1m']
                pc_color = Colors.GREEN if pc_1m >= 0 else Colors.RED
                output_lines.append(f"{Colors.WHITE}│{Colors.RESET} Change 1m:   {pc_color}{pc_1m:>9.2f}%{Colors.RESET} {Colors.WHITE}│{Colors.RESET}")
            
            if 'price_change_5m' in metrics:
                pc_5m = metrics['price_change_5m']
                pc_color = Colors.GREEN if pc_5m >= 0 else Colors.RED
                output_lines.append(f"{Colors.WHITE}│{Colors.RESET} Change 5m:   {pc_color}{pc_5m:>9.2f}%{Colors.RESET} {Colors.WHITE}│{Colors.RESET}")
            
            if 'price_change_15m' in metrics:
                pc_15m = metrics['price_change_15m']
                pc_color = Colors.GREEN if pc_15m >= 0 else Colors.RED
                output_lines.append(f"{Colors.WHITE}│{Colors.RESET} Change 15m:  {pc_color}{pc_15m:>9.2f}%{Colors.RESET} {Colors.WHITE}│{Colors.RESET}")
            
            if 'price_change_1h' in metrics:
                pc_1h = metrics['price_change_1h']
                pc_color = Colors.GREEN if pc_1h >= 0 else Colors.RED
                output_lines.append(f"{Colors.WHITE}│{Colors.RESET} Change 1h:   {pc_color}{pc_1h:>9.2f}%{Colors.RESET} {Colors.WHITE}│{Colors.RESET}")
            
            output_lines.append(f"{Colors.WHITE}└──────────────────────────────────────────────────────┘{Colors.RESET}")
        
        # Footer with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_lines.append(f"\n{Colors.GRAY}Last updated: {timestamp}{Colors.RESET}")
        output_lines.append(f"{Colors.AQUA}{'═' * width}{Colors.RESET}\n")
        
        return "\n".join(output_lines)
    
    def _create_ascii_candle(self, open_price: float, high_price: float,
                            low_price: float, close_price: float, 
                            width: int = 50) -> str:
        """
        Create a simplified ASCII candlestick representation
        
        Args:
            open_price: Opening price
            high_price: High price
            low_price: Low price
            close_price: Closing price
            width: Width of the chart
            
        Returns:
            ASCII art candlestick
        """
        is_bullish = close_price >= open_price
        
        # Normalize prices to chart height (20 chars)
        chart_height = 15
        price_range = high_price - low_price
        if price_range == 0:
            price_range = 1
        
        def price_to_row(price):
            """Convert price to row position"""
            normalized = (high_price - price) / price_range
            return int(normalized * chart_height)
        
        # Get row positions
        high_row = 0
        low_row = chart_height
        open_row = price_to_row(open_price)
        close_row = price_to_row(close_price)
        
        # Ensure body bounds are correct
        body_top = min(open_row, close_row)
        body_bottom = max(open_row, close_row)
        
        # Build the ASCII chart
        lines = []
        color = Colors.GREEN if is_bullish else Colors.RED
        
        lines.append(f"{Colors.WHITE}┌─ ASCII Candle ───────────────────────────────────────┐{Colors.RESET}")
        
        for row in range(chart_height + 1):
            # Price label
            row_price = high_price - (row / chart_height) * price_range
            
            # Build the candle
            if row == high_row or row == low_row:
                # Wick endpoints
                candle = f"{color}    │{Colors.RESET}"
            elif row > high_row and row < body_top:
                # Upper wick
                candle = f"{color}    │{Colors.RESET}"
            elif row >= body_top and row <= body_bottom:
                # Body
                if is_bullish:
                    candle = f"{Colors.GREEN}  ┃━┃{Colors.RESET}"
                else:
                    candle = f"{Colors.RED}  ┃█┃{Colors.RESET}"
            elif row > body_bottom and row < low_row:
                # Lower wick
                candle = f"{color}    │{Colors.RESET}"
            else:
                candle = "      "
            
            # Add price label
            if row % 3 == 0:  # Show price every 3 rows
                lines.append(f"{Colors.WHITE}│{Colors.RESET} {row_price:>8,.2f} {candle}  {Colors.WHITE}│{Colors.RESET}")
            else:
                lines.append(f"{Colors.WHITE}│{Colors.RESET}          {candle}  {Colors.WHITE}│{Colors.RESET}")
        
        lines.append(f"{Colors.WHITE}└──────────────────────────────────────────────────────┘{Colors.RESET}")
        
        # Add legend
        legend = f"   {Colors.GREEN}█{Colors.RESET} Bullish  {Colors.RED}█{Colors.RESET} Bearish"
        lines.append(legend)
        
        return "\n".join(lines)
    
    def render_multi_symbol(self, symbols: List[str], market_data: Dict[str, Any],
                           interval: Any = 60) -> str:
        """
        Render a comparison view of multiple symbols
        
        Args:
            symbols: List of trading symbols
            market_data: Market data from get_market_data()
            interval: Time interval (string like "1m" or int like 60 for seconds)
            
        Returns:
            Formatted comparison view
        """
        # Convert interval to string format if it's an integer (seconds)
        if isinstance(interval, int):
            interval_str = self.seconds_to_interval.get(interval, f"{interval}s")
        else:
            interval_str = interval
            
        output_lines = []
        
        output_lines.append(f"\n{Colors.AQUA}{'═' * 80}{Colors.RESET}")
        output_lines.append(f"{Colors.AQUA}{Colors.BOLD}📊 Multi-Symbol View - {interval_str.upper()}{Colors.RESET}")
        output_lines.append(f"{Colors.AQUA}{'═' * 80}{Colors.RESET}\n")
        
        for symbol in symbols:
            if "symbols" not in market_data or symbol not in market_data["symbols"]:
                output_lines.append(f"{Colors.RED}❌ {symbol}: Not found{Colors.RESET}\n")
                continue
            
            symbol_data = market_data["symbols"][symbol]
            
            # Try to get OHLC data with both formats (string and integer)
            ohlc = None
            if "ohlc" in symbol_data:
                ohlc = symbol_data["ohlc"].get(interval_str) or symbol_data["ohlc"].get(interval)
            
            if not ohlc:
                output_lines.append(f"{Colors.YELLOW}⚠️  {symbol}: No {interval_str} data{Colors.RESET}\n")
                continue
                
            open_price = float(ohlc.get('open', 0))
            close_price = float(ohlc.get('close', 0))
            
            change = close_price - open_price
            change_pct = (change / open_price * 100) if open_price > 0 else 0
            
            color = Colors.GREEN if change >= 0 else Colors.RED
            arrow = "▲" if change >= 0 else "▼"
            
            output_lines.append(
                f"{Colors.WHITE}{symbol:>10}{Colors.RESET} │ "
                f"{color}{close_price:>10,.2f}{Colors.RESET} │ "
                f"{color}{arrow} {abs(change):>8,.2f} ({change_pct:+.2f}%){Colors.RESET}"
            )
        
        output_lines.append(f"\n{Colors.AQUA}{'═' * 80}{Colors.RESET}\n")
        
        return "\n".join(output_lines)


def create_chart(symbol: str, interval: int = 60) -> str:
    """
    Helper function to create a chart for a symbol
    
    Args:
        symbol: Trading symbol
        interval: Time interval in seconds (60=1m, 120=2m, 300=5m, 900=15m, 3600=1h)
        
    Returns:
        Rendered chart string
    """
    from src.data_layer.aggregator.fetch_data import get_market_data_from_cache
    from src.data_layer.aggregator.worker import InMemoryCache
    
    # Interval mapping for display
    interval_map = {
        60: '1m',
        120: '2m',
        300: '5m',
        900: '15m',
        3600: '1h'
    }
    interval_str = interval_map.get(interval, f"{interval}s")
    
    try:
        # Get cache instance
        cache = InMemoryCache.get_instance()
        
        # Fetch OHLC and tick data for the symbol (cache expects interval in seconds)
        ohlc_data = cache.get_ohlc(symbol, interval)
        tick_data = cache.get_tick(symbol)
        metrics = cache.get_metrics(symbol)
        
        # Check if we have OHLC data
        if not ohlc_data:
            return f"{Colors.YELLOW}⚠️  No {interval_str} OHLC data available for {symbol}{Colors.RESET}\n{Colors.GRAY}Tip: Make sure the market stream is running and the symbol is being tracked.{Colors.RESET}"
        
        # Build market data structure expected by chart renderer
        market_data = {
            "timestamp": datetime.now().isoformat(),
            "symbols": {
                symbol: {
                    "ohlc": {
                        interval: ohlc_data  # Store with integer key for cache compatibility
                    }
                }
            }
        }
        
        # Add metrics if available
        if metrics:
            market_data["symbols"][symbol].update({
                "directional_bias": metrics.directional_bias,
                "volatility": metrics.volatility,
                "price_change_1m": metrics.price_change_1m,
                "price_change_5m": metrics.price_change_5m,
                "price_change_15m": metrics.price_change_15m,
                "price_change_1h": metrics.price_change_1h,
                "last_price": metrics.last_price
            })
        
        # Add tick data if available
        if tick_data:
            market_data["symbols"][symbol]["last_tick"] = tick_data
        
        # Create and render chart (pass integer interval, render method will handle conversion)
        chart = CandlestickChart()
        return chart.render(symbol, market_data, interval)
        
    except Exception as e:
        logger.error(f"Error creating chart for {symbol}: {e}", exc_info=True)
        return f"{Colors.RED}❌ Failed to create chart: {str(e)}{Colors.RESET}"


def create_multi_symbol_chart(symbols: List[str], interval: int = 60) -> str:
    """
    Helper function to create a multi-symbol comparison chart
    
    Args:
        symbols: List of trading symbols
        interval: Time interval in seconds (60=1m, 120=2m, 300=5m, 900=15m, 3600=1h)
        
    Returns:
        Rendered comparison chart string
    """
    from src.data_layer.aggregator.worker import InMemoryCache
    
    # Interval mapping for display
    interval_map = {
        60: '1m',
        120: '2m',
        300: '5m',
        900: '15m',
        3600: '1h'
    }
    interval_str = interval_map.get(interval, f"{interval}s")
    
    try:
        # Get cache instance
        cache = InMemoryCache.get_instance()
        
        # Build market data structure for all symbols
        market_data = {
            "timestamp": datetime.now().isoformat(),
            "symbols": {}
        }
        
        for symbol in symbols:
            # Fetch using integer interval (seconds)
            ohlc_data = cache.get_ohlc(symbol, interval)
            if ohlc_data:
                market_data["symbols"][symbol] = {
                    "ohlc": {
                        interval: ohlc_data  # Store with integer key
                    }
                }
        
        # Create and render chart
        chart = CandlestickChart()
        return chart.render_multi_symbol(symbols, market_data, interval)
        
    except Exception as e:
        logger.error(f"Error creating multi-symbol chart: {e}", exc_info=True)
        return f"{Colors.RED}❌ Failed to create multi-symbol chart: {str(e)}{Colors.RESET}"
