"""
Balance Monitor Tool for Chronos Agent

This tool enables the Chronos agent to monitor account balance and track changes,
providing important information for risk management decisions.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

class BalanceMonitor:
    """
    Balance monitoring tool for Chronos risk management agent.
    Tracks account balance, calculates changes over time, and helps
    enforce risk limits based on available capital.
    """
    
    def __init__(self, trading_client=None, initial_balance: float = 0.0):
        """
        Initialize the balance monitor.
        
        Parameters:
        -----------
        trading_client : TradingClient, optional
            Reference to the trading client for live balance updates
        initial_balance : float, optional
            Initial balance to start monitoring if trading_client is not provided
        """
        self.trading_client = trading_client
        self._current_balance = initial_balance
        self._starting_balance = initial_balance
        self.balance_history = []
        self.last_updated = datetime.now()
        
        # Configurable thresholds for alerts
        self.drawdown_alert_threshold = 0.1  # 10% drawdown triggers alert
        self.balance_change_alert_threshold = 0.05  # 5% change triggers alert
        
        # Tracking metrics
        self.peak_balance = initial_balance
        self.peak_date = datetime.now()
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_date = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    @property
    def current_balance(self) -> float:
        """Get the most up-to-date account balance"""
        self.refresh_balance()
        return self._current_balance
    
    def refresh_balance(self) -> float:
        """
        Refresh balance from trading client if available.
        
        Returns:
        --------
        float: Current account balance
        """
        if self.trading_client and hasattr(self.trading_client, 'balance_info'):
            balance = self.trading_client.balance_info.get('balance', self._current_balance)
            
            # Only record if balance changed
            if balance != self._current_balance:
                timestamp = datetime.now()
                self.balance_history.append({
                    'timestamp': timestamp,
                    'balance': balance,
                    'previous': self._current_balance,
                    'change': balance - self._current_balance,
                    'change_pct': (balance - self._current_balance) / self._current_balance if self._current_balance else 0
                })
                
                # Update tracking metrics
                self._current_balance = balance
                self.last_updated = timestamp
                
                if balance > self.peak_balance:
                    self.peak_balance = balance
                    self.peak_date = timestamp
                
                # Calculate drawdown from peak
                if self.peak_balance > 0:
                    current_drawdown = (self.peak_balance - balance) / self.peak_balance
                    self.current_drawdown = max(0, current_drawdown)
                    
                    if self.current_drawdown > self.max_drawdown:
                        self.max_drawdown = self.current_drawdown
                        self.max_drawdown_date = timestamp
        
        return self._current_balance
    
    def force_refresh(self, callback: Optional[Callable] = None) -> None:
        """
        Force a balance refresh from the trading client by making API call.
        
        Parameters:
        -----------
        callback : callable, optional
            Callback function to execute after refresh completes
        """
        if not self.trading_client:
            self.logger.warning("Cannot force refresh: No trading client connected")
            return
        
        def handle_balance_update(data):
            if data.get('balance'):
                balance = data['balance'].get('balance', self._current_balance)
                self._current_balance = balance
                self.last_updated = datetime.now()
                
                self.logger.info(f"Balance refreshed: {balance}")
                
                if callback:
                    callback(balance)
        
        self.trading_client.get_balance(callback=handle_balance_update)
    
    def set_manual_balance(self, balance: float) -> None:
        """
        Manually set the account balance when trading client is unavailable.
        
        Parameters:
        -----------
        balance : float
            The current account balance to set
        """
        timestamp = datetime.now()
        self.balance_history.append({
            'timestamp': timestamp,
            'balance': balance,
            'previous': self._current_balance,
            'change': balance - self._current_balance,
            'change_pct': (balance - self._current_balance) / self._current_balance if self._current_balance else 0
        })
        
        self._current_balance = balance
        self.last_updated = timestamp
        
        # Update peak if applicable
        if balance > self.peak_balance:
            self.peak_balance = balance
            self.peak_date = timestamp
    
    def get_session_change(self) -> Dict[str, Any]:
        """
        Calculate change in balance since the start of the session.
        
        Returns:
        --------
        dict: Session balance metrics
        """
        current = self.current_balance
        change = current - self._starting_balance
        pct_change = change / self._starting_balance if self._starting_balance else 0
        
        return {
            'starting_balance': self._starting_balance,
            'current_balance': current,
            'change': change,
            'change_pct': pct_change,
            'is_profit': change >= 0
        }
    
    def get_drawdown_metrics(self) -> Dict[str, Any]:
        """
        Calculate drawdown metrics from peak balance.
        
        Returns:
        --------
        dict: Drawdown metrics
        """
        return {
            'peak_balance': self.peak_balance,
            'peak_date': self.peak_date,
            'current_drawdown': self.current_drawdown,
            'current_drawdown_pct': self.current_drawdown * 100,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown * 100,
            'max_drawdown_date': self.max_drawdown_date
        }
    
    def estimate_max_position_size(self, risk_per_trade: float = 0.02) -> float:
        """
        Calculate the maximum recommended position size based on current balance.
        
        Parameters:
        -----------
        risk_per_trade : float
            Maximum risk per trade as a decimal (default 2%)
            
        Returns:
        --------
        float: Maximum recommended position size
        """
        return self.current_balance * risk_per_trade
    
    def check_risk_limits(self, proposed_trade_size: float, 
                         max_risk_pct: float = 0.02) -> Dict[str, Any]:
        """
        Check if a proposed trade exceeds risk management limits.
        
        Parameters:
        -----------
        proposed_trade_size : float
            The size of the proposed trade
        max_risk_pct : float
            Maximum risk percentage per trade (default 2%)
            
        Returns:
        --------
        dict: Risk assessment results
        """
        max_position = self.current_balance * max_risk_pct
        is_within_limits = proposed_trade_size <= max_position
        
        return {
            'current_balance': self.current_balance,
            'proposed_size': proposed_trade_size,
            'max_position_size': max_position,
            'is_within_limits': is_within_limits,
            'pct_of_balance': proposed_trade_size / self.current_balance if self.current_balance else 0,
            'recommended_size': max_position if not is_within_limits else proposed_trade_size
        }
    
    def get_balance_trend(self, periods: int = 5) -> str:
        """
        Analyze the recent balance trend.
        
        Parameters:
        -----------
        periods : int
            Number of most recent balance records to analyze
            
        Returns:
        --------
        str: Trend analysis ('increasing', 'decreasing', 'fluctuating', 'stable')
        """
        if len(self.balance_history) < 2:
            return 'stable'
        
        # Get the most recent records
        recent = self.balance_history[-min(periods, len(self.balance_history)):]
        
        # Calculate changes
        changes = [entry['change_pct'] for entry in recent]
        positive_changes = sum(1 for c in changes if c > 0.001)
        negative_changes = sum(1 for c in changes if c < -0.001)
        
        # Determine trend
        if positive_changes > len(changes) * 0.7:
            return 'increasing'
        elif negative_changes > len(changes) * 0.7:
            return 'decreasing'
        elif positive_changes + negative_changes > len(changes) * 0.6:
            return 'fluctuating'
        else:
            return 'stable'
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive balance status report.
        
        Returns:
        --------
        dict: Balance status report
        """
        session_change = self.get_session_change()
        drawdown = self.get_drawdown_metrics()
        trend = self.get_balance_trend()
        
        return {
            'current_balance': self._current_balance,
            'last_updated': self.last_updated,
            'session_change': session_change,
            'drawdown': drawdown,
            'trend': trend,
            'history_entries': len(self.balance_history)
        }
    
    def reset_session(self) -> None:
        """Reset the session tracking, setting current balance as the new starting balance"""
        self._starting_balance = self._current_balance