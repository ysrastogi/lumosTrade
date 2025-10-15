import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import csv
from pathlib import Path

class TradeLogger:
    """
    Comprehensive trade logging and performance tracking system
    Handles logging of trades, performance metrics, and generates reports
    """
    
    def __init__(self, log_dir: str = "logs", config: Dict = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.config = config or {}
        self.log_retention_days = self.config.get('log_retention_days', 31)  # Default to keep 1 month of logs
        self.log_level = self.config.get('log_level', 'INFO')
        
        # Set up logging
        self._setup_logging()
        
        # Performance tracking
        self.session_stats = {
            'session_start': datetime.now(),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'symbols_traded': set(),
            'trade_types_used': set()
        }
        
        # File paths
        self.daily_log_file = self._get_daily_log_file()
        self.trade_csv_file = self._get_trade_csv_file()
        self.performance_json_file = self._get_performance_json_file()
        
        self.logger.info("TradeLogger initialized")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        # Create logger
        self.logger = logging.getLogger('TradeLogger')
        self.logger.setLevel(getattr(logging, self.log_level))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler for daily logs
        daily_log_path = self.log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(daily_log_path)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _get_daily_log_file(self) -> Path:
        """Get daily log file path"""
        return self.log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.log"
    
    def _get_trade_csv_file(self) -> Path:
        """Get trade CSV file path"""
        return self.log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.csv"
    
    def _get_performance_json_file(self) -> Path:
        """Get performance JSON file path"""
        return self.log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.json"
    
    def log_trade_placed(self, trade_data: Dict):
        """Log when a trade is placed"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'TRADE_PLACED',
            'trade_id': trade_data.get('trade_id'),
            'symbol': trade_data.get('symbol'),
            'trade_type': trade_data.get('trade_type'),
            'stake': trade_data.get('stake'),
            'duration': trade_data.get('duration')
        }
        
        self._write_to_json_log(log_entry)
        self.logger.info(f"Trade placed: {trade_data.get('trade_id')} - {trade_data.get('symbol')} {trade_data.get('trade_type')} ${trade_data.get('stake')}")
    
    def log_trade_activated(self, trade_data: Dict):
        """Log when a trade is activated"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'TRADE_ACTIVATED',
            'trade_id': trade_data.get('trade_id'),
            'contract_id': trade_data.get('contract_id'),
            'entry_price': trade_data.get('entry_price')
        }
        
        self._write_to_json_log(log_entry)
        self.logger.info(f"Trade activated: {trade_data.get('trade_id')} at {trade_data.get('entry_price')}")
    
    def log_trade_completed(self, trade_data: Dict):
        """Log when a trade is completed"""
        # Update session statistics
        self._update_session_stats(trade_data)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'TRADE_COMPLETED',
            'trade_id': trade_data.get('trade_id'),
            'symbol': trade_data.get('symbol'),
            'trade_type': trade_data.get('trade_type'),
            'stake': trade_data.get('stake'),
            'entry_price': trade_data.get('entry_price'),
            'exit_price': trade_data.get('exit_price'),
            'profit_loss': trade_data.get('profit_loss'),
            'status': trade_data.get('status'),
            'duration_seconds': trade_data.get('duration_seconds')
        }
        
        self._write_to_json_log(log_entry)
        self._write_to_csv(trade_data)
        
        status = trade_data.get('status', '').upper()
        pnl = trade_data.get('profit_loss', 0)
        self.logger.info(f"Trade completed: {trade_data.get('trade_id')} - {status} - P&L: ${pnl:.2f}")
    
    def log_trade_error(self, trade_data: Dict, error_message: str):
        """Log trade errors"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'TRADE_ERROR',
            'trade_id': trade_data.get('trade_id'),
            'error_message': error_message,
            'trade_data': trade_data
        }
        
        self._write_to_json_log(log_entry)
        self.logger.error(f"Trade error: {trade_data.get('trade_id')} - {error_message}")
    
    def log_portfolio_update(self, portfolio_data: Dict):
        """Log portfolio updates"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'PORTFOLIO_UPDATE',
            'balance': portfolio_data.get('balance'),
            'daily_pnl': portfolio_data.get('daily_pnl'),
            'total_trades': portfolio_data.get('total_trades'),
            'win_rate': portfolio_data.get('win_rate')
        }
        
        self._write_to_json_log(log_entry)
        self.logger.info(f"Portfolio update: Balance ${portfolio_data.get('balance', 0):.2f}, Daily P&L ${portfolio_data.get('daily_pnl', 0):.2f}")
    
    def log_risk_event(self, event_type: str, message: str, data: Dict = None):
        """Log risk management events"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': f'RISK_{event_type.upper()}',
            'message': message,
            'data': data or {}
        }
        
        self._write_to_json_log(log_entry)
        self.logger.warning(f"Risk event: {event_type} - {message}")
    
    def _update_session_stats(self, trade_data: Dict):
        """Update session statistics"""
        self.session_stats['total_trades'] += 1
        pnl = trade_data.get('profit_loss', 0)
        status = trade_data.get('status', '').lower()
        
        # Track symbols and trade types
        self.session_stats['symbols_traded'].add(trade_data.get('symbol'))
        self.session_stats['trade_types_used'].add(trade_data.get('trade_type'))
        
        if status == 'won':
            self.session_stats['winning_trades'] += 1
            self.session_stats['total_profit'] += pnl
            self.session_stats['consecutive_wins'] += 1
            self.session_stats['consecutive_losses'] = 0
            
            if pnl > self.session_stats['largest_win']:
                self.session_stats['largest_win'] = pnl
            
            if self.session_stats['consecutive_wins'] > self.session_stats['max_consecutive_wins']:
                self.session_stats['max_consecutive_wins'] = self.session_stats['consecutive_wins']
        
        elif status == 'lost':
            self.session_stats['losing_trades'] += 1
            self.session_stats['total_loss'] += abs(pnl)
            self.session_stats['consecutive_losses'] += 1
            self.session_stats['consecutive_wins'] = 0
            
            if abs(pnl) > self.session_stats['largest_loss']:
                self.session_stats['largest_loss'] = abs(pnl)
            
            if self.session_stats['consecutive_losses'] > self.session_stats['max_consecutive_losses']:
                self.session_stats['max_consecutive_losses'] = self.session_stats['consecutive_losses']
    
    def _write_to_json_log(self, log_entry: Dict):
        """Write log entry to JSON file"""
        try:
            with open(self.daily_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write to JSON log: {e}")
    
    def _write_to_csv(self, trade_data: Dict):
        """Write trade data to CSV file"""
        try:
            # Check if file exists to write header
            file_exists = self.trade_csv_file.exists()
            
            with open(self.trade_csv_file, 'a', newline='') as f:
                fieldnames = [
                    'timestamp', 'trade_id', 'symbol', 'trade_type', 'stake',
                    'entry_price', 'exit_price', 'profit_loss', 'status',
                    'duration_seconds', 'contract_id'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                # Prepare row data
                row_data = {
                    'timestamp': datetime.now().isoformat(),
                    'trade_id': trade_data.get('trade_id'),
                    'symbol': trade_data.get('symbol'),
                    'trade_type': trade_data.get('trade_type'),
                    'stake': trade_data.get('stake'),
                    'entry_price': trade_data.get('entry_price'),
                    'exit_price': trade_data.get('exit_price'),
                    'profit_loss': trade_data.get('profit_loss'),
                    'status': trade_data.get('status'),
                    'duration_seconds': trade_data.get('duration_seconds'),
                    'contract_id': trade_data.get('contract_id')
                }
                
                writer.writerow(row_data)
                
        except Exception as e:
            self.logger.error(f"Failed to write to CSV: {e}")
    
    def get_session_performance(self) -> Dict:
        """Get current session performance metrics"""
        total_trades = self.session_stats['total_trades']
        winning_trades = self.session_stats['winning_trades']
        
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
        else:
            win_rate = 0.0
        
        net_profit = self.session_stats['total_profit'] - self.session_stats['total_loss']
        
        return {
            'session_duration': str(datetime.now() - self.session_stats['session_start']),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': self.session_stats['losing_trades'],
            'win_rate': round(win_rate, 2),
            'total_profit': round(self.session_stats['total_profit'], 2),
            'total_loss': round(self.session_stats['total_loss'], 2),
            'net_profit': round(net_profit, 2),
            'largest_win': round(self.session_stats['largest_win'], 2),
            'largest_loss': round(self.session_stats['largest_loss'], 2),
            'max_consecutive_wins': self.session_stats['max_consecutive_wins'],
            'max_consecutive_losses': self.session_stats['max_consecutive_losses'],
            'current_consecutive_wins': self.session_stats['consecutive_wins'],
            'current_consecutive_losses': self.session_stats['consecutive_losses'],
            'symbols_traded': list(self.session_stats['symbols_traded']),
            'trade_types_used': list(self.session_stats['trade_types_used'])
        }
    
    def save_session_performance(self):
        """Save session performance to file"""
        performance_data = {
            'session_date': datetime.now().strftime('%Y-%m-%d'),
            'session_start': self.session_stats['session_start'].isoformat(),
            'session_end': datetime.now().isoformat(),
            'performance': self.get_session_performance(),
            'retention_info': {
                'retention_days': self.log_retention_days,
                'expiry_date': (datetime.now() + timedelta(days=self.log_retention_days)).strftime('%Y-%m-%d')
            }
        }
        
        try:
            with open(self.performance_json_file, 'w') as f:
                json.dump(performance_data, f, indent=2)
            
            self.logger.info(f"Session performance saved to {self.performance_json_file} (will be retained for {self.log_retention_days} days)")
            
        except Exception as e:
            self.logger.error(f"Failed to save session performance: {e}")
    
    def generate_daily_report(self) -> str:
        """Generate daily trading report"""
        performance = self.get_session_performance()
        
        # Get retention status
        retention_status = self.get_log_retention_status()
        
        report = f"""
📊 DAILY TRADING REPORT - {datetime.now().strftime('%Y-%m-%d')}
{'='*60}

📈 SESSION OVERVIEW
Duration: {performance['session_duration']}
Total Trades: {performance['total_trades']}

🎯 PERFORMANCE METRICS
Win Rate: {performance['win_rate']:.1f}%
Winning Trades: {performance['winning_trades']}
Losing Trades: {performance['losing_trades']}

💰 PROFIT & LOSS
Total Profit: ${performance['total_profit']:.2f}
Total Loss: ${performance['total_loss']:.2f}
Net P&L: ${performance['net_profit']:.2f}
Largest Win: ${performance['largest_win']:.2f}
Largest Loss: ${performance['largest_loss']:.2f}

🔥 STREAKS
Max Consecutive Wins: {performance['max_consecutive_wins']}
Max Consecutive Losses: {performance['max_consecutive_losses']}
Current Win Streak: {performance['current_consecutive_wins']}
Current Loss Streak: {performance['current_consecutive_losses']}

📊 TRADING ACTIVITY
Symbols Traded: {', '.join(performance['symbols_traded']) if performance['symbols_traded'] else 'None'}
Trade Types Used: {', '.join(performance['trade_types_used']) if performance['trade_types_used'] else 'None'}

💾 LOG INFORMATION
Log Storage Path: {retention_status['storage_path']}
Log Retention Period: {retention_status['retention_policy']}
Current Log Files: {retention_status['total_files']}
Files Expiring Soon: {retention_status['soon_to_expire_count']}

{'='*60}
Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Logs will be retained until: {(datetime.now() + timedelta(days=self.log_retention_days)).strftime('%Y-%m-%d')}
        """
        
        return report.strip()
    
    def export_data(self, start_date: datetime = None, end_date: datetime = None) -> Dict[str, str]:
        """Export trading data for a date range"""
        if not start_date:
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if not end_date:
            end_date = datetime.now()
        
        export_data = {
            'csv_files': [],
            'json_files': [],
            'log_files': []
        }
        
        # Find files in date range
        for file_path in self.log_dir.glob('*'):
            if file_path.is_file():
                file_date_str = None
                
                # Extract date from filename
                if 'trades_' in file_path.name and file_path.suffix == '.csv':
                    file_date_str = file_path.name.replace('trades_', '').replace('.csv', '')
                elif 'performance_' in file_path.name and file_path.suffix == '.json':
                    file_date_str = file_path.name.replace('performance_', '').replace('.json', '')
                elif 'trading_' in file_path.name and file_path.suffix == '.log':
                    file_date_str = file_path.name.replace('trading_', '').replace('.log', '')
                
                if file_date_str:
                    try:
                        file_date = datetime.strptime(file_date_str, '%Y%m%d')
                        if start_date <= file_date <= end_date:
                            if file_path.suffix == '.csv':
                                export_data['csv_files'].append(str(file_path))
                            elif file_path.suffix == '.json':
                                export_data['json_files'].append(str(file_path))
                            elif file_path.suffix == '.log':
                                export_data['log_files'].append(str(file_path))
                    except ValueError:
                        continue
        
        return export_data
    
    def cleanup_old_logs(self):
        """Clean up log files older than the retention period (1 month by default)"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.log_retention_days)
            cutoff_timestamp = cutoff_date.timestamp()
            
            # Clean up all log file types
            for extension in ['.log', '.csv', '.json']:
                old_files = list(self.log_dir.glob(f'*{extension}'))
                
                for file_path in old_files:
                    # Check file modification time
                    mod_time = file_path.stat().st_mtime
                    if mod_time < cutoff_timestamp:
                        file_path.unlink()
                        self.logger.info(f"Cleaned up old file: {file_path} (older than {self.log_retention_days} days)")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")
    
    def get_log_retention_status(self) -> Dict:
        """
        Get current status of log retention and storage
        
        Returns:
        --------
        dict: Log retention status information
        """
        log_files = list(self.log_dir.glob('*.*'))
        now = datetime.now()
        cutoff_date = now - timedelta(days=self.log_retention_days)
        
        # Count files by type
        file_counts = {
            'log': len([f for f in log_files if f.suffix == '.log']),
            'csv': len([f for f in log_files if f.suffix == '.csv']),
            'json': len([f for f in log_files if f.suffix == '.json']),
            'other': len([f for f in log_files if f.suffix not in ['.log', '.csv', '.json']])
        }
        
        # Check for files that will expire soon (within 3 days)
        soon_to_expire = []
        for file_path in log_files:
            mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            days_to_expiry = (mod_time + timedelta(days=self.log_retention_days) - now).days
            
            if 0 <= days_to_expiry <= 3:
                soon_to_expire.append({
                    'file': str(file_path),
                    'days_remaining': days_to_expiry
                })
        
        return {
            'retention_policy': f"{self.log_retention_days} days",
            'total_files': len(log_files),
            'file_counts': file_counts,
            'cutoff_date': cutoff_date.strftime('%Y-%m-%d'),
            'storage_path': str(self.log_dir),
            'soon_to_expire': soon_to_expire[:10],  # Limit to 10 entries
            'soon_to_expire_count': len(soon_to_expire)
        }
    
    def close(self):
        """Close the logger and save final performance"""
        self.save_session_performance()
        self.cleanup_old_logs()
        
        # Log retention status summary
        status = self.get_log_retention_status()
        self.logger.info(f"Log retention: {status['retention_policy']}, Total files: {status['total_files']}")
        if status['soon_to_expire_count'] > 0:
            self.logger.info(f"{status['soon_to_expire_count']} files will expire within 3 days")
            
        self.logger.info("TradeLogger closed")
