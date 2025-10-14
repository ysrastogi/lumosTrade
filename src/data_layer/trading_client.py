import json
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
from src.data_layer.market_stream.stream import MarketStream
from src.utils.callback_manager import CallbackManager

class TradeType(Enum):
    """Trade types supported by Deriv API"""
    CALL = "CALL"
    PUT = "PUT"
    DIGITDIFF = "DIGITDIFF"
    DIGITEVEN = "DIGITEVEN"
    DIGITODD = "DIGITODD"
    DIGITOVER = "DIGITOVER"
    DIGITUNDER = "DIGITUNDER"
    UPORDOWN = "UPORDOWN"

class TradeStatus(Enum):
    """Trade status types"""
    PENDING = "pending"
    ACTIVE = "active"
    WON = "won"
    LOST = "lost"
    CANCELLED = "cancelled"

class Trade:
    """Represents a single trade"""
    def __init__(self, trade_id: str, symbol: str, trade_type: TradeType, 
                 stake: float, duration: int, entry_price: float = None):
        self.trade_id = trade_id
        self.symbol = symbol
        self.trade_type = trade_type
        self.stake = stake
        self.duration = duration
        self.entry_price = entry_price
        self.exit_price = None
        self.status = TradeStatus.PENDING
        self.profit_loss = 0.0
        self.created_at = datetime.now()
        self.closed_at = None
        self.contract_id = None
        self.proposal_id = None
        self.payout = 0.0

class TradingClient:
    """
    Comprehensive Deriv API Trading Client
    Implements all essential trading operations: authorization, balance, symbols,
    contracts, proposals, buy/sell, portfolio, statements, and subscriptions
    """
    
    def __init__(self, market_stream: MarketStream = None, config_path: str = "config/tradding_config.yaml"):
        # Use existing market stream or create new one
        self.market_stream = market_stream or MarketStream(config_path)
        self.config = self.market_stream.config
        
        # Account information
        self.account_info = {}
        self.balance_info = {}
        self.active_symbols = []
        self.available_contracts = {}
        
        # Trading state
        self.active_trades = {}
        self.trade_history = []
        self.portfolio_contracts = []
        self.profit_table = []
        self.statement_transactions = []
        
        # Subscriptions tracking
        self.active_subscriptions = {}
        self.proposal_subscriptions = {}
        
        # Risk management
        self.max_daily_loss = self.config.get('trading', {}).get('risk_management', {}).get('max_loss_per_day', 100)
        self.max_trades_per_day = self.config.get('trading', {}).get('risk_management', {}).get('max_trades_per_day', 50)
        self.max_stake = self.config.get('trading', {}).get('max_stake', 1000)
        self.default_stake = self.config.get('trading', {}).get('default_stake', 10)
        
        # Daily counters
        self.daily_trades_count = 0
        self.daily_loss = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Callbacks and request tracking
        self.request_callbacks = {}
        self.pending_requests = {}
        
        # Threading
        self.request_lock = threading.Lock()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Callback manager
        self.callback_manager = CallbackManager()
    
    def connect(self) -> bool:
        """Connect to Deriv API"""
        if not self.market_stream.is_connected:
            success = self.market_stream.connect()
            if success and self.market_stream.auth_token:
                # Auto-authorize after connection
                return self.authorize()
            return success
        return True
    
    def disconnect(self):
        """Disconnect from Deriv API"""
        # Cancel all subscriptions before disconnecting
        self.forget_all_subscriptions()
        self.market_stream.disconnect()
    
    def is_ready(self) -> bool:
        """Check if trading client is ready"""
        return self.market_stream.is_ready() and bool(self.account_info)
    
    def _send_request(self, request: Dict, callback: Callable = None) -> int:
        """Send request to WebSocket API"""
        if not self.market_stream.is_connected:
            self.logger.error("Not connected to WebSocket")
            return None
        
        with self.request_lock:
            req_id = self.market_stream._get_next_request_id()
            request['req_id'] = req_id
            
            if callback:
                self.request_callbacks[req_id] = callback
            
            self.market_stream._send_message(request)
            self.pending_requests[req_id] = {
                'request': request,
                'timestamp': datetime.now(),
                'callback': callback
            }
            
            return req_id
    
    def _handle_api_response(self, data: Dict):
        """Handle API responses"""
        req_id = data.get('req_id')
        msg_type = data.get('msg_type')
        
        # Remove from pending requests
        if req_id and req_id in self.pending_requests:
            del self.pending_requests[req_id]
        
        # Call specific callback if available
        if req_id and req_id in self.request_callbacks:
            callback = self.request_callbacks[req_id]
            callback(data)
            del self.request_callbacks[req_id]
            return
        
        # Handle specific message types
        if msg_type == 'authorize':
            self._handle_authorize_response(data)
        elif msg_type == 'balance':
            self._handle_balance_response(data)
        elif msg_type == 'active_symbols':
            self._handle_active_symbols_response(data)
        elif msg_type == 'contracts_for':
            self._handle_contracts_for_response(data)
        elif msg_type == 'proposal':
            self._handle_proposal_response(data)
        elif msg_type == 'buy':
            self._handle_buy_response(data)
        elif msg_type == 'sell':
            self._handle_sell_response(data)
        elif msg_type == 'portfolio':
            self._handle_portfolio_response(data)
        elif msg_type == 'profit_table':
            self._handle_profit_table_response(data)
        elif msg_type == 'statement':
            self._handle_statement_response(data)
        elif msg_type == 'proposal_open_contract':
            self._handle_contract_update(data)
        
        # Handle errors
        if data.get('error'):
            self._handle_error_response(data)
    
    # 1. 🔐 AUTHORIZATION
    def authorize(self, token: str = None, custom_callback: Callable = None) -> bool:
        """Authorize WebSocket session
        
        Args:
            token: Optional token to use for authorization. If not provided,
                  will use the token from market_stream.auth_token.
            custom_callback: Optional callback to handle the authorization response.
                            If provided, this callback will be called with the response data.
        """
        auth_token = token or self.market_stream.auth_token
        
        if not auth_token:
            self.logger.error("No auth token available")
            return False
        
        request = {"authorize": auth_token}
        
        def auth_callback(data):
            if data.get('authorize'):  
                self.account_info = data['authorize']
                self.logger.info(f"✅ Authorized - Login ID: {self.account_info.get('loginid')}")
                self.logger.info(f"💰 Balance: ${self.account_info.get('balance', 0):.2f} {self.account_info.get('currency', 'USD')}")
                
                # Automatically get balance and active symbols
                self.get_balance()
                self.get_active_symbols()
            else:
                self.logger.error("❌ Authorization failed")
            
            # Call the custom callback if provided
            if custom_callback:
                custom_callback(data)
        
        req_id = self._send_request(request, auth_callback)
        return req_id is not None
    
    def _handle_authorize_response(self, data: Dict):
        """Handle authorization response"""
        if data.get('authorize'):
            self.account_info = data['authorize']
            self.logger.info(f"Account authorized: {self.account_info.get('loginid')}")
    
    # 2. 💰 ACCOUNT BALANCE
    def get_balance(self, callback: Callable = None) -> int:
        """Get current account balance"""
        request = {"balance": 1}
        
        def balance_callback(data):
            if callback:
                callback(data)
        
        return self._send_request(request, balance_callback)
    
    def _handle_balance_response(self, data: Dict):
        """Handle balance response"""
        if data.get('balance'):
            self.balance_info = data['balance']
            balance = self.balance_info.get('balance', 0)
            currency = self.balance_info.get('currency', 'USD')
            self.logger.info(f"💰 Current balance: ${balance:.2f} {currency}")
    
    # 3. 📜 ACTIVE SYMBOLS
    def get_active_symbols(self, product_type: str = "basic", callback: Callable = None) -> int:
        """Get available trading symbols"""
        request = {
            "active_symbols": "brief",
            "product_type": product_type
        }
        
        def symbols_callback(data):
            if callback:
                callback(data)
        
        return self._send_request(request, symbols_callback)
    
    def _handle_active_symbols_response(self, data: Dict):
        """Handle active symbols response"""
        if data.get('active_symbols'):
            self.active_symbols = data['active_symbols']
            self.logger.info(f"📜 Loaded {len(self.active_symbols)} active symbols")
    
    def get_symbols_by_market(self, market: str = None) -> List[Dict]:
        """Get symbols filtered by market"""
        if not market:
            return self.active_symbols
        
        return [symbol for symbol in self.active_symbols 
                if symbol.get('market') == market]
    
    # 4. 📊 CONTRACTS FOR SYMBOL
    def get_contracts_for_symbol(self, symbol: str, currency: str = "USD", callback: Callable = None) -> int:
        """Get available contracts for a symbol"""
        request = {
            "contracts_for": symbol,
            "currency": currency
        }
        
        def contracts_callback(data):
            if data.get('contracts_for'):
                self.available_contracts[symbol] = data['contracts_for']
                self.logger.info(f"📊 Loaded contracts for {symbol}")
            if callback:
                callback(data)
        
        return self._send_request(request, contracts_callback)
    
    def _handle_contracts_for_response(self, data: Dict):
        """Handle contracts for response"""
        if data.get('contracts_for'):
            symbol = data['contracts_for'].get('symbol')
            if symbol:
                self.available_contracts[symbol] = data['contracts_for']
    
    # 5. 💡 TRADE PROPOSAL
    def get_proposal(self, symbol: str, contract_type: str, amount: float, 
                    duration: int, duration_unit: str = "s", basis: str = "stake",
                    currency: str = "USD", subscribe: bool = False, callback: Callable = None) -> int:
        """Get price proposal for a trade"""
        request = {
            "proposal": 1,
            "amount": amount,
            "basis": basis,
            "contract_type": contract_type,
            "currency": currency,
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol": symbol
        }
        
        if subscribe:
            request["subscribe"] = 1
        
        def proposal_callback(data):
            if subscribe and data.get('proposal'):
                # Store subscription for proposal updates
                proposal_id = data['proposal'].get('id')
                if proposal_id:
                    self.proposal_subscriptions[proposal_id] = {
                        'symbol': symbol,
                        'contract_type': contract_type,
                        'callback': callback
                    }
            if callback:
                callback(data)
        
        return self._send_request(request, proposal_callback)
    
    def _handle_proposal_response(self, data: Dict):
        """Handle proposal response"""
        if data.get('proposal'):
            proposal = data['proposal']
            ask_price = proposal.get('ask_price', 0)
            payout = proposal.get('payout', 0)
            spot = proposal.get('spot', 0)
            self.logger.info(f"💡 Proposal - Price: ${ask_price}, Payout: ${payout:.2f}, Spot: {spot}")
    
    # 6. 🛒 BUY CONTRACT
    def buy_contract(self, proposal_id: str, price: float, callback: Callable = None) -> int:
        """Buy a contract using proposal ID"""
        request = {
            "buy": proposal_id,
            "price": price
        }
        
        def buy_callback(data):
            if data.get('buy'):
                buy_info = data['buy']
                contract_id = buy_info.get('contract_id')
                self.logger.info(f"🛒 Contract purchased - ID: {contract_id}")
                
                # Subscribe to contract updates
                if contract_id:
                    self.subscribe_to_contract(contract_id)
            
            if callback:
                callback(data)
        
        return self._send_request(request, buy_callback)
    
    def _handle_buy_response(self, data: Dict):
        """Handle buy response"""
        if data.get('buy'):
            buy_info = data['buy']
            contract_id = buy_info.get('contract_id')
            buy_price = buy_info.get('buy_price', 0)
            self.logger.info(f"✅ Contract bought - ID: {contract_id}, Price: ${buy_price}")
    
    # 7. 📦 PORTFOLIO (Open Contracts)
    def get_portfolio(self, callback: Callable = None) -> int:
        """Get open positions"""
        request = {"portfolio": 1}
        
        def portfolio_callback(data):
            if callback:
                callback(data)
        
        return self._send_request(request, portfolio_callback)
    
    def _handle_portfolio_response(self, data: Dict):
        """Handle portfolio response"""
        if data.get('portfolio'):
            portfolio = data['portfolio']
            contracts = portfolio.get('contracts', [])
            self.portfolio_contracts = contracts
            self.logger.info(f"📦 Portfolio: {len(contracts)} open contracts")
            
            # Trigger portfolio callbacks
            self.callback_manager.trigger_callbacks("portfolio", data)
    
    # 8. 💸 SELL CONTRACT
    def sell_contract(self, contract_id: int, price: float = 0, callback: Callable = None) -> int:
        """Sell/close a contract"""
        request = {
            "sell": contract_id,
            "price": price
        }
        
        def sell_callback(data):
            if data.get('sell'):
                sell_info = data['sell']
                sell_price = sell_info.get('sell_price', 0)
                self.logger.info(f"💸 Contract sold - Price: ${sell_price}")
            
            if callback:
                callback(data)
        
        return self._send_request(request, sell_callback)
    
    def _handle_sell_response(self, data: Dict):
        """Handle sell response"""
        if data.get('sell'):
            sell_info = data['sell']
            contract_id = sell_info.get('contract_id')
            sell_price = sell_info.get('sell_price', 0)
            self.logger.info(f"✅ Contract sold - ID: {contract_id}, Price: ${sell_price}")
    
    # 9. 📉 PROFIT TABLE
    def get_profit_table(self, limit: int = 50, callback: Callable = None) -> int:
        """Get profit table (trade history)"""
        request = {
            "profit_table": 1,
            "limit": limit
        }
        
        def profit_callback(data):
            if callback:
                callback(data)
        
        return self._send_request(request, profit_callback)
    
    def _handle_profit_table_response(self, data: Dict):
        """Handle profit table response"""
        if data.get('profit_table'):
            profit_table = data['profit_table']
            transactions = profit_table.get('transactions', [])
            self.profit_table = transactions
            self.logger.info(f"📉 Profit table: {len(transactions)} transactions")
    
    # 10. 🧾 STATEMENT (Transaction History)
    def get_statement(self, limit: int = 50, callback: Callable = None) -> int:
        """Get account statement"""
        request = {
            "statement": 1,
            "limit": limit
        }
        
        def statement_callback(data):
            if callback:
                callback(data)
        
        return self._send_request(request, statement_callback)
    
    def _handle_statement_response(self, data: Dict):
        """Handle statement response"""
        if data.get('statement'):
            statement = data['statement']
            transactions = statement.get('transactions', [])
            self.statement_transactions = transactions
            self.logger.info(f"🧾 Statement: {len(transactions)} transactions")
    
    # 11. 🔄 SUBSCRIPTIONS MANAGEMENT
    def subscribe_to_contract(self, contract_id: int, callback: Callable = None) -> int:
        """Subscribe to contract updates"""
        request = {
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1
        }
        
        def contract_callback(data):
            if callback:
                callback(data)
        
        req_id = self._send_request(request, contract_callback)
        if req_id:
            self.active_subscriptions[req_id] = {
                'type': 'contract',
                'contract_id': contract_id,
                'callback': callback
            }
        
        return req_id
    
    def forget_subscription(self, subscription_id: str) -> int:
        """Cancel a specific subscription"""
        request = {"forget": subscription_id}
        
        def forget_callback(data):
            if subscription_id in self.active_subscriptions:
                del self.active_subscriptions[subscription_id]
            self.logger.info(f"🔄 Subscription {subscription_id} cancelled")
        
        return self._send_request(request, forget_callback)
    
    def forget_all_subscriptions(self, subscription_type: str = None) -> int:
        """Cancel all subscriptions or specific type"""
        if subscription_type:
            request = {"forget_all": subscription_type}
        else:
            request = {"forget_all": ["proposal", "ticks", "candles"]}
        
        def forget_all_callback(data):
            self.active_subscriptions.clear()
            self.proposal_subscriptions.clear()
            self.logger.info("🔄 All subscriptions cancelled")
        
        return self._send_request(request, forget_all_callback)
    
    def _handle_contract_update(self, data: Dict):
        """Handle contract update from subscription"""
        if data.get('proposal_open_contract'):
            contract = data['proposal_open_contract']
            contract_id = contract.get('contract_id')
            current_spot = contract.get('current_spot')
            is_sold = contract.get('is_sold', False)
            profit = contract.get('profit', 0)
            
            if is_sold:
                self.logger.info(f"📦 Contract {contract_id} closed - Profit: ${profit:.2f}")
            else:
                self.logger.info(f"📊 Contract {contract_id} update - Spot: {current_spot}, P&L: ${profit:.2f}")
    
    def _handle_error_response(self, data: Dict):
        """Handle API errors"""
        error = data.get('error', {})
        error_code = error.get('code')
        error_message = error.get('message', 'Unknown error')
        self.logger.error(f"❌ API Error [{error_code}]: {error_message}")
    
    # CONVENIENCE METHODS
    def place_trade(self, symbol: str, contract_type: str, amount: float, 
                   duration: int, duration_unit: str = "s", callback: Callable = None) -> Dict:
        """Simplified trade placement (proposal + buy)"""
        
        def proposal_callback(proposal_data):
            if proposal_data.get('proposal'):
                proposal = proposal_data['proposal']
                proposal_id = proposal.get('id')
                ask_price = proposal.get('ask_price')
                
                if proposal_id and ask_price:
                    self.logger.info(f"💡 Got proposal - ID: {proposal_id}, Price: ${ask_price}")
                    
                    # Buy the contract
                    def buy_callback(buy_data):
                        if callback:
                            callback({
                                'proposal': proposal_data,
                                'buy': buy_data,
                                'success': bool(buy_data.get('buy'))
                            })
                    
                    self.buy_contract(proposal_id, ask_price, buy_callback)
                else:
                    if callback:
                        callback({'error': 'Invalid proposal response', 'success': False})
            else:
                if callback:
                    callback({'error': 'Failed to get proposal', 'success': False})
        
        # First get proposal
        return self.get_proposal(symbol, contract_type, amount, duration, duration_unit, callback=proposal_callback)
    
    def get_account_summary(self) -> Dict:
        """Get comprehensive account summary"""
        return {
            'account_info': self.account_info,
            'balance_info': self.balance_info,
            'active_symbols_count': len(self.active_symbols),
            'portfolio_contracts': len(self.portfolio_contracts),
            'profit_table_entries': len(self.profit_table),
            'statement_entries': len(self.statement_transactions),
            'active_subscriptions': len(self.active_subscriptions)
        }
    
    def get_trading_summary(self) -> Dict:
        """Get trading performance summary"""
        total_trades = len(self.profit_table)
        total_profit = sum(t.get('pl', 0) for t in self.profit_table)
        winning_trades = len([t for t in self.profit_table if t.get('pl', 0) > 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': round(win_rate, 2),
            'total_profit': round(total_profit, 2),
            'average_profit': round(total_profit / total_trades, 2) if total_trades > 0 else 0,
            'open_contracts': len(self.portfolio_contracts)
        }
        
    def add_portfolio_callback(self, callback: Callable) -> None:
        """Add a callback for portfolio updates
        
        Args:
            callback: Callback function to be called when portfolio is updated
        """
        self.callback_manager.add_callback("portfolio", callback)
        self.logger.info("Added callback for portfolio updates")
    
    def remove_portfolio_callback(self, callback: Callable) -> bool:
        """Remove a portfolio update callback
        
        Args:
            callback: Callback function to be removed
            
        Returns:
            bool: True if callback was successfully removed, False otherwise
        """
        result = self.callback_manager.remove_callback("portfolio", callback)
        if result:
            self.logger.info("Removed portfolio update callback")
        else:
            self.logger.warning("Failed to remove portfolio update callback")
        return result
