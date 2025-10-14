"""
Message handler for processing different types of WebSocket messages
"""

import logging
from typing import Dict, Callable, Any
from datetime import datetime

from src.utils.callback_manager import CallbackManager
from src.data_layer.market_stream.models import TickData, OHLCData, ContractData, GRANULARITY_MAP

logger = logging.getLogger(__name__)


class MessageHandler:
    """Handles different types of WebSocket messages"""
    
    def __init__(
        self,
        auth_token: str,
        callback_manager: CallbackManager,
        subscription_manager,  # Circular import
        connection_manager,  # Circular import
        subscribe_configured_symbols_func: Callable[[], None]
    ):
        """Initialize the message handler
        
        Args:
            auth_token: Authentication token for Deriv API
            callback_manager: Callback manager for external callbacks
            subscription_manager: Subscription manager reference
            connection_manager: Connection manager reference
            subscribe_configured_symbols_func: Function to subscribe to configured symbols
        """
        self.logger = logger.getChild("MessageHandler")
        self.auth_token = auth_token
        self.callback_manager = callback_manager
        self.subscription_manager = subscription_manager
        self.connection_manager = connection_manager
        self.subscribe_configured_symbols = subscribe_configured_symbols_func
    
    def handle_message(self, data: Dict[str, Any]) -> None:
        """Handle a WebSocket message
        
        Args:
            data: Message data
        """
        msg_type = data.get('msg_type')
        
        # Handle all Deriv API message types
        if msg_type == 'authorize':
            self._handle_auth_response(data)
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
        elif msg_type == 'tick':
            self._handle_tick_data(data)
        elif msg_type == 'candles':
            # Check if this is an OHLC subscription (from ticks_history with style=candles)
            if data.get('echo_req', {}).get('subscribe') == 1 and data.get('echo_req', {}).get('style') == 'candles':
                self._handle_ohlc_data(data)
            else:
                # Regular candles handling
                self._handle_candle_data(data)
        elif msg_type == 'ohlc':
            self._handle_ohlc_data(data)
        elif msg_type == 'ping':
            self._send_pong()
        elif msg_type == 'forget':
            self._handle_forget_response(data)
        elif msg_type == 'forget_all':
            self._handle_forget_all_response(data)
        else:
            # Check for registered callbacks based on request ID
            req_id = data.get('req_id')
            if req_id:
                callback = self.subscription_manager.get_callback(req_id)
                if callback:
                    callback(data)
                    self.subscription_manager.remove_callback(req_id)
                else:
                    self.logger.debug(f"No callback found for req_id {req_id}")
            else:
                self.logger.debug(f"Unhandled message type: {msg_type}")
    
        # Handle any errors in the response
        if data.get('error'):
            self._handle_error_response(data)

    def _handle_auth_response(self, data: Dict[str, Any]) -> None:
        """Handle authentication response
        
        Args:
            data: Authentication response data
        """
        if data.get('authorize'):
            self.connection_manager.set_authenticated(True)
            self.logger.info("Successfully authenticated with Deriv API")
            
            # Trigger subscriptions after successful authentication
            self.subscribe_configured_symbols()
        else:
            error = data.get('error', {})
            self.logger.error(f"Authentication failed: {error.get('message', 'Unknown error')}")
            
            # Retry authentication if there's a token issue
            if error.get('code') == 'InvalidToken':
                self.logger.warning("Invalid token, check your DERIV_AUTH_TOKEN")
                
                # Try the hardcoded token as fallback
                if self.auth_token != "jKdxoLX9rw9LXin":
                    self.logger.info("Trying fallback authentication token...")
                    self.auth_token = "jKdxoLX9rw9LXin"
                    self._retry_authentication()
    
    def _retry_authentication(self) -> None:
        """Retry authentication with the fallback token"""
        req_id = self.connection_manager.get_next_request_id()
        auth_request = {
            "authorize": self.auth_token,
            "req_id": req_id
        }
        
        self.connection_manager.send_message(auth_request)
        self.logger.info("Fallback authentication request sent")
        self.subscription_manager.register_callback(req_id, self._handle_auth_callback)
    
    def _handle_auth_callback(self, data: Dict[str, Any]) -> None:
        """Handle authentication callback
        
        Args:
            data: Authentication callback data
        """
        if data.get('authorize'):
            self.connection_manager.set_authenticated(True)
            self.logger.info("Successfully authenticated with Deriv API")
            self.subscribe_configured_symbols()
        else:
            error = data.get('error', {})
            self.logger.error(f"Authentication callback failed: {error.get('message', 'Unknown error')}")
    
    def _send_pong(self) -> None:
        """Send pong response to ping request"""
        pong_request = {"pong": 1}
        self.connection_manager.send_message(pong_request)
    
    def _handle_tick_data(self, data: Dict[str, Any]) -> None:
        """Handle incoming tick data
        
        Args:
            data: Tick data
        """
        tick = data.get('tick', {})
        if tick:
            symbol = tick.get('symbol')
            price = tick.get('quote')
            timestamp = tick.get('epoch')
            
            self.logger.info(f"Tick - {symbol}: {price} at {datetime.fromtimestamp(timestamp)}")
            
            # Call any registered callbacks for this symbol
            callback_key = f"tick_{symbol}"
            callback = self.subscription_manager.get_callback(callback_key)
            if callback:
                callback(data)
            
            # Trigger callbacks registered via the callback manager
            self.callback_manager.trigger_callbacks("tick", data)
            
            # Convert to strongly typed data model
            tick_data = TickData.from_dict(data)
            self.callback_manager.trigger_callbacks("tick_structured", tick_data)
    
    def _handle_candle_data(self, data: Dict[str, Any]) -> None:
        """Handle incoming candle data
        
        Args:
            data: Candle data
        """
        candles = data.get('candles', [])
        if candles:
            self.logger.info(f"Received {len(candles)} candles")
            
            # Process each candle
            for candle in candles:
                open_price = candle.get('open')
                high_price = candle.get('high')
                low_price = candle.get('low')
                close_price = candle.get('close')
                timestamp = candle.get('epoch')
                
                self.logger.debug(f"Candle - O:{open_price} H:{high_price} L:{low_price} C:{close_price} T:{datetime.fromtimestamp(timestamp)}")
            
            # Trigger callbacks registered via the callback manager
            self.callback_manager.trigger_callbacks("candles", data)
    
    def _handle_ohlc_data(self, data: Dict[str, Any]) -> None:
        """Handle incoming OHLC data
        
        Args:
            data: OHLC data
        """
        # Check for both old format (ohlc) and new format (candles from ticks_history)
        if data.get('ohlc'):
            # Old format handling
            ohlc = data.get('ohlc', {})
            symbol = ohlc.get('symbol')
            open_price = ohlc.get('open')
            high_price = ohlc.get('high')
            low_price = ohlc.get('low')
            close_price = ohlc.get('close')
            timestamp = ohlc.get('epoch')
            
            self.logger.info(f"OHLC - {symbol}: O:{open_price} H:{high_price} L:{low_price} C:{close_price} at {datetime.fromtimestamp(timestamp)}")
            
            # Call any registered callbacks for this symbol
            interval = GRANULARITY_MAP.get(data.get('granularity', 60), "1m")
            callback_key = f"ohlc_{symbol}_{interval}"
            callback = self.subscription_manager.get_callback(callback_key)
            if callback:
                callback(data)
                
            # Trigger callbacks registered via the callback manager
            self.callback_manager.trigger_callbacks("ohlc", data)
            
            # Convert to strongly typed data model
            ohlc_data = OHLCData.from_dict(data)
            self.callback_manager.trigger_callbacks("ohlc_structured", ohlc_data)
            
        elif data.get('candles') and data.get('echo_req', {}).get('ticks_history'):
            # New format handling (from ticks_history with style=candles)
            candles = data.get('candles', [])
            if not candles:
                return
                
            # Get symbol from echo_req
            symbol = data.get('echo_req', {}).get('ticks_history')
            granularity = data.get('echo_req', {}).get('granularity', 60)
            interval = GRANULARITY_MAP.get(granularity, "1m")
            
            # Handle the most recent candle as OHLC data
            latest_candle = candles[-1] if candles else None
            if latest_candle:
                open_price = latest_candle.get('open')
                high_price = latest_candle.get('high')
                low_price = latest_candle.get('low')
                close_price = latest_candle.get('close')
                timestamp = latest_candle.get('epoch')
                
                self.logger.info(f"OHLC from history - {symbol}: O:{open_price} H:{high_price} L:{low_price} C:{close_price} at {datetime.fromtimestamp(timestamp)}")
                
                # Create synthetic OHLC format to maintain compatibility
                synthetic_ohlc_data = {
                    'ohlc': {
                        'symbol': symbol,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'epoch': timestamp
                    },
                    'granularity': granularity
                }
                
                # Call any registered callbacks for this symbol
                callback_key = f"ohlc_{symbol}_{interval}"
                callback = self.subscription_manager.get_callback(callback_key)
                if callback:
                    # Pass both the original data and the synthetic format
                    callback(data)
                
                # Trigger callbacks registered via the callback manager
                self.callback_manager.trigger_callbacks("ohlc", synthetic_ohlc_data)
                
                # Convert to strongly typed data model and trigger structured callbacks
                ohlc_data = OHLCData.from_dict(synthetic_ohlc_data)
                self.callback_manager.trigger_callbacks("ohlc_structured", ohlc_data)
                
                # Also trigger candles callback for full history
                self.callback_manager.trigger_callbacks("candles", data)
    
    def _handle_contract_update(self, data: Dict[str, Any]) -> None:
        """Handle contract update from subscription
        
        Args:
            data: Contract update data
        """
        if data.get('proposal_open_contract'):
            contract = data['proposal_open_contract']
            contract_id = contract.get('contract_id')
            current_spot = contract.get('current_spot')
            profit = contract.get('profit', 0)
            is_sold = contract.get('is_sold', False)
            
            status = "CLOSED" if is_sold else "OPEN"
            self.logger.info(f"Contract {contract_id} ({status}): Spot {current_spot}, P&L ${profit:.2f}")
            
            # Trigger callbacks registered via the callback manager
            self.callback_manager.trigger_callbacks("contract", data)
            
            # Convert to strongly typed data model
            contract_data = ContractData.from_dict(data)
            self.callback_manager.trigger_callbacks("contract_structured", contract_data)
    
    def _handle_balance_response(self, data: Dict[str, Any]) -> None:
        """Handle balance response
        
        Args:
            data: Balance response data
        """
        if data.get('balance'):
            balance_info = data['balance']
            self.logger.info(f"Balance: ${balance_info.get('balance', 0):.2f} {balance_info.get('currency', 'USD')}")
            
            # Trigger callbacks registered via the callback manager
            self.callback_manager.trigger_callbacks("balance", data)
    
    def _handle_active_symbols_response(self, data: Dict[str, Any]) -> None:
        """Handle active symbols response
        
        Args:
            data: Active symbols response data
        """
        if data.get('active_symbols'):
            symbols_count = len(data['active_symbols'])
            self.logger.info(f"Loaded {symbols_count} active symbols")
            
            # Trigger callbacks registered via the callback manager
            self.callback_manager.trigger_callbacks("active_symbols", data)
    
    def _handle_contracts_for_response(self, data: Dict[str, Any]) -> None:
        """Handle contracts for response
        
        Args:
            data: Contracts for response data
        """
        if data.get('contracts_for'):
            symbol = data['contracts_for'].get('symbol')
            contracts_count = len(data['contracts_for'].get('available', []))
            self.logger.info(f"Loaded {contracts_count} contracts for {symbol}")
            
            # Trigger callbacks registered via the callback manager
            self.callback_manager.trigger_callbacks("contracts_for", data)
    
    def _handle_proposal_response(self, data: Dict[str, Any]) -> None:
        """Handle proposal response
        
        Args:
            data: Proposal response data
        """
        if data.get('proposal'):
            proposal = data['proposal']
            ask_price = proposal.get('ask_price', 0)
            payout = proposal.get('payout', 0)
            self.logger.info(f"Proposal: Price ${ask_price}, Payout ${payout:.2f}")
            
            # Trigger callbacks registered via the callback manager
            self.callback_manager.trigger_callbacks("proposal", data)
    
    def _handle_buy_response(self, data: Dict[str, Any]) -> None:
        """Handle buy response
        
        Args:
            data: Buy response data
        """
        if data.get('buy'):
            buy_info = data['buy']
            contract_id = buy_info.get('contract_id')
            buy_price = buy_info.get('buy_price', 0)
            self.logger.info(f"Contract purchased: ID {contract_id}, Price ${buy_price}")
            
            # Trigger callbacks registered via the callback manager
            self.callback_manager.trigger_callbacks("buy", data)
    
    def _handle_sell_response(self, data: Dict[str, Any]) -> None:
        """Handle sell response
        
        Args:
            data: Sell response data
        """
        if data.get('sell'):
            sell_info = data['sell']
            contract_id = sell_info.get('contract_id')
            sell_price = sell_info.get('sell_price', 0)
            self.logger.info(f"Contract sold: ID {contract_id}, Price ${sell_price}")
            
            # Trigger callbacks registered via the callback manager
            self.callback_manager.trigger_callbacks("sell", data)
    
    def _handle_portfolio_response(self, data: Dict[str, Any]) -> None:
        """Handle portfolio response
        
        Args:
            data: Portfolio response data
        """
        if data.get('portfolio'):
            contracts = data['portfolio'].get('contracts', [])
            self.logger.info(f"Portfolio: {len(contracts)} open contracts")
            
            # Trigger callbacks registered via the callback manager
            self.callback_manager.trigger_callbacks("portfolio", data)
    
    def _handle_profit_table_response(self, data: Dict[str, Any]) -> None:
        """Handle profit table response
        
        Args:
            data: Profit table response data
        """
        if data.get('profit_table'):
            transactions = data['profit_table'].get('transactions', [])
            self.logger.info(f"Profit table: {len(transactions)} transactions")
            
            # Trigger callbacks registered via the callback manager
            self.callback_manager.trigger_callbacks("profit_table", data)
    
    def _handle_statement_response(self, data: Dict[str, Any]) -> None:
        """Handle statement response
        
        Args:
            data: Statement response data
        """
        if data.get('statement'):
            transactions = data['statement'].get('transactions', [])
            self.logger.info(f"Statement: {len(transactions)} transactions")
            
            # Trigger callbacks registered via the callback manager
            self.callback_manager.trigger_callbacks("statement", data)
    
    def _handle_forget_response(self, data: Dict[str, Any]) -> None:
        """Handle forget subscription response
        
        Args:
            data: Forget response data
        """
        if data.get('forget'):
            subscription_id = data['forget']
            self.logger.info(f"Subscription {subscription_id} cancelled")
    
    def _handle_forget_all_response(self, data: Dict[str, Any]) -> None:
        """Handle forget all subscriptions response
        
        Args:
            data: Forget all response data
        """
        if data.get('forget_all'):
            self.logger.info("All subscriptions cancelled")
    
    def _handle_error_response(self, data: Dict[str, Any]) -> None:
        """Handle API error responses
        
        Args:
            data: Error response data
        """
        error = data.get('error', {})
        error_code = error.get('code', 'Unknown')
        error_message = error.get('message', 'Unknown error')
        req_id = data.get('req_id', 'Unknown')
        self.logger.error(f"API Error [{error_code}] (req_id: {req_id}): {error_message}")