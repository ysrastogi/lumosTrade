"""
Message handler for WebSocket messages
"""
import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, Callable

from pydantic import parse_obj_as, ValidationError

from src.data_layer.trading_client import TradingClient
from .models import (
    BaseWSMessage, ErrorWSMessage, PingMessage, PongMessage,
    AuthRequestMessage, AuthResponseMessage, WSMessageType,
    SubscribeMessage, UnsubscribeMessage, MarketDataMessage,
    TickMessage, OHLCMessage, BalanceRequestMessage, BalanceResponseMessage,
    SymbolsRequestMessage, SymbolsResponseMessage, ContractsRequestMessage,
    ContractsResponseMessage, ProposalRequestMessage, ProposalResponseMessage,
    BuyRequestMessage, BuyResponseMessage, SellRequestMessage, SellResponseMessage,
    PortfolioRequestMessage, PortfolioResponseMessage, ProfitTableRequestMessage,
    ProfitTableResponseMessage, StatementRequestMessage, StatementResponseMessage
)

logger = logging.getLogger(__name__)

class MessageHandler:
    """
    Handles processing of WebSocket messages
    """
    def __init__(self, trading_client, connection_manager, market_data_handler):
        """
        Initialize the MessageHandler
        
        Args:
            trading_client: The trading client for executing trading operations
            connection_manager: Manager for client connections
            market_data_handler: Handler for market data
        """
        self.trading_client = trading_client
        self.connection_manager = connection_manager
        self.market_data_handler = market_data_handler

    async def handle_message(self, client_id: str, data: Dict):
        """
        Handle an incoming WebSocket message
        
        Args:
            client_id: The ID of the client connection
            data: The message data
        """
        client = self.connection_manager.get_client(client_id)
        if not client:
            logger.warning(f"Received message for unknown client ID: {client_id}")
            return
        
        try:
            # Auto-detect message type for backwards compatibility with older clients
            if "type" not in data:
                # Try to infer type from the message content
                if "balance" in data:
                    data["type"] = WSMessageType.BALANCE
                    logger.warning(f"Auto-detected balance request, adding missing 'type' field: {data}")
                elif "ping" in data:
                    data["type"] = WSMessageType.PING
                    data["timestamp"] = int(time.time() * 1000)
                    logger.warning(f"Auto-detected ping request, adding missing fields: {data}")
                elif "subscribe" in data:
                    data["type"] = WSMessageType.SUBSCRIBE
                    logger.warning(f"Auto-detected subscription request, adding missing 'type' field: {data}")
                elif "unsubscribe" in data:
                    data["type"] = WSMessageType.UNSUBSCRIBE
                    logger.warning(f"Auto-detected unsubscription request, adding missing 'type' field: {data}")
                else:
                    return await self.connection_manager.send_error(client_id, "missing_type", "Missing 'type' field")
            
            message_type = data.get("type")
            
            # Handle different message types
            if message_type == WSMessageType.PING:
                await self._handle_ping(client_id, data)
            
            elif message_type == WSMessageType.AUTH:
                await self._handle_auth(client_id, data)
            
            elif message_type == WSMessageType.SUBSCRIBE:
                await self._handle_subscribe(client_id, data)
            
            elif message_type == WSMessageType.UNSUBSCRIBE:
                await self._handle_unsubscribe(client_id, data)
            
            elif message_type == WSMessageType.BALANCE:
                await self._handle_balance(client_id, data)
            
            elif message_type == WSMessageType.SYMBOLS:
                await self._handle_symbols(client_id, data)
            
            elif message_type == WSMessageType.CONTRACTS:
                await self._handle_contracts(client_id, data)
            
            elif message_type == WSMessageType.PROPOSAL:
                await self._handle_proposal(client_id, data)
            
            elif message_type == WSMessageType.BUY:
                await self._handle_buy(client_id, data)
            
            elif message_type == WSMessageType.SELL:
                await self._handle_sell(client_id, data)
            
            elif message_type == WSMessageType.PORTFOLIO:
                await self._handle_portfolio(client_id, data)
            
            elif message_type == WSMessageType.PROFIT_TABLE:
                await self._handle_profit_table(client_id, data)
            
            elif message_type == WSMessageType.STATEMENT:
                await self._handle_statement(client_id, data)
            
            else:
                await self.connection_manager.send_error(client_id, "unknown_type", f"Unknown message type: {message_type}")
            
        except ValidationError as e:
            await self.connection_manager.send_error(client_id, "validation_error", f"Validation error: {str(e)}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.connection_manager.send_error(client_id, "internal_error", f"Internal error: {str(e)}")
    
    async def _handle_ping(self, client_id: str, data: Dict):
        """Handle ping messages"""
        try:
            ping_msg = parse_obj_as(PingMessage, data)
            pong_msg = PongMessage(
                req_id=ping_msg.req_id,
                timestamp=ping_msg.timestamp
            )
            await self.connection_manager.send_message(client_id, pong_msg.dict())
        except Exception as e:
            logger.error(f"Error handling ping: {e}")
            await self.connection_manager.send_error(client_id, "invalid_ping", "Invalid ping message")
    
    async def _handle_auth(self, client_id: str, data: Dict):
        """Handle authentication messages"""
        try:
            # Extract token - either from parsed message or directly from data
            if "token" in data:
                token = data["token"]
                req_id = data.get("req_id", str(uuid.uuid4()))
            else:
                auth_msg = parse_obj_as(AuthRequestMessage, data)
                token = auth_msg.token
                req_id = auth_msg.req_id
            
            # Store token for this client
            self.connection_manager.set_auth_token(client_id, token)
            
            # Define the authentication result handler
            def auth_result_handler(auth_data):
                try:
                    if auth_data.get('authorize'):
                        # Authentication successful
                        account_info = auth_data.get('authorize', {})
                        
                        # Set client as authenticated
                        self.connection_manager.set_authenticated(client_id, True)
                        self.connection_manager.set_login_id(client_id, account_info.get('loginid'))
                        
                        # Send success response
                        auth_response = {
                            "type": WSMessageType.AUTH_SUCCESS,
                            "req_id": req_id,
                            "message": "Authentication successful",
                            "data": {
                                "loginid": account_info.get('loginid'),
                                "balance": account_info.get('balance'),
                                "currency": account_info.get('currency'),
                                "is_virtual": bool(account_info.get('is_virtual')),
                                "fullname": account_info.get('fullname'),
                            }
                        }
                        asyncio.create_task(self.connection_manager.send_message(client_id, auth_response))
                    else:
                        # Authentication failed
                        error_msg = auth_data.get('error', {}).get('message', 'Authentication failed')
                        error_response = {
                            "type": WSMessageType.ERROR,
                            "req_id": req_id,
                            "code": "auth_failed",
                            "message": error_msg
                        }
                        asyncio.create_task(self.connection_manager.send_message(client_id, error_response))
                        
                except Exception as e:
                    logger.error(f"Error in auth callback: {e}")
                    error_response = {
                        "type": WSMessageType.ERROR,
                        "req_id": req_id,
                        "code": "auth_error",
                        "message": f"Authentication error: {str(e)}"
                    }
                    asyncio.create_task(self.connection_manager.send_message(client_id, error_response))
            
            # Use TradingClient's authorize method with the custom callback
            self.trading_client.authorize(token, auth_result_handler)
            
        except Exception as e:
            logger.error(f"Error handling auth: {e}")
            await self.connection_manager.send_error(client_id, "auth_error", f"Authentication error: {str(e)}")
    
    async def _handle_subscribe(self, client_id: str, data: Dict):
        """Handle subscription messages"""
        try:
            sub_msg = parse_obj_as(SubscribeMessage, data)
            
            for symbol in sub_msg.symbols:
                # Create subscription ID
                subscription_id = f"{symbol}_{sub_msg.stream_type}"
                if sub_msg.interval:
                    subscription_id += f"_{sub_msg.interval}"
                
                # Handle the subscription
                first_subscriber = self.connection_manager.add_subscription(client_id, symbol, subscription_id)
                
                # Subscribe to market data if this is the first client for this subscription
                if first_subscriber:
                    if sub_msg.stream_type == "tick":
                        self.market_data_handler.subscribe_ticks(symbol)
                    elif sub_msg.stream_type == "ohlc":
                        if sub_msg.interval:
                            self.market_data_handler.subscribe_ohlc(symbol, sub_msg.interval)
                    elif sub_msg.stream_type == "candles":
                        # Add your candles handling here if needed
                        pass
            
            # Send confirmation
            await self.connection_manager.send_message(client_id, {
                "type": WSMessageType.INFO,
                "message": f"Subscribed to {len(sub_msg.symbols)} symbols with {sub_msg.stream_type}",
                "symbols": sub_msg.symbols,
                "stream_type": sub_msg.stream_type,
                "interval": sub_msg.interval,
                "req_id": sub_msg.req_id
            })
            
            logger.info(f"Client {client_id} subscribed to {sub_msg.symbols} with {sub_msg.stream_type}")
            
        except Exception as e:
            logger.error(f"Error handling subscription: {e}")
            await self.connection_manager.send_error(client_id, "subscription_error", f"Subscription error: {str(e)}")
    
    async def _handle_unsubscribe(self, client_id: str, data: Dict):
        """Handle unsubscribe messages"""
        try:
            unsub_msg = parse_obj_as(UnsubscribeMessage, data)
            
            if unsub_msg.subscription_id:
                # Unsubscribe from specific subscription ID
                for symbol, sub_ids in self.connection_manager.get_client_subscriptions(client_id).items():
                    if unsub_msg.subscription_id in sub_ids:
                        self.connection_manager.remove_subscription(client_id, symbol, unsub_msg.subscription_id)
            
            elif unsub_msg.symbols:
                # Unsubscribe from all subscriptions for given symbols
                for symbol in unsub_msg.symbols:
                    self.connection_manager.remove_all_symbol_subscriptions(client_id, symbol)
            
            else:
                # Unsubscribe from all
                self.connection_manager.remove_all_subscriptions(client_id)
            
            # Send confirmation
            await self.connection_manager.send_message(client_id, {
                "type": WSMessageType.INFO,
                "message": "Unsubscribed successfully",
                "req_id": unsub_msg.req_id
            })
            
            logger.info(f"Client {client_id} unsubscribed")
            
        except Exception as e:
            logger.error(f"Error handling unsubscribe: {e}")
            await self.connection_manager.send_error(client_id, "unsubscribe_error", f"Unsubscribe error: {str(e)}")
    
    async def _handle_balance(self, client_id: str, data: Dict):
        """Handle balance request"""
        client = self.connection_manager.get_client(client_id)
        if not client.authenticated:
            # Check if they have a token but not authenticated yet
            if client.auth_token:
                # Try to authenticate them first
                logger.warning(f"Client {client_id} has token but is not authenticated, attempting auth")
                await self._handle_auth(client_id, {"type": WSMessageType.AUTH, "token": client.auth_token, "req_id": str(uuid.uuid4())})
                
                # Wait a moment for authentication to complete
                await asyncio.sleep(0.5)
                
                if not self.connection_manager.is_authenticated(client_id):
                    return await self.connection_manager.send_error(client_id, "auth_failed", "Authentication failed")
            else:
                return await self.connection_manager.send_error(client_id, "not_authenticated", "Authentication required")
        
        try:
            # Extract request ID if present
            req_id = data.get("req_id")
            
            # Create a response handler
            def balance_response_handler(balance_data):
                if balance_data and balance_data.get("balance"):
                    balance = balance_data.get("balance", {})
                    response = {
                        "type": WSMessageType.BALANCE,
                        "req_id": req_id,
                        "balance": balance.get("balance"),
                        "currency": balance.get("currency"),
                        "loginid": balance.get("loginid"),
                        "is_virtual": bool(balance.get("is_virtual"))
                    }
                    asyncio.create_task(self.connection_manager.send_message(client_id, response))
                else:
                    error_msg = balance_data.get('error', {}).get('message', 'Failed to get balance')
                    asyncio.create_task(self.connection_manager.send_error(client_id, "balance_error", error_msg))
            
            # Make sure we're using the client's token for this request
            self.trading_client.authorize(client.auth_token)
            
            # Get balance with the custom response handler
            self.trading_client.get_balance(callback=balance_response_handler)
        
        except Exception as e:
            logger.error(f"Error handling balance request: {e}")
            await self.connection_manager.send_error(client_id, "balance_error", f"Balance error: {str(e)}")
            
        # Return immediately as the response will be sent asynchronously
        return
    
    async def _handle_symbols(self, client_id: str, data: Dict):
        """Handle symbols request"""
        try:
            # Parse as proper message or extract req_id directly
            req_id = data.get("req_id", str(uuid.uuid4()))
            
            # Create a response handler
            def symbols_response_handler(symbols_data):
                if symbols_data and symbols_data.get("active_symbols"):
                    response = {
                        "type": WSMessageType.SYMBOLS,
                        "req_id": req_id,
                        "symbols": symbols_data.get("active_symbols", [])
                    }
                    asyncio.create_task(self.connection_manager.send_message(client_id, response))
                else:
                    error_msg = symbols_data.get('error', {}).get('message', 'Failed to get symbols')
                    asyncio.create_task(self.connection_manager.send_error(client_id, "symbols_error", error_msg))
            
            # Make sure we're using the client's token if they're authenticated
            client = self.connection_manager.get_client(client_id)
            if client and client.authenticated and client.auth_token:
                self.trading_client.authorize(client.auth_token)
            
            # Get active symbols with the custom response handler
            self.trading_client.get_active_symbols(callback=symbols_response_handler)
            
            # Return immediately as the response will be sent asynchronously
            return
        
        except Exception as e:
            logger.error(f"Error handling symbols request: {e}")
            await self.connection_manager.send_error(client_id, "symbols_error", f"Symbols error: {str(e)}")
    
    async def _handle_contracts(self, client_id: str, data: Dict):
        """Handle contracts request"""
        try:
            # Parse as proper message or extract values directly
            if "symbol" in data:
                symbol = data["symbol"]
                req_id = data.get("req_id", str(uuid.uuid4()))
            else:
                contracts_msg = parse_obj_as(ContractsRequestMessage, data)
                symbol = contracts_msg.symbol
                req_id = contracts_msg.req_id
                
            # Create a response handler
            def contracts_response_handler(contracts_data):
                if contracts_data and contracts_data.get("contracts_for"):
                    contracts = contracts_data.get("contracts_for", {})
                    response = {
                        "type": WSMessageType.CONTRACTS,
                        "req_id": req_id,
                        "symbol": symbol,
                        "contracts": contracts.get("available", [])
                    }
                    asyncio.create_task(self.connection_manager.send_message(client_id, response))
                else:
                    error_msg = contracts_data.get('error', {}).get('message', 'Failed to get contracts')
                    asyncio.create_task(self.connection_manager.send_error(client_id, "contracts_error", error_msg))
            
            # Make sure we're using the client's token if they're authenticated
            client = self.connection_manager.get_client(client_id)
            if client and client.authenticated and client.auth_token:
                self.trading_client.authorize(client.auth_token)
            
            # Get contracts with the custom response handler
            self.trading_client.get_contracts_for_symbol(symbol, callback=contracts_response_handler)
            
            # Return immediately as the response will be sent asynchronously
            return
            
        except Exception as e:
            logger.error(f"Error handling contracts request: {e}")
            await self.connection_manager.send_error(client_id, "contracts_error", f"Contracts error: {str(e)}")
    
    async def _handle_proposal(self, client_id: str, data: Dict):
        """Handle proposal request"""
        client = self.connection_manager.get_client(client_id)
        if not client.authenticated:
            # Check if they have a token but not authenticated yet
            if client.auth_token:
                # Try to authenticate them first
                logger.warning(f"Client {client_id} has token but is not authenticated, attempting auth")
                await self._handle_auth(client_id, {"type": WSMessageType.AUTH, "token": client.auth_token, "req_id": str(uuid.uuid4())})
                
                # Wait a moment for authentication to complete
                await asyncio.sleep(0.5)
                
                if not self.connection_manager.is_authenticated(client_id):
                    return await self.connection_manager.send_error(client_id, "auth_failed", "Authentication failed")
            else:
                return await self.connection_manager.send_error(client_id, "not_authenticated", "Authentication required")
        
        try:
            # Try to parse as proper message, or extract values directly
            if "symbol" in data and "contract_type" in data and "amount" in data and "duration" in data:
                symbol = data["symbol"]
                contract_type = data["contract_type"]
                amount = float(data["amount"])
                duration = int(data["duration"])
                duration_unit = data.get("duration_unit", "s")
                barrier = data.get("barrier")
                req_id = data.get("req_id", str(uuid.uuid4()))
            else:
                proposal_msg = parse_obj_as(ProposalRequestMessage, data)
                symbol = proposal_msg.symbol
                contract_type = proposal_msg.contract_type
                amount = proposal_msg.amount
                duration = proposal_msg.duration
                duration_unit = proposal_msg.duration_unit
                barrier = proposal_msg.barrier
                req_id = proposal_msg.req_id
            
            # Create a response handler
            def proposal_response_handler(proposal_data):
                if proposal_data and proposal_data.get("proposal"):
                    proposal = proposal_data.get("proposal", {})
                    response = {
                        "type": WSMessageType.PROPOSAL,
                        "req_id": req_id,
                        "symbol": symbol,
                        "proposal_id": proposal.get("id"),
                        "ask_price": proposal.get("ask_price"),
                        "payout": proposal.get("payout"),
                        "spot": proposal.get("spot"),
                        "display_value": proposal.get("display_value"),
                        "date_start": proposal.get("date_start"),
                        "date_expiry": proposal.get("date_expiry")
                    }
                    asyncio.create_task(self.connection_manager.send_message(client_id, response))
                else:
                    error_msg = proposal_data.get('error', {}).get('message', 'Failed to get proposal')
                    asyncio.create_task(self.connection_manager.send_error(client_id, "proposal_error", error_msg))
            
            # Make sure we're using the client's token 
            if client.auth_token:
                self.trading_client.authorize(client.auth_token)
            
            # Get proposal with the custom response handler
            # Use barrier if provided
            if barrier:
                self.trading_client.get_proposal(
                    symbol=symbol,
                    contract_type=contract_type,
                    amount=amount,
                    duration=duration,
                    duration_unit=duration_unit,
                    barrier=barrier,
                    callback=proposal_response_handler
                )
            else:
                self.trading_client.get_proposal(
                    symbol=symbol,
                    contract_type=contract_type,
                    amount=amount,
                    duration=duration,
                    duration_unit=duration_unit,
                    callback=proposal_response_handler
                )
            
            # Return immediately as the response will be sent asynchronously
            return
            
        except Exception as e:
            logger.error(f"Error handling proposal request: {e}")
            await self.connection_manager.send_error(client_id, "proposal_error", f"Proposal error: {str(e)}")
    
    async def _handle_buy(self, client_id: str, data: Dict):
        """Handle buy request"""
        client = self.connection_manager.get_client(client_id)
        if not client.authenticated:
            # Check if they have a token but not authenticated yet
            if client.auth_token:
                # Try to authenticate them first
                logger.warning(f"Client {client_id} has token but is not authenticated, attempting auth")
                await self._handle_auth(client_id, {"type": WSMessageType.AUTH, "token": client.auth_token, "req_id": str(uuid.uuid4())})
                
                # Wait a moment for authentication to complete
                await asyncio.sleep(0.5)
                
                if not self.connection_manager.is_authenticated(client_id):
                    return await self.connection_manager.send_error(client_id, "auth_failed", "Authentication failed")
            else:
                return await self.connection_manager.send_error(client_id, "not_authenticated", "Authentication required")
        
        try:
            # Try to parse as proper message or extract values directly
            if "proposal_id" in data and "price" in data:
                proposal_id = data["proposal_id"]
                price = float(data["price"]) 
                req_id = data.get("req_id", str(uuid.uuid4()))
            else:
                buy_msg = parse_obj_as(BuyRequestMessage, data)
                proposal_id = buy_msg.proposal_id
                price = buy_msg.price
                req_id = buy_msg.req_id
                
            # Create a response handler
            def buy_response_handler(buy_data):
                if buy_data and buy_data.get("buy"):
                    buy = buy_data.get("buy", {})
                    response = {
                        "type": WSMessageType.BUY,
                        "req_id": req_id,
                        "contract_id": buy.get("contract_id"),
                        "transaction_id": buy.get("transaction_id"),
                        "balance_after": buy.get("balance_after"),
                        "buy_price": buy.get("buy_price"),
                        "purchase_time": buy.get("purchase_time"),
                        "longcode": buy.get("longcode")
                    }
                    asyncio.create_task(self.connection_manager.send_message(client_id, response))
                else:
                    error_msg = buy_data.get('error', {}).get('message', 'Failed to buy contract')
                    asyncio.create_task(self.connection_manager.send_error(client_id, "buy_error", error_msg))
            
            # Make sure we're using the client's token
            if client.auth_token:
                self.trading_client.authorize(client.auth_token)
                
            # Buy contract with the custom response handler
            self.trading_client.buy_contract(
                proposal_id=proposal_id,
                price=price,
                callback=buy_response_handler
            )
            
            # Return immediately as the response will be sent asynchronously
            return
            
        except Exception as e:
            logger.error(f"Error handling buy request: {e}")
            await self.connection_manager.send_error(client_id, "buy_error", f"Buy error: {str(e)}")
    
    async def _handle_sell(self, client_id: str, data: Dict):
        """Handle sell request"""
        if not self.connection_manager.is_authenticated(client_id):
            return await self.connection_manager.send_error(client_id, "not_authenticated", "Authentication required")
        
        try:
            sell_msg = parse_obj_as(SellRequestMessage, data)
            
            client = self.connection_manager.get_client(client_id)
            if client.auth_token:
                self.trading_client.authorize(client.auth_token)
            
            sell_data = self.trading_client.sell_contract(
                contract_id=sell_msg.contract_id
            )
            
            if sell_data and sell_data.get("sell"):
                sell = sell_data.get("sell", {})
                await self.connection_manager.send_message(client_id, {
                    "type": WSMessageType.SELL,
                    "req_id": sell_msg.req_id,
                    "contract_id": sell_msg.contract_id,
                    "transaction_id": sell.get("transaction_id"),
                    "sold_for": sell.get("sold_for"),
                    "remaining_contracts": sell.get("contract_id") != sell_msg.contract_id,
                })
            else:
                error_msg = sell_data.get("error", {}).get("message", "Failed to sell contract")
                await self.connection_manager.send_error(client_id, "sell_error", error_msg)
        
        except Exception as e:
            logger.error(f"Error handling sell request: {e}")
            await self.connection_manager.send_error(client_id, "sell_error", f"Sell error: {str(e)}")
    
    async def _handle_portfolio(self, client_id: str, data: Dict):
        """Handle portfolio request"""
        if not self.connection_manager.is_authenticated(client_id):
            return await self.connection_manager.send_error(client_id, "not_authenticated", "Authentication required")
        
        try:
            portfolio_msg = parse_obj_as(PortfolioRequestMessage, data)
            
            client = self.connection_manager.get_client(client_id)
            if client.auth_token:
                self.trading_client.authorize(client.auth_token)
                
            portfolio_data = self.trading_client.get_portfolio()
            
            if portfolio_data and portfolio_data.get("portfolio"):
                await self.connection_manager.send_message(client_id, {
                    "type": WSMessageType.PORTFOLIO,
                    "req_id": portfolio_msg.req_id,
                    "contracts": portfolio_data.get("portfolio", {}).get("contracts", [])
                })
            else:
                await self.connection_manager.send_error(client_id, "portfolio_error", "Failed to get portfolio")
        
        except Exception as e:
            logger.error(f"Error handling portfolio request: {e}")
            await self.connection_manager.send_error(client_id, "portfolio_error", f"Portfolio error: {str(e)}")
    
    async def _handle_profit_table(self, client_id: str, data: Dict):
        """Handle profit table request"""
        if not self.connection_manager.is_authenticated(client_id):
            return await self.connection_manager.send_error(client_id, "not_authenticated", "Authentication required")
        
        try:
            profit_msg = parse_obj_as(ProfitTableRequestMessage, data)
            
            client = self.connection_manager.get_client(client_id)
            if client.auth_token:
                self.trading_client.authorize(client.auth_token)
                
            profit_data = self.trading_client.get_profit_table(
                limit=profit_msg.limit,
                offset=profit_msg.offset,
                start_date=profit_msg.start_date,
                end_date=profit_msg.end_date
            )
            
            if profit_data and profit_data.get("profit_table"):
                profit_table = profit_data.get("profit_table", {})
                await self.connection_manager.send_message(client_id, {
                    "type": WSMessageType.PROFIT_TABLE,
                    "req_id": profit_msg.req_id,
                    "transactions": profit_table.get("transactions", []),
                    "count": profit_table.get("count", 0)
                })
            else:
                await self.connection_manager.send_error(client_id, "profit_table_error", "Failed to get profit table")
        
        except Exception as e:
            logger.error(f"Error handling profit table request: {e}")
            await self.connection_manager.send_error(client_id, "profit_table_error", f"Profit table error: {str(e)}")
    
    async def _handle_statement(self, client_id: str, data: Dict):
        """Handle statement request"""
        if not self.connection_manager.is_authenticated(client_id):
            return await self.connection_manager.send_error(client_id, "not_authenticated", "Authentication required")
        
        try:
            statement_msg = parse_obj_as(StatementRequestMessage, data)
            
            client = self.connection_manager.get_client(client_id)
            if client.auth_token:
                self.trading_client.authorize(client.auth_token)
                
            statement_data = self.trading_client.get_statement(
                limit=statement_msg.limit,
                offset=statement_msg.offset,
                start_date=statement_msg.start_date,
                end_date=statement_msg.end_date
            )
            
            if statement_data and statement_data.get("statement"):
                statement = statement_data.get("statement", {})
                await self.connection_manager.send_message(client_id, {
                    "type": WSMessageType.STATEMENT,
                    "req_id": statement_msg.req_id,
                    "transactions": statement.get("transactions", []),
                    "count": statement.get("count", 0)
                })
            else:
                await self.connection_manager.send_error(client_id, "statement_error", "Failed to get statement")
        
        except Exception as e:
            logger.error(f"Error handling statement request: {e}")
            await self.connection_manager.send_error(client_id, "statement_error", f"Statement error: {str(e)}")