"""
WebSocket server for market data and trading
"""
import json
import logging
import asyncio
import time
from typing import Dict, List, Any, Set, Optional, Union
from datetime import datetime
import uuid
import websockets
from pydantic import ValidationError, parse_obj_as

from src.data_layer.market_stream import MarketStream
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

class ClientConnection:
    """Represents a connected client"""
    def __init__(self, websocket, client_id: str):
        self.websocket = websocket
        self.client_id = client_id
        self.authenticated = False
        self.auth_token = None
        self.login_id = None
        self.subscriptions: Dict[str, Set[str]] = {}  # symbol -> set of subscription_ids
        self.last_activity = time.time()
    
    def __str__(self):
        return f"Client({self.client_id}{'✓' if self.authenticated else ''})"


class WebSocketServer:
    """
    WebSocket server for real-time market data and trading API
    """
    def __init__(self):
        self.connections: Dict[str, ClientConnection] = {}
        self.market_stream = MarketStream()
        self.trading_client = TradingClient(market_stream=self.market_stream)
        self.subscription_to_clients: Dict[str, Set[str]] = {}  # subscription_id -> set of client_ids
        self.running = False
        self.server_task = None
        self.heartbeat_task = None
        self.cleanup_task = None
        
        # Initialize market stream callbacks
        self._setup_market_callbacks()
    
    def _setup_market_callbacks(self):
        """Set up callbacks for market data"""
        # Handle tick data
        self.market_stream.add_callback("tick", self._on_tick_data)
        
        # Handle OHLC data
        self.market_stream.add_callback("ohlc", self._on_ohlc_data)
        
        # Handle portfolio updates
        self.trading_client.add_portfolio_callback(self._on_portfolio_update)
    
    async def start(self, host: str = "0.0.0.0", port: int = 8765):
        """Start the WebSocket server"""
        if self.running:
            return
        
        self.running = True
        
        # Connect to Deriv API
        if not self.market_stream.is_connected:
            self.market_stream.connect()
        
        # Start server
        logger.info(f"Starting WebSocket server on {host}:{port}")
        self.server = await websockets.serve(self._handle_client, host, port)
        
        # Start heartbeat task
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop the WebSocket server"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            
        if self.cleanup_task:
            self.cleanup_task.cancel()
            
        # Close the WebSocket server
        if hasattr(self, 'server'):
            self.server.close()
        
        # Close all client connections
        for client_id, connection in list(self.connections.items()):
            try:
                await connection.websocket.close()
            except Exception as e:
                logger.error(f"Error closing connection for {client_id}: {e}")
            
            self._remove_client(client_id)
        
        # Disconnect from Deriv API
        self.market_stream.disconnect()
        
        logger.info("WebSocket server stopped")
    
    async def _handle_client(self, websocket, path):
        """Handle a client connection"""
        client_id = str(uuid.uuid4())
        client = ClientConnection(websocket, client_id)
        
        # Add client to connections
        self.connections[client_id] = client
        
        logger.info(f"New connection: {client}")
        
        try:
            # Welcome message
            await self._send_info_message(
                client, 
                f"Welcome to LumosTrade WebSocket API! Client ID: {client_id}"
            )
            
            # Listen for messages
            async for message in websocket:
                client.last_activity = time.time()
                
                try:
                    # Parse message
                    data = json.loads(message)
                    await self._handle_message(client, data)
                except json.JSONDecodeError:
                    await self._send_error(client, "invalid_format", "Invalid JSON format")
                except Exception as e:
                    logger.error(f"Error handling message from {client}: {e}")
                    await self._send_error(client, "internal_error", f"Internal error: {str(e)}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {client}")
        except Exception as e:
            logger.error(f"Error handling connection {client}: {e}")
        finally:
            self._remove_client(client_id)
    
    def _remove_client(self, client_id: str):
        """Remove a client and clean up subscriptions"""
        if client_id not in self.connections:
            return
        
        client = self.connections[client_id]
        
        # Clean up subscriptions
        for symbol, subscription_ids in client.subscriptions.items():
            for sub_id in subscription_ids:
                if sub_id in self.subscription_to_clients:
                    self.subscription_to_clients[sub_id].discard(client_id)
                    
                    # If no more clients for this subscription, unsubscribe from market stream
                    if not self.subscription_to_clients[sub_id]:
                        # Unsubscribe from Deriv API
                        self.market_stream.unsubscribe(sub_id)
                        del self.subscription_to_clients[sub_id]
        
        # Remove client
        del self.connections[client_id]
        logger.info(f"Removed client: {client}")
    
    async def _handle_message(self, client: ClientConnection, data: Dict):
        """Handle incoming WebSocket messages"""
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
                    return await self._send_error(client, "missing_type", "Missing 'type' field")
            
            message_type = data.get("type")
            
            # Handle different message types
            if message_type == WSMessageType.PING:
                await self._handle_ping(client, data)
            
            elif message_type == WSMessageType.AUTH:
                await self._handle_auth(client, data)
            
            elif message_type == WSMessageType.SUBSCRIBE:
                await self._handle_subscribe(client, data)
            
            elif message_type == WSMessageType.UNSUBSCRIBE:
                await self._handle_unsubscribe(client, data)
            
            elif message_type == WSMessageType.BALANCE:
                await self._handle_balance(client, data)
            
            elif message_type == WSMessageType.SYMBOLS:
                await self._handle_symbols(client, data)
            
            elif message_type == WSMessageType.CONTRACTS:
                await self._handle_contracts(client, data)
            
            elif message_type == WSMessageType.PROPOSAL:
                await self._handle_proposal(client, data)
            
            elif message_type == WSMessageType.BUY:
                await self._handle_buy(client, data)
            
            elif message_type == WSMessageType.SELL:
                await self._handle_sell(client, data)
            
            elif message_type == WSMessageType.PORTFOLIO:
                await self._handle_portfolio(client, data)
            
            elif message_type == WSMessageType.PROFIT_TABLE:
                await self._handle_profit_table(client, data)
            
            elif message_type == WSMessageType.STATEMENT:
                await self._handle_statement(client, data)
            
            else:
                await self._send_error(client, "unknown_type", f"Unknown message type: {message_type}")
            
        except ValidationError as e:
            await self._send_error(client, "validation_error", f"Validation error: {str(e)}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self._send_error(client, "internal_error", f"Internal error: {str(e)}")
    
    async def _handle_ping(self, client: ClientConnection, data: Dict):
        """Handle ping messages"""
        try:
            ping_msg = parse_obj_as(PingMessage, data)
            pong_msg = PongMessage(
                req_id=ping_msg.req_id,
                timestamp=ping_msg.timestamp
            )
            await self._send_message(client, pong_msg.dict())
        except Exception as e:
            logger.error(f"Error handling ping: {e}")
            await self._send_error(client, "invalid_ping", "Invalid ping message")
    
    async def _handle_auth(self, client: ClientConnection, data: Dict):
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
            client.auth_token = token
            
            # Define the authentication result handler
            def auth_result_handler(auth_data):
                try:
                    if auth_data.get('authorize'):
                        # Authentication successful
                        client.authenticated = True
                        client.login_id = auth_data['authorize'].get('loginid')
                        
                        # Create success response
                        auth_response = {
                            "type": WSMessageType.AUTH_SUCCESS,
                            "req_id": req_id,
                            "message": "Authentication successful",
                            "data": {
                                "login_id": client.login_id,
                                "currency": auth_data['authorize'].get('currency'),
                                "balance": auth_data['authorize'].get('balance'),
                                "is_virtual": bool(auth_data['authorize'].get('is_virtual')),
                            }
                        }
                        
                        # Send response asynchronously
                        asyncio.create_task(self._send_message(client, auth_response))
                        logger.info(f"Authentication successful for {client}")
                    else:
                        # Authentication failed
                        error_response = {
                            "type": WSMessageType.AUTH_FAILURE,
                            "req_id": req_id,
                            "message": "Authentication failed"
                        }
                        asyncio.create_task(self._send_message(client, error_response))
                        logger.warning(f"Authentication failed for {client}")
                except Exception as e:
                    logger.error(f"Error in auth callback: {e}")
                    error_response = {
                        "type": WSMessageType.ERROR,
                        "req_id": req_id,
                        "code": "auth_error",
                        "message": f"Authentication error: {str(e)}"
                    }
                    asyncio.create_task(self._send_message(client, error_response))
            
            # Use TradingClient's authorize method with the custom callback
            self.trading_client.authorize(client.auth_token, auth_result_handler)
            
        except Exception as e:
            logger.error(f"Error handling auth: {e}")
            await self._send_error(client, "auth_error", f"Authentication error: {str(e)}")
    
    async def _handle_subscribe(self, client: ClientConnection, data: Dict):
        """Handle subscription messages"""
        try:
            sub_msg = parse_obj_as(SubscribeMessage, data)
            
            for symbol in sub_msg.symbols:
                # Create subscription ID
                subscription_id = f"{symbol}_{sub_msg.stream_type}"
                if sub_msg.interval:
                    subscription_id += f"_{sub_msg.interval}"
                
                # Initialize subscription for symbol if needed
                if symbol not in client.subscriptions:
                    client.subscriptions[symbol] = set()
                
                # Add subscription ID to client
                client.subscriptions[symbol].add(subscription_id)
                
                # Add client to subscription tracking
                if subscription_id not in self.subscription_to_clients:
                    self.subscription_to_clients[subscription_id] = set()
                
                self.subscription_to_clients[subscription_id].add(client.client_id)
                
                # Subscribe to market data from Deriv API
                if len(self.subscription_to_clients[subscription_id]) == 1:
                    # First client subscribing to this data
                    if sub_msg.stream_type == "tick":
                        self.market_stream.subscribe_ticks(symbol)
                    elif sub_msg.stream_type == "ohlc":
                        interval = sub_msg.interval or "1m"
                        self.market_stream.subscribe_ohlc(symbol, interval)
                    elif sub_msg.stream_type == "candles":
                        interval = sub_msg.interval or "1m"
                        self.market_stream.subscribe_candles(symbol, interval)
            
            # Send confirmation
            await self._send_message(client, {
                "type": WSMessageType.INFO,
                "message": f"Subscribed to {len(sub_msg.symbols)} symbols with {sub_msg.stream_type}",
                "symbols": sub_msg.symbols,
                "stream_type": sub_msg.stream_type,
                "interval": sub_msg.interval,
                "req_id": sub_msg.req_id
            })
            
            logger.info(f"Client {client} subscribed to {sub_msg.symbols} with {sub_msg.stream_type}")
            
        except Exception as e:
            logger.error(f"Error handling subscription: {e}")
            await self._send_error(client, "subscription_error", f"Subscription error: {str(e)}")
    
    async def _handle_unsubscribe(self, client: ClientConnection, data: Dict):
        """Handle unsubscribe messages"""
        try:
            unsub_msg = parse_obj_as(UnsubscribeMessage, data)
            
            if unsub_msg.subscription_id:
                # Unsubscribe from specific subscription ID
                for symbol, sub_ids in client.subscriptions.items():
                    if unsub_msg.subscription_id in sub_ids:
                        sub_ids.discard(unsub_msg.subscription_id)
                        
                        if unsub_msg.subscription_id in self.subscription_to_clients:
                            self.subscription_to_clients[unsub_msg.subscription_id].discard(client.client_id)
                            
                            # If no more clients for this subscription, unsubscribe from market stream
                            if not self.subscription_to_clients[unsub_msg.subscription_id]:
                                # Determine subscription type and parameters
                                parts = unsub_msg.subscription_id.split("_")
                                if len(parts) >= 2:
                                    symbol = parts[0]
                                    stream_type = parts[1]
                                    
                                    if stream_type == "tick":
                                        self.market_stream.unsubscribe_ticks(symbol)
                                    elif stream_type in ["ohlc", "candles"] and len(parts) >= 3:
                                        interval = parts[2]
                                        self.market_stream.unsubscribe_ohlc(symbol, interval)
                                
                                del self.subscription_to_clients[unsub_msg.subscription_id]
                        break
            
            elif unsub_msg.symbols:
                # Unsubscribe from all subscriptions for given symbols
                for symbol in unsub_msg.symbols:
                    if symbol in client.subscriptions:
                        for sub_id in list(client.subscriptions[symbol]):
                            # Check stream type if specified
                            if unsub_msg.stream_type:
                                if not sub_id.startswith(f"{symbol}_{unsub_msg.stream_type}"):
                                    continue
                            
                            # Remove subscription
                            client.subscriptions[symbol].discard(sub_id)
                            
                            if sub_id in self.subscription_to_clients:
                                self.subscription_to_clients[sub_id].discard(client.client_id)
                                
                                # If no more clients for this subscription, unsubscribe from market stream
                                if not self.subscription_to_clients[sub_id]:
                                    # Determine subscription type and parameters
                                    parts = sub_id.split("_")
                                    if len(parts) >= 2:
                                        symbol_part = parts[0]
                                        stream_type = parts[1]
                                        
                                        if stream_type == "tick":
                                            self.market_stream.unsubscribe_ticks(symbol_part)
                                        elif stream_type in ["ohlc", "candles"] and len(parts) >= 3:
                                            interval = parts[2]
                                            self.market_stream.unsubscribe_ohlc(symbol_part, interval)
                                    
                                    del self.subscription_to_clients[sub_id]
                        
                        # Remove symbol if no subscriptions left
                        if not client.subscriptions[symbol]:
                            del client.subscriptions[symbol]
            
            else:
                # Unsubscribe from all
                for symbol, sub_ids in list(client.subscriptions.items()):
                    for sub_id in list(sub_ids):
                        # Check stream type if specified
                        if unsub_msg.stream_type and not sub_id.startswith(f"{symbol}_{unsub_msg.stream_type}"):
                            continue
                        
                        # Remove subscription
                        sub_ids.discard(sub_id)
                        
                        if sub_id in self.subscription_to_clients:
                            self.subscription_to_clients[sub_id].discard(client.client_id)
                            
                            # If no more clients for this subscription, unsubscribe from market stream
                            if not self.subscription_to_clients[sub_id]:
                                # Determine subscription type and parameters
                                parts = sub_id.split("_")
                                if len(parts) >= 2:
                                    symbol_part = parts[0]
                                    stream_type = parts[1]
                                    
                                    if stream_type == "tick":
                                        self.market_stream.unsubscribe_ticks(symbol_part)
                                    elif stream_type in ["ohlc", "candles"] and len(parts) >= 3:
                                        interval = parts[2]
                                        self.market_stream.unsubscribe_ohlc(symbol_part, interval)
                                
                                del self.subscription_to_clients[sub_id]
                    
                    # Remove symbol if no subscriptions left
                    if not sub_ids:
                        del client.subscriptions[symbol]
            
            # Send confirmation
            await self._send_message(client, {
                "type": WSMessageType.INFO,
                "message": "Unsubscribed successfully",
                "req_id": unsub_msg.req_id
            })
            
            logger.info(f"Client {client} unsubscribed")
            
        except Exception as e:
            logger.error(f"Error handling unsubscribe: {e}")
            await self._send_error(client, "unsubscribe_error", f"Unsubscribe error: {str(e)}")
    
    async def _handle_balance(self, client: ClientConnection, data: Dict):
        """Handle balance request"""
        if not client.authenticated:
            # Check if they have a token but not authenticated yet
            if client.auth_token:
                # Try to authenticate them first
                logger.warning(f"Client {client} has token but is not authenticated, attempting auth")
                await self._handle_auth(client, {"type": WSMessageType.AUTH, "token": client.auth_token, "req_id": str(uuid.uuid4())})
                
                # Wait a moment for authentication to complete
                await asyncio.sleep(0.5)
                
                if not client.authenticated:
                    return await self._send_error(client, "not_authenticated", "Authentication required")
            else:
                return await self._send_error(client, "not_authenticated", "Authentication required")
        
        try:
            # Extract request ID if present
            req_id = data.get("req_id")
            
            # Create a response handler
            def balance_response_handler(balance_data):
                if balance_data and balance_data.get("balance"):
                    asyncio.create_task(self._send_message(client, {
                        "type": WSMessageType.BALANCE,
                        "req_id": req_id,
                        "balance": balance_data.get("balance", {}).get("balance"),
                        "currency": balance_data.get("balance", {}).get("currency"),
                        "loginid": balance_data.get("balance", {}).get("loginid"),
                        "is_virtual": bool(balance_data.get("balance", {}).get("is_virtual")),
                    }))
                else:
                    asyncio.create_task(self._send_error(client, "balance_error", "Failed to get balance"))
            
            # Make sure we're using the client's token for this request
            self.trading_client.authorize(client.auth_token)
            
            # Get balance with the custom response handler
            self.trading_client.get_balance(callback=balance_response_handler)
        
        except Exception as e:
            logger.error(f"Error handling balance request: {e}")
            await self._send_error(client, "balance_error", f"Balance error: {str(e)}")
            
        # Return immediately as the response will be sent asynchronously
        return
    
    async def _handle_symbols(self, client: ClientConnection, data: Dict):
        """Handle symbols request"""
        try:
            # Parse as proper message or extract req_id directly
            req_id = data.get("req_id", str(uuid.uuid4()))
            
            # Create a response handler
            def symbols_response_handler(symbols_data):
                if symbols_data and symbols_data.get("active_symbols"):
                    asyncio.create_task(self._send_message(client, {
                        "type": WSMessageType.SYMBOLS,
                        "req_id": req_id,
                        "symbols": symbols_data.get("active_symbols", [])
                    }))
                else:
                    asyncio.create_task(self._send_error(client, "symbols_error", "Failed to get symbols"))
            
            # Make sure we're using the client's token if they're authenticated
            if client.authenticated and client.auth_token:
                self.trading_client.authorize(client.auth_token)
            
            # Get active symbols with the custom response handler
            self.trading_client.get_active_symbols(callback=symbols_response_handler)
            
            # Return immediately as the response will be sent asynchronously
            return
        
        except Exception as e:
            logger.error(f"Error handling symbols request: {e}")
            await self._send_error(client, "symbols_error", f"Symbols error: {str(e)}")
    
    async def _handle_contracts(self, client: ClientConnection, data: Dict):
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
                    asyncio.create_task(self._send_message(client, {
                        "type": WSMessageType.CONTRACTS,
                        "req_id": req_id,
                        "symbol": symbol,
                        "contracts": contracts_data.get("contracts_for", {}).get("available", [])
                    }))
                else:
                    asyncio.create_task(self._send_error(client, "contracts_error", "Failed to get contracts"))
            
            # Make sure we're using the client's token if they're authenticated
            if client.authenticated and client.auth_token:
                self.trading_client.authorize(client.auth_token)
            
            # Get contracts with the custom response handler
            self.trading_client.get_contracts_for_symbol(symbol, callback=contracts_response_handler)
            
            # Return immediately as the response will be sent asynchronously
            return
            
        except Exception as e:
            logger.error(f"Error handling contracts request: {e}")
            await self._send_error(client, "contracts_error", f"Contracts error: {str(e)}")
    
    async def _handle_proposal(self, client: ClientConnection, data: Dict):
        """Handle proposal request"""
        if not client.authenticated:
            # Check if they have a token but not authenticated yet
            if client.auth_token:
                # Try to authenticate them first
                logger.warning(f"Client {client} has token but is not authenticated, attempting auth")
                await self._handle_auth(client, {"type": WSMessageType.AUTH, "token": client.auth_token, "req_id": str(uuid.uuid4())})
                
                # Wait a moment for authentication to complete
                await asyncio.sleep(0.5)
                
                if not client.authenticated:
                    return await self._send_error(client, "not_authenticated", "Authentication required")
            else:
                return await self._send_error(client, "not_authenticated", "Authentication required")
        
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
                    asyncio.create_task(self._send_message(client, {
                        "type": WSMessageType.PROPOSAL,
                        "req_id": req_id,
                        "symbol": symbol,
                        "proposal_id": proposal.get("id"),
                        "ask_price": proposal.get("ask_price"),
                        "payout": proposal.get("payout"),
                        "spot": proposal.get("spot"),
                        "display_value": proposal.get("display_value"),
                        "date_start": proposal.get("date_start"),
                        "date_expiry": proposal.get("date_expiry"),
                    }))
                else:
                    asyncio.create_task(self._send_error(client, "proposal_error", "Failed to get proposal"))
            
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
            await self._send_error(client, "proposal_error", f"Proposal error: {str(e)}")
    
    async def _handle_buy(self, client: ClientConnection, data: Dict):
        """Handle buy request"""
        if not client.authenticated:
            # Check if they have a token but not authenticated yet
            if client.auth_token:
                # Try to authenticate them first
                logger.warning(f"Client {client} has token but is not authenticated, attempting auth")
                await self._handle_auth(client, {"type": WSMessageType.AUTH, "token": client.auth_token, "req_id": str(uuid.uuid4())})
                
                # Wait a moment for authentication to complete
                await asyncio.sleep(0.5)
                
                if not client.authenticated:
                    return await self._send_error(client, "not_authenticated", "Authentication required")
            else:
                return await self._send_error(client, "not_authenticated", "Authentication required")
        
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
                    asyncio.create_task(self._send_message(client, {
                        "type": WSMessageType.BUY,
                        "req_id": req_id,
                        "contract_id": buy.get("contract_id"),
                        "transaction_id": buy.get("transaction_id"),
                        "balance_after": buy.get("balance_after"),
                        "buy_price": buy.get("buy_price"),
                        "purchase_time": buy.get("purchase_time"),
                        "longcode": buy.get("longcode"),
                    }))
                else:
                    error_msg = buy_data.get("error", {}).get("message", "Failed to buy contract")
                    asyncio.create_task(self._send_error(client, "buy_error", error_msg))
            
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
            await self._send_error(client, "buy_error", f"Buy error: {str(e)}")
    
    async def _handle_sell(self, client: ClientConnection, data: Dict):
        """Handle sell request"""
        if not client.authenticated:
            return await self._send_error(client, "not_authenticated", "Authentication required")
        
        try:
            sell_msg = parse_obj_as(SellRequestMessage, data)
            
            sell_data = self.trading_client.sell_contract(
                contract_id=sell_msg.contract_id
            )
            
            if sell_data and sell_data.get("sell"):
                sell = sell_data.get("sell", {})
                await self._send_message(client, {
                    "type": WSMessageType.SELL,
                    "req_id": sell_msg.req_id,
                    "contract_id": sell_msg.contract_id,
                    "transaction_id": sell.get("transaction_id"),
                    "sold_for": sell.get("sold_for"),
                    "remaining_contracts": sell.get("contract_id") != sell_msg.contract_id,
                })
            else:
                error_msg = sell_data.get("error", {}).get("message", "Failed to sell contract")
                await self._send_error(client, "sell_error", error_msg)
        
        except Exception as e:
            logger.error(f"Error handling sell request: {e}")
            await self._send_error(client, "sell_error", f"Sell error: {str(e)}")
    
    async def _handle_portfolio(self, client: ClientConnection, data: Dict):
        """Handle portfolio request"""
        if not client.authenticated:
            return await self._send_error(client, "not_authenticated", "Authentication required")
        
        try:
            portfolio_msg = parse_obj_as(PortfolioRequestMessage, data)
            
            portfolio_data = self.trading_client.get_portfolio()
            
            if portfolio_data and portfolio_data.get("portfolio"):
                await self._send_message(client, {
                    "type": WSMessageType.PORTFOLIO,
                    "req_id": portfolio_msg.req_id,
                    "contracts": portfolio_data.get("portfolio", {}).get("contracts", [])
                })
            else:
                await self._send_error(client, "portfolio_error", "Failed to get portfolio")
        
        except Exception as e:
            logger.error(f"Error handling portfolio request: {e}")
            await self._send_error(client, "portfolio_error", f"Portfolio error: {str(e)}")
    
    async def _handle_profit_table(self, client: ClientConnection, data: Dict):
        """Handle profit table request"""
        if not client.authenticated:
            return await self._send_error(client, "not_authenticated", "Authentication required")
        
        try:
            profit_msg = parse_obj_as(ProfitTableRequestMessage, data)
            
            profit_data = self.trading_client.get_profit_table(
                limit=profit_msg.limit,
                offset=profit_msg.offset,
                start_date=profit_msg.start_date,
                end_date=profit_msg.end_date
            )
            
            if profit_data and profit_data.get("profit_table"):
                profit_table = profit_data.get("profit_table", {})
                await self._send_message(client, {
                    "type": WSMessageType.PROFIT_TABLE,
                    "req_id": profit_msg.req_id,
                    "transactions": profit_table.get("transactions", []),
                    "count": profit_table.get("count", 0)
                })
            else:
                await self._send_error(client, "profit_table_error", "Failed to get profit table")
        
        except Exception as e:
            logger.error(f"Error handling profit table request: {e}")
            await self._send_error(client, "profit_table_error", f"Profit table error: {str(e)}")
    
    async def _handle_statement(self, client: ClientConnection, data: Dict):
        """Handle statement request"""
        if not client.authenticated:
            return await self._send_error(client, "not_authenticated", "Authentication required")
        
        try:
            statement_msg = parse_obj_as(StatementRequestMessage, data)
            
            statement_data = self.trading_client.get_statement(
                limit=statement_msg.limit,
                offset=statement_msg.offset,
                start_date=statement_msg.start_date,
                end_date=statement_msg.end_date
            )
            
            if statement_data and statement_data.get("statement"):
                statement = statement_data.get("statement", {})
                await self._send_message(client, {
                    "type": WSMessageType.STATEMENT,
                    "req_id": statement_msg.req_id,
                    "transactions": statement.get("transactions", []),
                    "count": statement.get("count", 0)
                })
            else:
                await self._send_error(client, "statement_error", "Failed to get statement")
        
        except Exception as e:
            logger.error(f"Error handling statement request: {e}")
            await self._send_error(client, "statement_error", f"Statement error: {str(e)}")
    
    async def _on_tick_data(self, tick_data: Dict):
        """Handle tick data from market stream"""
        try:
            symbol = tick_data.get("tick", {}).get("symbol")
            if not symbol:
                return
            
            subscription_id = f"{symbol}_tick"
            
            if subscription_id in self.subscription_to_clients:
                # Create tick message
                tick_msg = {
                    "type": WSMessageType.TICK,
                    "data": {
                        "symbol": symbol,
                        "price": tick_data.get("tick", {}).get("quote"),
                        "timestamp": tick_data.get("tick", {}).get("epoch"),
                        "pip_size": tick_data.get("tick", {}).get("pip_size", 0.01)
                    }
                }
                
                # Send to all subscribed clients
                for client_id in self.subscription_to_clients[subscription_id]:
                    if client_id in self.connections:
                        await self._send_message(self.connections[client_id], tick_msg)
        
        except Exception as e:
            logger.error(f"Error handling tick data: {e}")
    
    async def _on_ohlc_data(self, ohlc_data: Dict):
        """Handle OHLC data from market stream"""
        try:
            symbol = ohlc_data.get("ohlc", {}).get("symbol")
            interval = ohlc_data.get("ohlc", {}).get("granularity")
            
            if not symbol or not interval:
                return
            
            subscription_id = f"{symbol}_ohlc_{interval}"
            
            if subscription_id in self.subscription_to_clients:
                # Create OHLC message
                ohlc_msg = {
                    "type": WSMessageType.OHLC,
                    "data": {
                        "symbol": symbol,
                        "interval": interval,
                        "open": ohlc_data.get("ohlc", {}).get("open"),
                        "high": ohlc_data.get("ohlc", {}).get("high"),
                        "low": ohlc_data.get("ohlc", {}).get("low"),
                        "close": ohlc_data.get("ohlc", {}).get("close"),
                        "timestamp": ohlc_data.get("ohlc", {}).get("open_time")
                    }
                }
                
                # Send to all subscribed clients
                for client_id in self.subscription_to_clients[subscription_id]:
                    if client_id in self.connections:
                        await self._send_message(self.connections[client_id], ohlc_msg)
        
        except Exception as e:
            logger.error(f"Error handling OHLC data: {e}")
    
    async def _on_portfolio_update(self, portfolio_data: Dict):
        """Handle portfolio updates"""
        try:
            # Send portfolio update to all authenticated clients
            for client_id, client in self.connections.items():
                if client.authenticated:
                    await self._send_message(client, {
                        "type": WSMessageType.PORTFOLIO,
                        "contracts": portfolio_data.get("portfolio", {}).get("contracts", [])
                    })
        
        except Exception as e:
            logger.error(f"Error handling portfolio update: {e}")
    
    async def _send_message(self, client: ClientConnection, message: Dict):
        """Send a WebSocket message to a client"""
        try:
            if not isinstance(message, str):
                message = json.dumps(message)
            
            await client.websocket.send(message)
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection already closed for {client}")
            self._remove_client(client.client_id)
        except Exception as e:
            logger.error(f"Error sending message to {client}: {e}")
    
    async def _send_error(self, client: ClientConnection, code: str, message: str, data: Dict = None):
        """Send an error message to a client"""
        error_msg = ErrorWSMessage(
            type=WSMessageType.ERROR,
            code=code,
            message=message,
            data=data
        )
        await self._send_message(client, error_msg.dict())
    
    async def _send_info_message(self, client: ClientConnection, message: str):
        """Send an info message to a client"""
        await self._send_message(client, {
            "type": WSMessageType.INFO,
            "message": message
        })
    
    async def _heartbeat_loop(self):
        """Send heartbeat messages to all clients"""
        try:
            while self.running:
                timestamp = int(time.time() * 1000)
                
                # Send heartbeat to all clients
                for client_id, client in list(self.connections.items()):
                    try:
                        await self._send_message(client, {
                            "type": WSMessageType.PING,
                            "timestamp": timestamp
                        })
                    except Exception as e:
                        logger.error(f"Error sending heartbeat to {client}: {e}")
                
                # Sleep until next heartbeat
                await asyncio.sleep(30)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in heartbeat loop: {e}")
    
    async def _cleanup_loop(self):
        """Clean up inactive connections"""
        try:
            while self.running:
                current_time = time.time()
                
                # Check all connections
                for client_id, client in list(self.connections.items()):
                    # If inactive for more than 5 minutes, remove
                    if current_time - client.last_activity > 300:  # 5 minutes
                        logger.info(f"Removing inactive client: {client}")
                        try:
                            await client.websocket.close()
                        except:
                            pass
                        self._remove_client(client_id)
                
                # Sleep until next check
                await asyncio.sleep(60)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")