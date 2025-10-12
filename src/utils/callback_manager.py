"""
MarketStream and WebSocketServer callback integration
"""

from typing import Dict, Any, Callable, List, Set, Optional

class CallbackManager:
    """
    Helper class to manage callbacks for market data streams
    """
    
    def __init__(self):
        """Initialize the callback manager"""
        self.callbacks: Dict[str, List[Callable]] = {}
    
    def add_callback(self, event_type: str, callback: Callable) -> None:
        """Add a callback function for a specific event type"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        
        self.callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: str, callback: Callable) -> bool:
        """Remove a callback function for a specific event type"""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
            return True
        return False
    
    def trigger_callbacks(self, event_type: str, data: Any) -> None:
        """Trigger all callbacks for a specific event type"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    # Check if the callback is a coroutine function (async)
                    import asyncio
                    import inspect
                    
                    if inspect.iscoroutinefunction(callback):
                        # Create a new event loop to run async callbacks from sync contexts
                        # This is essential for callbacks like WebSocketServer._on_portfolio_update
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # If we're already in an event loop, create a future
                                asyncio.create_task(callback(data))
                            else:
                                # If no event loop is running, create one temporarily
                                asyncio.run(callback(data))
                        except RuntimeError:
                            # If we're outside of an event loop and can't get one
                            asyncio.run(callback(data))
                    else:
                        # Regular synchronous callback
                        callback(data)
                except Exception as e:
                    # Log but don't crash on callback errors
                    import logging
                    logging.getLogger(__name__).error(f"Error in callback for {event_type}: {e}")