"""
Signal formatting utilities to ensure consistent display of signals
across different components (Apollo, Memory, UI).
"""
from typing import Dict, Any, Union, List
from src.agents.apollo_workspace.models import Signal

def normalize_signal_dict(signal: Union[Signal, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize a signal (either Signal object or dictionary) to a dictionary 
    with consistent keys for display in the UI.
    
    This resolves inconsistencies between the Signal class properties and
    the expected dictionary keys in ApolloFlow._format_signal.
    
    Args:
        signal: A Signal object or dictionary with signal data
        
    Returns:
        A normalized dictionary with consistent keys
    """
    if isinstance(signal, Signal):
        # Convert Signal object to normalized dictionary
        signal_dict = {
            "symbol": signal.symbol,
            "direction": signal.direction.upper(),
            "entry_price": signal.entry if hasattr(signal, 'entry') else getattr(signal, 'entry_price', 0),
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit if signal.take_profit is not None else getattr(signal, 'target', 0),
            "confidence": signal.confidence * 100 if signal.confidence <= 1.0 else signal.confidence,
            "pattern": signal.pattern,
            "timeframe": getattr(signal, 'timeframe', '1h'),
            "timestamp": getattr(signal, 'timestamp', None)
        }
        
        # Add reasoning if available
        if hasattr(signal, 'reasoning') and signal.reasoning:
            signal_dict["reasoning"] = signal.reasoning
            
        # Add invalidation criteria if available
        if hasattr(signal, 'invalidation_criteria') and signal.invalidation_criteria:
            signal_dict["invalidation_criteria"] = signal.invalidation_criteria
            
        return signal_dict
    else:
        # If already a dictionary, ensure consistent keys
        signal_dict = signal.copy()  # Create a copy to avoid modifying original
        
        # Ensure entry_price is set correctly
        if 'entry' in signal_dict and 'entry_price' not in signal_dict:
            signal_dict['entry_price'] = signal_dict['entry']
            
        # Ensure direction is uppercase
        if 'direction' in signal_dict and isinstance(signal_dict['direction'], str):
            signal_dict['direction'] = signal_dict['direction'].upper()
            
        # Ensure confidence is a percentage (0-100 range)
        if 'confidence' in signal_dict and signal_dict['confidence'] <= 1.0:
            signal_dict['confidence'] = signal_dict['confidence'] * 100
            
        return signal_dict

def normalize_signal_list(signals: List[Union[Signal, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Normalize a list of signals to dictionaries with consistent keys.
    
    Args:
        signals: List of Signal objects or dictionaries
        
    Returns:
        List of normalized dictionaries
    """
    return [normalize_signal_dict(signal) for signal in signals]