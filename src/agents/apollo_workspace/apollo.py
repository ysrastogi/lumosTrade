from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
from src.llm.client import GeminiClient
from src.agents.apollo_workspace.models import Signal
from src.agents.apollo_workspace.memory_manager import ApolloMemoryManager
from src.agents.apollo_workspace.tools.probability_calculator import ProbabilityCalculator
from src.agents.apollo_workspace.tools.confluence_analyzer import ConfluenceAnalyzer
from src.agents.apollo_workspace.tools.backtest_validator import BacktestValidator
from src.agents.apollo_workspace.tools.signal_generator import SignalGenerator
from src.agents.apollo_workspace.prompts import ApolloPrompts
import json
import logging
logger = logging.getLogger(__name__)

from src.llm.client import gemini

class ApolloAgent:
    """
    LLM-powered signal explanation and narrative generation
    """
    
    def __init__(self, llm_client, memory_system=None, use_redis=True):
        self.llm = gemini
        
        # Initialize memory system
        self.memory_system = memory_system
        if not memory_system:
            self.memory_system = ApolloMemoryManager(use_redis=use_redis)
        
        # Initialize tools
        self.probability_calculator = ProbabilityCalculator(self.memory_system)
        self.confluence_analyzer = ConfluenceAnalyzer()
        self.backtest_validator = BacktestValidator(self.memory_system)
        self.signal_generator = SignalGenerator()
        
    def generate_reasoning(self, signal: Signal, 
                          supporting_factors: List[str] = None,
                          risk_factors: List[str] = None,
                          probabilities: Dict[str, float] = None,
                          market_data: dict = None) -> str:
        """
        Generate comprehensive signal explanation
        """
        # Handle optional parameters
        supporting_factors = supporting_factors or []
        risk_factors = risk_factors or []
        market_data = market_data or {"regime": "unknown", "regime_confidence": 0.5, "summary": ""}
        
        if probabilities is None:
            probabilities = {
                "base_win_rate": 0.5,
                "adjusted_win_rate": 0.5,
                "expected_value": 0.5,
                "sample_size": 0
            }
        
        # If confluence analyzer is available but factors weren't provided, get them
        if not supporting_factors and not risk_factors and self.confluence_analyzer and market_data:
            supporting_factors, risk_factors = self.confluence_analyzer.analyze(signal, market_data)
            
        # Get the prompt from the prompts module
        prompt = ApolloPrompts.reasoning_prompt(
            signal=signal, 
            supporting_factors=supporting_factors, 
            risk_factors=risk_factors,
            probabilities=probabilities, 
            market_data=market_data
        )

        print(f"Generating reasoning with prompt:{prompt}")
        
        try:
            reasoning = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_output_tokens=2000
            )
            print(f"LLM reasoning: {reasoning}")
            
            # Store reasoning in global memory
            if hasattr(self.memory_system, 'store_to_global_memory'):
                # Execute this asynchronously without blocking
                import asyncio
                asyncio.create_task(self.memory_system.store_to_global_memory(
                    memory_type="signal_reasoning",
                    content={
                        "symbol": signal.symbol,
                        "signal_id": signal.id,
                        "timestamp": datetime.now().isoformat(),
                        "reasoning": reasoning,
                        "pattern": signal.pattern,
                        "direction": signal.direction,
                        "supporting_factors": supporting_factors,
                        "risk_factors": risk_factors,
                        "probabilities": probabilities
                    },
                    tags=[signal.symbol, "reasoning", signal.pattern, signal.direction]
                ))
        
            return reasoning
            
        except Exception as e:
            logger.error(f"Error generating LLM reasoning: {str(e)}")
            direction = signal.direction.upper()
            pattern = signal.pattern
            fallback_reasoning = f"This {direction} signal based on {pattern} pattern shows potential with a {probabilities['adjusted_win_rate']*100:.1f}% win rate. Key factors include market alignment with {market_data['regime']} regime. Monitor price action at key levels for confirmation."
            
            # Store fallback reasoning in global memory
            if hasattr(self.memory_system, 'store_to_global_memory'):
                # Execute this asynchronously without blocking
                import asyncio
                asyncio.create_task(self.memory_system.store_to_global_memory(
                    memory_type="signal_reasoning",
                    content={
                        "symbol": signal.symbol,
                        "signal_id": signal.id,
                        "timestamp": datetime.now().isoformat(),
                        "reasoning": fallback_reasoning,
                        "pattern": signal.pattern,
                        "direction": signal.direction,
                        "error": str(e),
                        "is_fallback": True,
                        "supporting_factors": supporting_factors,
                        "risk_factors": risk_factors,
                        "probabilities": probabilities
                    },
                    tags=[signal.symbol, "reasoning", "fallback", signal.pattern, signal.direction]
                ))
            
            return fallback_reasoning
    
    def generate_invalidation_criteria(self, signal: Signal, 
                                      market_data: dict) -> List[str]:
        """
        Identify conditions that would invalidate the setup
        """
        # Get the prompt from the prompts module
        prompt = ApolloPrompts.invalidation_criteria_prompt(
            signal=signal,
            market_data=market_data
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                max_output_tokens=5000,
                temperature=0.7
            )
            
            if not response:
                logger.warning(f"Empty response from LLM when generating invalidation criteria")
            else:
                criteria = self._parse_list_response(response)
                if criteria and len(criteria) >= 3:
                    # Store invalidation criteria in global memory
                    if hasattr(self.memory_system, 'store_to_global_memory'):
                        # Execute this asynchronously without blocking
                        import asyncio
                        asyncio.create_task(self.memory_system.store_to_global_memory(
                            memory_type="signal_invalidation_criteria",
                            content={
                                "symbol": signal.symbol,
                                "signal_id": signal.id,
                                "timestamp": datetime.now().isoformat(),
                                "invalidation_criteria": criteria,
                                "pattern": signal.pattern,
                                "direction": signal.direction,
                                "regime": market_data.get('regime', 'unknown'),
                            },
                            tags=[signal.symbol, "invalidation_criteria", signal.pattern, signal.direction]
                        ))
                    return criteria
                else:
                    logger.warning(f"Insufficient criteria from LLM: {len(criteria) if criteria else 0} (need 3+)")
                
        except Exception as e:
            logger.error(f"Error generating invalidation criteria: {str(e)}")
        
        # Use default criteria as fallback
        direction = signal.direction
        stop_loss = signal.stop_loss
        
        default_criteria = [
            f"Price closes {'below' if direction == 'buy' else 'above'} stop loss at {stop_loss}",
            f"Volume significantly decreases during price movement",
            f"Market regime changes from {market_data.get('regime', 'current')} to opposing condition",
            f"Support/resistance level {'below entry' if direction == 'buy' else 'above entry'} is broken",
            f"Price action forms a reversal pattern"
        ]
        
        # Store default invalidation criteria in global memory
        if hasattr(self.memory_system, 'store_to_global_memory'):
            # Execute this asynchronously without blocking
            import asyncio
            asyncio.create_task(self.memory_system.store_to_global_memory(
                memory_type="signal_invalidation_criteria",
                content={
                    "symbol": signal.symbol,
                    "signal_id": signal.id,
                    "timestamp": datetime.now().isoformat(),
                    "invalidation_criteria": default_criteria,
                    "pattern": signal.pattern,
                    "direction": signal.direction,
                    "regime": market_data.get('regime', 'unknown'),
                    "is_fallback": True
                },
                tags=[signal.symbol, "invalidation_criteria", "fallback", signal.pattern, signal.direction]
            ))
        
        return default_criteria
    
    def compare_to_historical(self, signal: Signal, 
                            similar_signals: List[Signal]) -> str:
        """
        Generate comparison to historical similar setups
        """
        if not similar_signals:
            no_data_message = "No historical data available for comparison."
            
            # Store result in global memory
            if hasattr(self.memory_system, 'store_to_global_memory'):
                # Execute this asynchronously without blocking
                import asyncio
                asyncio.create_task(self.memory_system.store_to_global_memory(
                    memory_type="historical_comparison",
                    content={
                        "symbol": signal.symbol,
                        "signal_id": signal.id,
                        "timestamp": datetime.now().isoformat(),
                        "comparison": no_data_message,
                        "pattern": signal.pattern,
                        "direction": signal.direction,
                        "similar_signals_count": 0,
                        "no_data": True
                    },
                    tags=[signal.symbol, "historical_comparison", "no_data", signal.pattern, signal.direction]
                ))
            
            return no_data_message
        
        # Get the prompt from the prompts module
        prompt = ApolloPrompts.historical_comparison_prompt(
            signal=signal,
            similar_signals=similar_signals
        )
        
        try:
            comparison = self.llm.generate(
                prompt=prompt, 
                max_output_tokens=2000, 
                temperature=0.7
            )
            
            if not comparison:
                logger.warning(f"Empty response from LLM when generating historical comparison")
            
            # Store comparison in global memory
            if hasattr(self.memory_system, 'store_to_global_memory') and comparison:
                # Execute this asynchronously without blocking
                import asyncio
                asyncio.create_task(self.memory_system.store_to_global_memory(
                    memory_type="historical_comparison",
                    content={
                        "symbol": signal.symbol,
                        "signal_id": signal.id,
                        "timestamp": datetime.now().isoformat(),
                        "comparison": comparison,
                        "pattern": signal.pattern,
                        "direction": signal.direction,
                        "similar_signals_count": len(similar_signals),
                        "similar_signals": [s.to_dict() for s in similar_signals[:5]]  # Store first 5 signals only
                    },
                    tags=[signal.symbol, "historical_comparison", signal.pattern, signal.direction]
                ))
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error generating historical comparison: {str(e)}")
            error_message = f"Historical comparison unavailable due to error: {str(e)}"
            
            # Store error in global memory
            if hasattr(self.memory_system, 'store_to_global_memory'):
                # Execute this asynchronously without blocking
                import asyncio
                asyncio.create_task(self.memory_system.store_to_global_memory(
                    memory_type="historical_comparison",
                    content={
                        "symbol": signal.symbol,
                        "signal_id": signal.id,
                        "timestamp": datetime.now().isoformat(),
                        "comparison": error_message,
                        "pattern": signal.pattern,
                        "direction": signal.direction,
                        "error": str(e),
                        "similar_signals_count": len(similar_signals),
                        "is_error": True
                    },
                    tags=[signal.symbol, "historical_comparison", "error", signal.pattern, signal.direction]
                ))
            
            return error_message
    

    
    def _parse_list_response(self, response: str) -> List[str]:
        """Parse LLM list response into Python list"""
        if not response:
            return []
            
        # Simplified parser - would need robust implementation
        lines = [line.strip('- ').strip() for line in response.split('\n') 
                if line.strip() and not line.startswith('[')]
        return [l for l in lines if l]
        
    def generate_signals(self, market_data: dict) -> List[Signal]:
        """
        Generate trading signals using the SignalGenerator tool
        """
        if not self.signal_generator:
            raise ValueError("SignalGenerator not initialized")
        
        # Check if market_data has the expected structure for trade_ideas
        if 'patterns' in market_data and isinstance(market_data['patterns'], list):
            # Create trade_ideas from patterns if not present
            if 'trade_ideas' not in market_data:
                market_data['trade_ideas'] = []
                for pattern in market_data['patterns']:
                    # Extract data from pattern for signal generation
                    bias = pattern.get('bias', 'neutral')
                    direction = 'buy' if bias == 'bullish' else 'sell' if bias == 'bearish' else None
                    
                    # Skip patterns without clear direction
                    if not direction:
                        continue
                    
                    # Get price data for entry/exit points
                    price = market_data.get('features', {}).get('close', 0)
                    if price <= 0 and 'price' in market_data:
                        price = market_data['price']
                    
                    # Skip if we can't determine price
                    if price <= 0:
                        continue
                    
                    # Calculate risk parameters based on ATR or price percentage
                    atr = market_data.get('features', {}).get('atr', price * 0.01)  # Default to 1% if no ATR
                    stop_distance = atr * 2  # 2x ATR for stop loss
                    target_distance = atr * 3  # 3x ATR for take profit
                    
                    # Create trade idea
                    target_price = price + target_distance if direction == 'buy' else price - target_distance
                    stop_price = price - stop_distance if direction == 'buy' else price + stop_distance
                    risk = abs(price - stop_price)
                    reward = abs(price - target_price)
                    risk_reward = reward / risk if risk > 0 else 0
                    
                    trade_idea = {
                        'symbol': market_data.get('symbol', 'UNKNOWN'),
                        'timestamp': market_data.get('timestamp', datetime.now().isoformat()),
                        'pattern': pattern.get('type', 'unknown'),
                        'direction': direction,
                        'confidence': pattern.get('confidence', 50) / 100.0,  # Convert percentage to decimal
                        'entry': price,
                        'stop_loss': stop_price,
                        'target': target_price,  # Renamed from take_profit to target
                        'risk_reward': risk_reward,  # Added risk_reward calculation
                        'timeframe': '1h'  # Default timeframe
                    }
                    
                    market_data['trade_ideas'].append(trade_idea)
                    
        # Call the signal generator with the updated market_data
        signals = self.signal_generator.generate_signals(market_data)
        
        # If no signals were generated but we have patterns with clear bias, create a basic signal
        if not signals and 'patterns' in market_data and isinstance(market_data['patterns'], list):
            bullish_patterns = [p for p in market_data['patterns'] if p.get('bias') == 'bullish']
            bearish_patterns = [p for p in market_data['patterns'] if p.get('bias') == 'bearish']
            
            direction = None
            confidence = 0
            pattern_type = "unknown"
            
            if len(bullish_patterns) > len(bearish_patterns):
                direction = "buy"
                confidence = max([p.get('confidence', 0) for p in bullish_patterns], default=75) / 100.0
                pattern_type = bullish_patterns[0].get('type', 'bullish_pattern')
            elif len(bearish_patterns) > len(bullish_patterns):
                direction = "sell"
                confidence = max([p.get('confidence', 0) for p in bearish_patterns], default=75) / 100.0
                pattern_type = bearish_patterns[0].get('type', 'bearish_pattern')
            
            if direction:
                price = market_data.get('features', {}).get('close', 0)
                if price <= 0 and 'price' in market_data:
                    price = market_data['price']
                    
                if price > 0:
                    atr = market_data.get('features', {}).get('atr', price * 0.01)
                    signal = Signal(
                        symbol=market_data.get('symbol', 'unknown'),
                        direction=direction,
                        entry=price,
                        stop_loss=price - atr*2 if direction == 'buy' else price + atr*2,
                        take_profit=price + atr*3 if direction == 'buy' else price - atr*3,
                        pattern=pattern_type,
                        confidence=confidence,
                        timeframe='1h'
                    )
                    signals = [signal]
        
        # Enrich signals with additional analysis
        for signal in signals:
            self.enrich_signal(signal, market_data)
            
        return signals
    
    def enrich_signal(self, signal: Signal, market_data: dict) -> Signal:
        """
        Enhance a signal with additional analysis using all tools
        
        Note: This method is synchronous for simplicity, so we don't use memory-based lookups
        that would require async calls
        """
        # Get supporting and risk factors
        if self.confluence_analyzer:
            supporting_factors, risk_factors = self.confluence_analyzer.analyze(signal, market_data)
        else:
            supporting_factors, risk_factors = [], []
            
        # Use default probabilities since memory access requires async
        probabilities = {
            "base_win_rate": 0.5,
            "adjusted_win_rate": 0.5,
            "expected_value": 0.5,
            "sample_size": 0
        }

        print(f"{supporting_factors=}, {risk_factors=}, {probabilities=}, {market_data=}")

            
        # Generate reasoning using LLM
        signal.reasoning = self.generate_reasoning(
            signal, supporting_factors, risk_factors, probabilities, market_data
        )
        
        # Generate invalidation criteria
        signal.invalidation_criteria = self.generate_invalidation_criteria(signal, market_data)
        
        # Add supporting factors
        signal.supporting_factors = supporting_factors
        signal.risk_factors = risk_factors
        
        # Use default historical metrics to avoid async calls
        signal.similar_historical_count = 0
        signal.historical_win_rate = 0.5
        
        # Store enriched signal data in global memory
        if hasattr(self.memory_system, 'store_to_global_memory'):
            # Execute this asynchronously without blocking
            import asyncio
            asyncio.create_task(self.memory_system.store_to_global_memory(
                memory_type="enriched_signal",
                content={
                    "symbol": signal.symbol,
                    "signal_id": signal.id,
                    "timestamp": datetime.now().isoformat(),
                    "pattern": signal.pattern,
                    "direction": signal.direction,
                    "entry": signal.entry,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "confidence": signal.confidence,
                    "timeframe": signal.timeframe,
                    "reasoning": signal.reasoning,
                    "invalidation_criteria": signal.invalidation_criteria,
                    "supporting_factors": supporting_factors,
                    "risk_factors": risk_factors,
                    "probabilities": probabilities,
                    "market_regime": market_data.get('regime', 'unknown'),
                    "market_summary": market_data.get('summary', ''),
                    "enriched_data": True
                },
                tags=[signal.symbol, "enriched_signal", signal.pattern, signal.direction]
            ))
        
        return signal
    
    def validate_signal(self, signal: Signal, market_data: dict) -> Dict[str, any]:
        """
        Validate a signal using the BacktestValidator tool
        """
        if not self.backtest_validator:
            return {"valid": True, "reason": "Backtest validator not available"}
            
        return self.backtest_validator.validate_signal(signal, market_data)
        
    def run_analysis_pipeline(self, market_data: dict) -> List[Signal]:
        """
        Run the complete analysis pipeline:
        1. Generate signals
        2. Enrich with confluence analysis and LLM reasoning
        
        Returns a list of fully analyzed trading signals
        """
        # Generate initial signals
        signals = self.generate_signals(market_data)
        
        # Skip validation to avoid async calls
        valid_signals = signals
                
        # Enrich all valid signals
        for signal in valid_signals:
            self.enrich_signal(signal, market_data)
            
        # Store overall analysis results in global memory
        if hasattr(self.memory_system, 'store_to_global_memory') and valid_signals:
            # Execute this asynchronously without blocking
            import asyncio
            asyncio.create_task(self.memory_system.store_to_global_memory(
                memory_type="apollo_analysis_results",
                content={
                    "symbol": market_data.get('symbol', 'unknown'),
                    "timestamp": datetime.now().isoformat(),
                    "market_regime": market_data.get('regime', 'unknown'),
                    "regime_confidence": market_data.get('regime_confidence', 0.5),
                    "analysis_summary": f"Generated {len(valid_signals)} signals for {market_data.get('symbol', 'unknown')}",
                    "signal_count": len(valid_signals),
                    "signals": [s.to_dict() for s in valid_signals],
                    "market_context": {
                        "regime": market_data.get('regime', 'unknown'),
                        "summary": market_data.get('summary', ''),
                        "patterns": market_data.get('patterns', []),
                        "trading_bias": market_data.get('trading_bias', 'neutral'),
                    }
                },
                tags=[market_data.get('symbol', 'unknown'), "analysis_results", "apollo"]
            ))
            
        return valid_signals
        
    async def get_athena_context(self, symbol: str) -> Dict:
        """
        Retrieve Athena's latest market context for a symbol
        
        Args:
            symbol: Market symbol to analyze
            
        Returns:
            Market context from Athena's memory
        """
        if not hasattr(self.memory_system, 'get_market_context'):
            raise ValueError("Memory system does not support get_market_context method")
        
        # Get the market context from memory
        market_context = await self.memory_system.get_market_context(symbol)
        
        return market_context
    
    async def process_athena_observation(self, symbol: str, athena_context=None) -> List[Signal]:
        """
        Process the latest Athena observation for a symbol and generate signals
        
        Args:
            symbol: Market symbol to analyze
            athena_context: Optional direct Athena context (for fallback)
            
        Returns:
            List of analyzed signals
        """
        # Get market context from Athena's memory using the direct method
        
        # If we were provided direct context, use it (fallback mechanism)
        if isinstance(athena_context, str):
            athena_context = json.loads(athena_context)

        if athena_context:
            # Handle different formats of context that might come from memory or direct
            if isinstance(athena_context, dict):
                # Extract market data from various possible formats
                
                # Format 1: Direct Athena observation format
                if 'regime' in athena_context and 'symbol' in athena_context:
                    market_data = {
                        "symbol": symbol,
                        "timestamp": athena_context.get("timestamp", datetime.now().isoformat()),
                        "regime": athena_context.get("regime", "unknown"),
                        "regime_confidence": athena_context.get("regime_confidence", 0.5),
                        "patterns": athena_context.get("patterns", []),
                        "trading_bias": athena_context.get("trading_bias", {}),
                        "summary": athena_context.get("summary", ""),
                        "indicators": athena_context.get("indicators", {}),
                        "features": athena_context.get("features", {})
                    }
                
                # Format 2: Memory context with embedded observations
                elif 'agent_observations' in athena_context:
                    # Look for observations in any agent, prioritizing Athena
                    found_obs = None
                    for agent_id, observations in athena_context.get('agent_observations', {}).items():
                        if observations and len(observations) > 0:
                            # Filter observations for this symbol
                            symbol_observations = [
                                obs for obs in observations if obs.get('symbol') == symbol
                            ]
                            if symbol_observations:
                                # Use the most recent observation
                                symbol_observations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                                found_obs = symbol_observations[0]
                                logger.info(f"Using observation from agent {agent_id} for {symbol}")
                                break
                    
                    if found_obs:
                        market_data = {
                            "symbol": symbol,
                            "timestamp": found_obs.get("timestamp", datetime.now().isoformat()),
                            "regime": found_obs.get("regime", "unknown"),
                            "regime_confidence": found_obs.get("regime_confidence", 0.5),
                            "patterns": found_obs.get("patterns", []),
                            "trading_bias": found_obs.get("trading_bias", {}),
                            "summary": found_obs.get("summary", ""),
                            "indicators": found_obs.get("indicators", {}),
                            "features": found_obs.get("features", {})
                        }
                    else:
                        # No valid observation found, create an empty market data
                        market_data = {
                            "symbol": symbol,
                            "timestamp": datetime.now().isoformat(),
                            "regime": "unknown",
                            "regime_confidence": 0.5,
                            "patterns": [],
                            "trading_bias": "neutral",
                            "summary": "",
                            "indicators": {},
                            "features": {}
                        }
                else:
                    # Default format
                    market_data = {
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "regime": athena_context.get("regime", "unknown"),
                        "regime_confidence": athena_context.get("regime_confidence", 0.5),
                        "patterns": athena_context.get("patterns", []),
                        "trading_bias": athena_context.get("trading_bias", "neutral"),
                        "summary": athena_context.get("summary", ""),
                        "indicators": athena_context.get("indicators", {}),
                        "features": athena_context.get("features", {})
                    }
            else:
                # Not a dict, just create a basic market data
                market_data = {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "regime": "unknown",
                    "regime_confidence": 0.5,
                    "patterns": [],
                    "trading_bias": "neutral",
                    "summary": "",
                    "indicators": {},
                    "features": {}
                }

            logger.info(f"Formatted market data for {symbol}: {market_data}")
        else:
            # Otherwise try to get it from memory
            logger.info(f"Apollo retrieving market context for {symbol}...")
            # Use the athena_demo agent ID to specifically target Athena's observations
            market_data = await self.memory_system.get_market_context(symbol, agent_ids=["athena_demo"])
            
            # Debug the received market data
            if not market_data:
                logger.error(f"No market data received for {symbol}")
                raise ValueError(f"Failed to get market context for {symbol}: No data found")
            elif 'error' in market_data:
                logger.error(f"Error in market data for {symbol}: {market_data['error']}")
                raise ValueError(f"Failed to get market context for {symbol}: {market_data['error']}")
        
        # Log successful retrieval
        logger.info(f"Using market context for {symbol} with {len(market_data)} data points")
        
        # Check if patterns are present in the market data
        if 'patterns' not in market_data or not market_data['patterns']:
            logger.warning(f"No patterns found in market data for {symbol}, trying to create patterns from available data")
            
            # First check if we can extract patterns from the summary
            if 'summary' in market_data and market_data['summary']:
                summary = market_data['summary'].lower()
                patterns = []
                
                # Extract volatility squeeze pattern if mentioned
                if 'squeeze' in summary or 'volatility squeeze' in summary or 'bollinger squeeze' in summary:
                    squeeze_pattern = {
                        "type": "volatility_squeeze",
                        "confidence": 90,
                        "description": "Bollinger squeeze - breakout imminent",
                        "bias": "neutral",
                        "key_metrics": {"bb_width": market_data.get('features', {}).get('bb_width', 0.001)}
                    }
                    patterns.append(squeeze_pattern)
                    logger.info(f"Extracted volatility squeeze pattern from summary")
                
                # Extract mean reversion pattern if mentioned
                if 'reversion' in summary or 'mean reversion' in summary:
                    # Determine bias from summary
                    bias = 'neutral'
                    if 'bullish' in summary:
                        bias = 'bullish'
                    elif 'bearish' in summary:
                        bias = 'bearish'
                        
                    reversion_pattern = {
                        "type": "mean_reversion",
                        "confidence": 80,
                        "description": "Price extended from mean - potential reversion",
                        "bias": bias,
                        "key_metrics": {"distance_from_mean": 100.0}
                    }
                    patterns.append(reversion_pattern)
                    logger.info(f"Extracted mean reversion pattern from summary with {bias} bias")
                
                if patterns:
                    market_data['patterns'] = patterns
            
            # If still no patterns, create a default pattern based on regime
            if not market_data.get('patterns') and 'regime' in market_data and market_data['regime']:
                regime = market_data['regime']
                regime_confidence = market_data.get('regime_confidence', 0.65)
                trading_bias = market_data.get('trading_bias', 'neutral')
                
                if isinstance(trading_bias, dict):
                    trading_bias = trading_bias.get('bias', 'neutral')
                
                # Create a basic pattern based on regime
                default_pattern = {
                    "type": f"{regime}_trend",
                    "confidence": regime_confidence * 100,
                    "description": f"{regime.capitalize()} market condition detected",
                    "bias": trading_bias if trading_bias != 'neutral' else 'bullish' if 'bull' in regime else 'bearish' if 'bear' in regime else 'neutral'
                }
                
                market_data['patterns'] = [default_pattern]
                logger.info(f"Created default pattern based on regime: {default_pattern['type']}")
            
        # Check if we have any summary data that mentions a pattern
        elif 'summary' in market_data and market_data['summary']:
            summary = market_data['summary']
            if 'mean reversion' in summary.lower() and not any('mean_reversion' in p.get('type', '') for p in market_data.get('patterns', [])):
                # Extract confidence if available
                import re
                confidence_match = re.search(r'(\d+)%\s*confidence', summary.lower())
                confidence = float(confidence_match.group(1)) if confidence_match else 75.0
                
                # Determine bias
                bias = 'neutral'
                if 'bullish' in summary.lower():
                    bias = 'bullish'
                elif 'bearish' in summary.lower():
                    bias = 'bearish'
                
                # Create pattern
                mean_reversion_pattern = {
                    "type": "mean_reversion",
                    "confidence": confidence,
                    "description": "Mean reversion pattern detected from market summary",
                    "bias": bias
                }
                
                # Add to patterns
                if 'patterns' not in market_data:
                    market_data['patterns'] = []
                market_data['patterns'].append(mean_reversion_pattern)
                logger.info(f"Added mean reversion pattern from summary with {confidence}% confidence")
        
        signals = self.run_analysis_pipeline(market_data)
        
        # Store the signal analyses in memory
        if hasattr(self.memory_system, 'store_signal_analysis'):
            for signal in signals:
                analysis = {
                    "reasoning": signal.reasoning,
                    "invalidation_criteria": signal.invalidation_criteria,
                    "supporting_factors": signal.supporting_factors,
                    "confidence": signal.confidence,
                    "timestamp": datetime.now().isoformat()
                }
                await self.memory_system.store_signal_analysis(signal, analysis)
                
                # Additionally store in global memory with more comprehensive data
                if hasattr(self.memory_system, 'store_to_global_memory'):
                    await self.memory_system.store_to_global_memory(
                        memory_type="comprehensive_signal_analysis",
                        content={
                            "symbol": signal.symbol,
                            "signal_id": signal.id,
                            "timestamp": datetime.now().isoformat(),
                            "pattern": signal.pattern,
                            "direction": signal.direction,
                            "entry": signal.entry,
                            "stop_loss": signal.stop_loss,
                            "take_profit": signal.take_profit,
                            "confidence": signal.confidence,
                            "timeframe": signal.timeframe,
                            "reasoning": signal.reasoning,
                            "invalidation_criteria": signal.invalidation_criteria,
                            "supporting_factors": signal.supporting_factors,
                            "risk_factors": signal.risk_factors if hasattr(signal, 'risk_factors') else [],
                            "similar_historical_count": signal.similar_historical_count if hasattr(signal, 'similar_historical_count') else 0,
                            "historical_win_rate": signal.historical_win_rate if hasattr(signal, 'historical_win_rate') else 0.5,
                            "market_context": {
                                "regime": market_data.get('regime', 'unknown'),
                                "summary": market_data.get('summary', ''),
                                "patterns": market_data.get('patterns', []),
                            },
                            "analysis_type": "athena_observation_processing",
                            "complete_signal_data": signal.to_dict()
                        },
                        tags=[signal.symbol, "signal_analysis", "comprehensive", signal.pattern, signal.direction]
                    )
        
        # Store overall observation processing results in global memory
        if hasattr(self.memory_system, 'store_to_global_memory'):
            await self.memory_system.store_to_global_memory(
                memory_type="athena_observation_processing",
                content={
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "market_regime": market_data.get('regime', 'unknown'),
                    "regime_confidence": market_data.get('regime_confidence', 0.5),
                    "signal_count": len(signals),
                    "signals_summary": [
                        {
                            "id": s.id,
                            "pattern": s.pattern,
                            "direction": s.direction,
                            "confidence": s.confidence
                        } for s in signals
                    ],
                    "market_summary": market_data.get('summary', 'No summary available'),
                    "patterns_detected": market_data.get('patterns', []),
                    "trading_bias": market_data.get('trading_bias', 'neutral'),
                },
                tags=[symbol, "observation_processing", "apollo_analysis"]
            )
        
        return signals
        
    async def get_memory_context(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the full memory context for analysis, including cross-agent intelligence.
        
        Args:
            symbol: Optional symbol to filter context by
            
        Returns:
            Dict: Memory context including cross-agent observations and insights
        """
        if not hasattr(self.memory_system, 'initialize'):
            return {"error": "Memory system not properly initialized"}
            
        try:
            # Ensure memory system is initialized
            if not self.memory_system._initialized:
                await self.memory_system.initialize()
                
            # Get raw context
            context = await self.memory_system.get_athena_context(symbol=symbol)
            
            # Get cross-agent observations and insights
            cross_agent_observations = await self.memory_system.recall_cross_agent_observations(
                symbol=symbol, limit=5
            )
            cross_agent_insights = await self.memory_system.recall_cross_agent_insights(
                symbol=symbol, limit=10
            )
            
            # Get global memory context for enhanced intelligence
            global_context = await self.memory_system.get_global_memory_context(symbol=symbol)
            
            # Format the response for better usability
            formatted_context = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "agent_id": self.memory_system.agent_id,
                # Memory context
                "memory_context": context,
                # Cross-agent intelligence
                "global_context": global_context,
                "cross_agent_observations": cross_agent_observations,
                "cross_agent_insights": cross_agent_insights,
                # Summary statistics
                "intelligence_summary": {
                    "cross_agent_observations": sum(len(obs) for obs in cross_agent_observations.values()),
                    "cross_agent_insights": sum(len(ins) for ins in cross_agent_insights.values()),
                    "total_agents_with_data": len(set(cross_agent_observations.keys()) | set(cross_agent_insights.keys())),
                    "enhanced_intelligence": len(cross_agent_observations) + len(cross_agent_insights) > 0
                }
            }
            
            # Store this comprehensive memory context in global memory for other agents
            if hasattr(self.memory_system, 'store_to_global_memory'):
                # Store only a summarized version to avoid massive memory usage
                summary_context = {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": self.memory_system.agent_id,
                    "context_type": "memory_context_summary",
                    "agents_with_observations": list(cross_agent_observations.keys()),
                    "agents_with_insights": list(cross_agent_insights.keys()),
                    "observation_counts": {agent: len(obs) for agent, obs in cross_agent_observations.items()},
                    "insight_counts": {agent: len(ins) for agent, ins in cross_agent_insights.items()},
                    "has_athena_context": bool(context),
                    "has_global_context": bool(global_context),
                    "total_observations": sum(len(obs) for obs in cross_agent_observations.values()),
                    "total_insights": sum(len(ins) for ins in cross_agent_insights.values()),
                }
                
                await self.memory_system.store_to_global_memory(
                    memory_type="memory_context_summary",
                    content=summary_context,
                    tags=[symbol if symbol else "all", "memory_context", "apollo_analysis"]
                )
            
            return formatted_context
            
        except Exception as e:
            logger.error(f"Error retrieving memory context: {str(e)}")
            error_context = {
                "error": f"Memory error: {str(e)}",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "agent_id": self.memory_system.agent_id if hasattr(self.memory_system, 'agent_id') else "unknown"
            }
            
            # Store error in global memory
            if hasattr(self.memory_system, 'store_to_global_memory'):
                try:
                    await self.memory_system.store_to_global_memory(
                        memory_type="memory_error",
                        content=error_context,
                        tags=[symbol if symbol else "all", "error", "memory_context"]
                    )
                except Exception as store_error:
                    logger.error(f"Error storing memory error: {str(store_error)}")
            
            return error_context