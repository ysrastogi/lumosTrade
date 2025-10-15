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

        
            return reasoning
            
        except Exception as e:
            # Fallback in case of LLM error
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating LLM reasoning: {str(e)}")
            
            # Return a fallback response
            direction = signal.direction.upper()
            pattern = signal.pattern
            return f"This {direction} signal based on {pattern} pattern shows potential with a {probabilities['adjusted_win_rate']*100:.1f}% win rate. Key factors include market alignment with {market_data['regime']} regime. Monitor price action at key levels for confirmation."
    
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
                max_output_tokens=2000,
                temperature=0.7
            )
            
            # Check if response is valid
            if not response:
                logger = logging.getLogger(__name__)
                logger.warning(f"Empty response from LLM when generating invalidation criteria")
                # Continue to fallback criteria
            else:
                # Parse the response into a list of criteria
                criteria = self._parse_list_response(response)
                
                # If we got valid criteria, return them
                if criteria and len(criteria) >= 3:
                    return criteria
                else:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Insufficient criteria from LLM: {len(criteria) if criteria else 0} (need 3+)")
                
        except Exception as e:
            # Log error but continue with default criteria
            import logging
            logger = logging.getLogger(__name__)
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
        
        return default_criteria
    
    def compare_to_historical(self, signal: Signal, 
                            similar_signals: List[Signal]) -> str:
        """
        Generate comparison to historical similar setups
        """
        if not similar_signals:
            return "No historical data available for comparison."
        
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
            
            # Check if response is valid
            if not comparison:
                logger = logging.getLogger(__name__)
                logger.warning(f"Empty response from LLM when generating historical comparison")
                # Continue to fallback response
            else:
                return comparison
            
        except Exception as e:
            # Provide fallback response
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating historical comparison: {str(e)}")
            
            # Count wins and losses for simple stats
            wins = [s for s in similar_signals if s.outcome == 'win']
            losses = [s for s in similar_signals if s.outcome == 'loss']
            win_rate = len(wins) / len(similar_signals) if similar_signals else 0
            
            return f"Historical analysis shows a {win_rate*100:.1f}% win rate across {len(similar_signals)} similar signals. This setup aligns with historical patterns that have shown positive results. Always monitor key price levels for confirmation."
    
    # Removed _build_reasoning_prompt as it's now handled by the prompts module
    
    def _parse_list_response(self, response: str) -> List[str]:
        """Parse LLM list response into Python list"""
        # Check if response is None or empty
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
            
        signals = self.signal_generator.generate_signals(market_data)
        
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
        
        # Use default historical metrics to avoid async calls
        signal.similar_historical_count = 0
        signal.historical_win_rate = 0.5
        
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
        import logging
        logger = logging.getLogger(__name__)
        
        # If we were provided direct context, use it (fallback mechanism)
        if isinstance(athena_context, str):
            athena_context = json.loads(athena_context)

        if athena_context:
            # logger.info(f"Using provided direct Athena context for {symbol}")
            market_data = {
                "symbol": symbol,
                "timestamp": athena_context.get("timestamp", datetime.now().isoformat()),
                "regime": athena_context.get("regime", "unknown"),
                "regime_confidence": athena_context.get("regime_confidence", 0.5),
                "patterns": athena_context.get("patterns", []),
                "trading_bias": athena_context.get("trading_bias", "neutral"),
                "summary": athena_context.get("summary", ""),
                "indicators": athena_context.get("indicators", {}),
                "features": athena_context.get("features", {})
            }

            print(f"Formatted market data for {symbol}: {market_data}")
        else:
            # Otherwise try to get it from memory
            logger.info(f"Apollo retrieving market context for {symbol}...")
            market_data = await self.memory_system.get_market_context(symbol)
            
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
            logger.warning(f"No patterns found in market data for {symbol}, trying to create a default pattern")
            # Create a default pattern based on available data
            if 'regime' in market_data and market_data['regime']:
                regime = market_data['regime']
                regime_confidence = market_data.get('regime_confidence', 0.65)
                trading_bias = market_data.get('trading_bias', 'neutral')
                
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
            
            return formatted_context
            
        except Exception as e:
            import logging
            import traceback
            logger = logging.getLogger(__name__)
            logger.error(f"Error retrieving memory context: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Memory error: {str(e)}"}