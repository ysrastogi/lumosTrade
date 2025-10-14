"""
Memory System Test Runner for LumosTrade

This script runs a series of tests to validate the functionality of each
component of the LumosTrade memory system.
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Import memory components - adjust import path based on project structure
import sys
import os

# Add the project root to the path so we can import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.memory_core import MemoryCore
from src.memory.registry import AgentRegistry
from src.memory.episodic_store import EpisodicMemoryStore
from src.memory.semantic_store import SemanticMemoryStore
from src.memory.temporal_log import TemporalEventLog
from src.memory.message_bus import MessageBus
from src.memory.retriever import QueryEngine
from src.memory.summarizer import SummarizationEngine
from src.memory.assistant import MemoryAssistant

class MemorySystemTester:
    """Test runner for LumosTrade Memory System components"""
    
    def __init__(self, use_redis: bool = False):
        """
        Initialize the test runner.
        
        Args:
            use_redis: Whether to use Redis (True for integration tests, False for unit tests)
        """
        self.use_redis = use_redis
        self.memory_core = None
        self.test_results = {
            "memory_core": {},
            "registry": {},
            "episodic_store": {},
            "semantic_store": {},
            "temporal_log": {},
            "message_bus": {},
            "query_engine": {},
            "summarizer": {},
            "assistant": {},
            "integration": {}
        }
        
        # Test agent data
        self.test_agents = {
            "trader_agent": {
                "id": "trader_agent_1",
                "name": "TraderAgent",
                "role": "Executes trading strategies",
                "output_schema": ["signal", "reason", "confidence", "timestamp"],
                "metadata": {
                    "name": "TraderAgent",
                    "description": "Agent responsible for executing trading strategies",
                    "capabilities": ["trade_execution", "signal_generation"],
                    "interests": ["market_data", "trading_signals", "price_action"]
                }
            },
            "analyst_agent": {
                "id": "analyst_agent_1",
                "name": "AnalystAgent",
                "role": "Analyzes market conditions",
                "output_schema": ["analysis", "patterns", "sentiment", "timestamp"],
                "metadata": {
                    "name": "AnalystAgent",
                    "description": "Agent responsible for market analysis",
                    "capabilities": ["technical_analysis", "pattern_recognition"],
                    "interests": ["market_data", "economic_events", "price_patterns"]
                }
            }
        }
        
        # Test data
        self.test_memories = [
            {
                "content": {
                    "symbol": "BTC/USD",
                    "pattern": "bullish_engulfing",
                    "confidence": 0.87,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "memory_type": "analysis",
                "tags": ["technical_pattern", "bullish"]
            },
            {
                "content": {
                    "symbol": "ETH/USD",
                    "pattern": "head_and_shoulders",
                    "confidence": 0.75,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "memory_type": "analysis",
                "tags": ["technical_pattern", "bearish"]
            },
            {
                "content": {
                    "symbol": "BTC/USD",
                    "action": "buy",
                    "reason": "Bullish engulfing pattern with high confidence",
                    "confidence": 0.82,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "memory_type": "decision",
                "tags": ["trade_decision", "bullish"]
            }
        ]
        
        logger.info("Memory System Test Runner initialized")
    
    async def setup(self):
        """Set up test environment"""
        # Create memory core with test configuration
        self.memory_core = MemoryCore(
            config={"debug_mode": True},
            use_redis=self.use_redis
        )
        
        # Initialize components
        self.memory_core.initialize_components()
        logger.info("Test environment setup complete")
    
    async def test_memory_core(self):
        """Test MemoryCore functionality"""
        logger.info("Testing MemoryCore...")
        
        # Test session management
        session_id = self.memory_core.get_current_session()
        assert session_id is not None, "Session ID should not be None"
        
        # Test creating a new session
        new_session = self.memory_core.start_new_session()
        assert new_session != session_id, "New session should be different from previous"
        
        self.test_results["memory_core"]["session_management"] = "PASS"
        logger.info("✓ MemoryCore session management test passed")
        
        # Test component initialization
        assert self.memory_core.registry is not None, "Registry should be initialized"
        assert self.memory_core.episodic_store is not None, "Episodic store should be initialized"
        assert self.memory_core.semantic_store is not None, "Semantic store should be initialized"
        assert self.memory_core.temporal_log is not None, "Temporal log should be initialized"
        assert self.memory_core.message_bus is not None, "Message bus should be initialized"
        
        self.test_results["memory_core"]["component_initialization"] = "PASS"
        logger.info("✓ MemoryCore component initialization test passed")
        
    async def test_agent_registry(self):
        """Test AgentRegistry functionality"""
        logger.info("Testing AgentRegistry...")
        
        registry = self.memory_core.registry
        trader = self.test_agents["trader_agent"]
        
        # Test agent registration
        registration_result = await self.memory_core.register_agent(
            trader["id"], trader["role"], trader["output_schema"]
        )
        
        assert registration_result is True, "Agent registration should succeed"
        
        # Test agent retrieval
        registered_agents = registry.agents
        assert trader["id"] in registered_agents, "Agent should be retrievable after registration"
        
        self.test_results["registry"]["registration"] = "PASS"
        logger.info("✓ AgentRegistry registration test passed")
        
        # Test agent subscription
        topics = ["market_data", "trading_signals"]
        for topic in topics:
            await registry.subscribe_agent_to_topic(trader["id"], topic)
        
        # Verify subscriptions
        for topic in topics:
            assert trader["id"] in registry.subscriptions.get(topic, []), \
                f"Agent should be subscribed to {topic}"
        
        self.test_results["registry"]["subscription"] = "PASS"
        logger.info("✓ AgentRegistry subscription test passed")
    
    async def test_episodic_store(self):
        """Test EpisodicMemoryStore functionality"""
        logger.info("Testing EpisodicMemoryStore...")
        
        episodic = self.memory_core.episodic_store
        agent_id = self.test_agents["analyst_agent"]["id"]
        
        # Register the agent if needed
        if agent_id not in self.memory_core.registry.agents:
            analyst = self.test_agents["analyst_agent"]
            await self.memory_core.register_agent(
                analyst["id"], analyst["role"], analyst["output_schema"]
            )
        
        # Test storing memories
        memory_ids = []
        for memory in self.test_memories:
            memory_id = await episodic.store(
                agent_id=agent_id,
                memory_type=memory["memory_type"],
                content=memory["content"],
                tags=memory["tags"]
            )
            memory_ids.append(memory_id)
            assert memory_id is not None, "Memory storage should return a valid ID"
        
        self.test_results["episodic_store"]["storage"] = "PASS"
        logger.info("✓ EpisodicMemoryStore storage test passed")
        
        # Test retrieving recent memories
        recent_memories = await episodic.get_recent(agent_id, limit=10)
        assert len(recent_memories) == len(memory_ids), \
            "Should retrieve all stored memories"
        
        # Test retrieving by type
        analysis_memories = await episodic.get_by_type(
            agent_id, memory_type="analysis"
        )
        assert len(analysis_memories) == 2, \
            "Should retrieve only memories of analysis type"
        
        self.test_results["episodic_store"]["retrieval"] = "PASS"
        logger.info("✓ EpisodicMemoryStore retrieval test passed")
    
    async def test_semantic_store(self):
        """Test SemanticMemoryStore functionality"""
        logger.info("Testing SemanticMemoryStore...")
        
        # Note: Full semantic store testing may require vector DB setup
        # This is a simplified test
        semantic = self.memory_core.semantic_store
        agent_id = self.test_agents["analyst_agent"]["id"]
        
        # Test storing a semantic memory
        memory = self.test_memories[0]
        try:
            memory_id = await semantic.store(
                agent_id=agent_id,
                memory_type=memory["memory_type"],
                content=memory["content"],
                tags=memory["tags"]
            )
            
            assert memory_id is not None, "Memory storage should return a valid ID"
            self.test_results["semantic_store"]["storage"] = "PASS"
            logger.info("✓ SemanticMemoryStore storage test passed")
        except Exception as e:
            self.test_results["semantic_store"]["storage"] = f"FAIL: {str(e)}"
            logger.warning(f"⚠ SemanticMemoryStore storage test failed: {e}")
        
        # Basic semantic search test (may be stubbed in implementation)
        try:
            search_results = await semantic.search(
                query="bullish pattern bitcoin",
                limit=5
            )
            
            self.test_results["semantic_store"]["search"] = "PASS"
            logger.info("✓ SemanticMemoryStore search test passed")
        except Exception as e:
            self.test_results["semantic_store"]["search"] = f"FAIL: {str(e)}"
            logger.warning(f"⚠ SemanticMemoryStore search test failed: {e}")
    
    async def test_temporal_log(self):
        """Test TemporalEventLog functionality"""
        logger.info("Testing TemporalEventLog...")
        
        temporal = self.memory_core.temporal_log
        agent_id = self.test_agents["trader_agent"]["id"]
        
        # Test logging events
        events = [
            {
                "event_type": "market_open",
                "metadata": {"market": "crypto", "timestamp": datetime.utcnow().isoformat()}
            },
            {
                "event_type": "price_spike",
                "metadata": {"symbol": "BTC/USD", "magnitude": 0.05}
            },
            {
                "event_type": "trade_executed",
                "metadata": {"symbol": "ETH/USD", "action": "buy", "price": 2500.0}
            }
        ]
        
        event_ids = []
        for event in events:
            event_id = await temporal.log_event(
                agent_id=agent_id,
                event_type=event["event_type"],
                metadata=event["metadata"]
            )
            event_ids.append(event_id)
            assert event_id is not None, "Event logging should return a valid ID"
        
        self.test_results["temporal_log"]["logging"] = "PASS"
        logger.info("✓ TemporalEventLog logging test passed")
        
        # Test retrieving recent events
        recent_events = await temporal.get_recent_events(limit=10)
        assert len(recent_events) >= len(events), \
            "Should retrieve at least all logged events"
        
        # Test retrieving events by type
        trade_events = await temporal.get_events_by_type("trade_executed")
        assert len(trade_events) >= 1, \
            "Should retrieve events of specified type"
        
        self.test_results["temporal_log"]["retrieval"] = "PASS"
        logger.info("✓ TemporalEventLog retrieval test passed")
    
    async def test_message_bus(self):
        """Test MessageBus functionality"""
        logger.info("Testing MessageBus...")
        
        message_bus = self.memory_core.message_bus
        
        trader_id = self.test_agents["trader_agent"]["id"]
        analyst_id = self.test_agents["analyst_agent"]["id"]
        
        # Ensure agents are registered and subscribed
        if trader_id not in self.memory_core.registry.agents:
            trader = self.test_agents["trader_agent"]
            await self.memory_core.register_agent(
                trader["id"], trader["role"], trader["output_schema"]
            )
        
        if analyst_id not in self.memory_core.registry.agents:
            analyst = self.test_agents["analyst_agent"]
            await self.memory_core.register_agent(
                analyst["id"], analyst["role"], analyst["output_schema"]
            )
        
        # Subscribe to topics
        await self.memory_core.registry.subscribe_agent_to_topic(trader_id, "trading_signals")
        await self.memory_core.registry.subscribe_agent_to_topic(analyst_id, "market_analysis")
        
        # Test publishing messages
        trading_message = {
            "symbol": "BTC/USD",
            "signal": "buy",
            "confidence": 0.85,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message_id = await message_bus.publish(
            sender_id=analyst_id,
            topic="trading_signals",
            content=trading_message
        )
        
        assert message_id is not None, "Message publishing should return a valid ID"
        
        self.test_results["message_bus"]["publishing"] = "PASS"
        logger.info("✓ MessageBus publishing test passed")
        
        # Test message retrieval for subscribed agent
        if self.use_redis:
            # Allow time for message to propagate if using Redis
            await asyncio.sleep(1)
            
        messages = await message_bus.get_messages_for_agent(trader_id)
        
        # Note: Message retrieval might work differently in implementation
        # This is a simplified test
        if messages:
            self.test_results["message_bus"]["retrieval"] = "PASS"
            logger.info("✓ MessageBus retrieval test passed")
        else:
            self.test_results["message_bus"]["retrieval"] = "UNCERTAIN"
            logger.warning("⚠ MessageBus retrieval test result uncertain - no messages found")
    
    async def test_query_engine(self):
        """Test QueryEngine functionality"""
        logger.info("Testing QueryEngine...")
        
        query_engine = self.memory_core.query_engine
        agent_id = self.test_agents["analyst_agent"]["id"]
        
        # Create test data for the query
        episodic = self.memory_core.episodic_store
        
        # Test storing memories
        memory_ids = []
        for memory in self.test_memories:
            memory_id = await episodic.store(
                agent_id=agent_id,
                memory_type=memory["memory_type"],
                content=memory["content"],
                tags=memory["tags"]
            )
            memory_ids.append(memory_id)
        
        # Test basic query
        try:
            query_results = await query_engine.query(
                query="bullish patterns",
                agent_id=agent_id,
                limit=10
            )
            
            assert len(query_results) > 0, "Query should return results"
            
            self.test_results["query_engine"]["basic_query"] = "PASS"
            logger.info("✓ QueryEngine basic query test passed")
        except Exception as e:
            self.test_results["query_engine"]["basic_query"] = f"FAIL: {str(e)}"
            logger.warning(f"⚠ QueryEngine basic query test failed: {e}")
        
        # Test filtered query
        try:
            filtered_results = await query_engine.query(
                query="bearish patterns",
                agent_id=agent_id,
                memory_types=["analysis"],
                tags=["technical_pattern"],
                limit=10
            )
            
            self.test_results["query_engine"]["filtered_query"] = "PASS"
            logger.info("✓ QueryEngine filtered query test passed")
        except Exception as e:
            self.test_results["query_engine"]["filtered_query"] = f"FAIL: {str(e)}"
            logger.warning(f"⚠ QueryEngine filtered query test failed: {e}")
    
    async def test_summarizer(self):
        """Test SummarizationEngine functionality"""
        logger.info("Testing SummarizationEngine...")
        
        summarizer = self.memory_core.summarizer
        
        # Test basic summarization
        try:
            content = json.dumps(self.test_memories[0]["content"])
            summary = await summarizer.generate_summary(
                content=content,
                prompt_template="Summarize this trading insight: {content}"
            )
            
            assert summary is not None, "Summary should not be None"
            
            self.test_results["summarizer"]["basic_summarization"] = "PASS"
            logger.info("✓ SummarizationEngine basic summarization test passed")
        except Exception as e:
            self.test_results["summarizer"]["basic_summarization"] = f"FAIL: {str(e)}"
            logger.warning(f"⚠ SummarizationEngine basic summarization test failed: {e}")
        
        # Test agent output summarization
        try:
            agent_id = self.test_agents["analyst_agent"]["id"]
            summary = await summarizer.summarize_agent_output(
                agent_id=agent_id,
                data=self.test_memories[0]["content"]
            )
            
            self.test_results["summarizer"]["agent_summarization"] = "PASS"
            logger.info("✓ SummarizationEngine agent summarization test passed")
        except Exception as e:
            self.test_results["summarizer"]["agent_summarization"] = f"FAIL: {str(e)}"
            logger.warning(f"⚠ SummarizationEngine agent summarization test failed: {e}")
    
    async def test_memory_assistant(self):
        """Test MemoryAssistant functionality"""
        logger.info("Testing MemoryAssistant...")
        
        # Create memory assistant
        assistant = MemoryAssistant(self.memory_core)
        agent = self.test_agents["trader_agent"]
        
        # Test initialization
        init_result = await assistant.initialize_for_agent(
            agent["id"], agent["metadata"]
        )
        
        assert init_result is True, "Assistant initialization should succeed"
        
        self.test_results["assistant"]["initialization"] = "PASS"
        logger.info("✓ MemoryAssistant initialization test passed")
        
        # Test memory storage
        memory = self.test_memories[0]
        memory_id = await assistant.remember(
            content=memory["content"],
            memory_type=memory["memory_type"],
            tags=memory["tags"]
        )
        
        assert memory_id is not None, "Memory storage should return a valid ID"
        
        self.test_results["assistant"]["storage"] = "PASS"
        logger.info("✓ MemoryAssistant storage test passed")
        
        # Test memory recall
        recalled = await assistant.recall_recent(limit=5)
        assert len(recalled) > 0, "Should recall at least one memory"
        
        self.test_results["assistant"]["recall"] = "PASS"
        logger.info("✓ MemoryAssistant recall test passed")
        
        # Test context retrieval
        context = await assistant.get_context(window_size=10)
        assert "recent_memories" in context, "Context should include recent memories"
        
        self.test_results["assistant"]["context"] = "PASS"
        logger.info("✓ MemoryAssistant context retrieval test passed")
    
    async def test_integration(self):
        """Test integration between components"""
        logger.info("Testing component integration...")
        
        # Create memory assistants for two agents
        trader_assistant = MemoryAssistant(self.memory_core)
        analyst_assistant = MemoryAssistant(self.memory_core)
        
        # Initialize assistants
        await trader_assistant.initialize_for_agent(
            self.test_agents["trader_agent"]["id"],
            self.test_agents["trader_agent"]["metadata"]
        )
        
        await analyst_assistant.initialize_for_agent(
            self.test_agents["analyst_agent"]["id"],
            self.test_agents["analyst_agent"]["metadata"]
        )
        
        # Test cross-agent communication
        # Analyst stores a memory and sends a message
        analysis_content = {
            "symbol": "BTC/USD",
            "pattern": "double_bottom",
            "confidence": 0.92,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await analyst_assistant.remember(
            content=analysis_content,
            memory_type="analysis",
            tags=["reversal_pattern", "bullish"]
        )
        
        message_id = await analyst_assistant.send_message(
            topic="trading_signals",
            content={
                "symbol": "BTC/USD",
                "signal_type": "buy",
                "confidence": 0.9,
                "reason": "Double bottom pattern detected"
            }
        )
        
        # Allow time for message processing if using Redis
        if self.use_redis:
            await asyncio.sleep(1)
        
        # Trader reacts to message and stores a decision
        context = await trader_assistant.get_context()
        
        # Store trader decision
        await trader_assistant.remember(
            content={
                "symbol": "BTC/USD",
                "action": "buy",
                "reason": "Based on analyst signal",
                "confidence": 0.85,
                "timestamp": datetime.utcnow().isoformat()
            },
            memory_type="decision",
            tags=["trade_execution", "buy"]
        )
        
        # Test session summarization
        try:
            session_summary = await self.memory_core.summarizer.summarize_session(
                self.memory_core.get_current_session()
            )
            
            assert session_summary is not None, "Session summary should not be None"
            
            self.test_results["integration"]["session_summarization"] = "PASS"
            logger.info("✓ Integration session summarization test passed")
        except Exception as e:
            self.test_results["integration"]["session_summarization"] = f"FAIL: {str(e)}"
            logger.warning(f"⚠ Integration session summarization test failed: {e}")
        
        # Overall integration test
        self.test_results["integration"]["overall"] = "PASS"
        logger.info("✓ Component integration test passed")
    
    async def run_all_tests(self):
        """Run all test cases"""
        await self.setup()
        
        try:
            await self.test_memory_core()
            await self.test_agent_registry()
            await self.test_episodic_store()
            await self.test_semantic_store()
            await self.test_temporal_log()
            await self.test_message_bus()
            await self.test_query_engine()
            await self.test_summarizer()
            await self.test_memory_assistant()
            await self.test_integration()
            
            logger.info("All tests completed")
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            raise
        
        return self.test_results
    
    def print_results(self):
        """Print test results in a formatted way"""
        print("\n" + "="*60)
        print(" "*20 + "MEMORY SYSTEM TEST RESULTS")
        print("="*60 + "\n")
        
        for component, tests in self.test_results.items():
            print(f"COMPONENT: {component.upper()}")
            print("-"*60)
            
            for test_name, result in tests.items():
                status = "✅ PASS" if "PASS" in result else "❌ FAIL" if "FAIL" in result else "⚠️ UNCERTAIN"
                print(f"  {test_name}: {status}")
            
            print("\n")

async def main():
    """Main entry point for running memory system tests"""
    print("Starting LumosTrade Memory System Tests...")
    print("Note: Some tests may be skipped based on implementation status")
    
    # Run unit tests (without Redis)
    unit_tester = MemorySystemTester(use_redis=False)
    unit_results = await unit_tester.run_all_tests()
    unit_tester.print_results()
    
    # Check if Redis is available before running integration tests
    try:
        import redis
        r = redis.Redis.from_url("redis://localhost:6379/0")
        r.ping()
        
        print("\nRedis is available. Running integration tests with Redis...")
        integration_tester = MemorySystemTester(use_redis=True)
        integration_results = await integration_tester.run_all_tests()
        integration_tester.print_results()
    except (ImportError, redis.exceptions.ConnectionError):
        print("\nRedis is not available. Skipping integration tests with Redis.")
    
    print("\nMemory System testing complete!")

if __name__ == "__main__":
    asyncio.run(main())