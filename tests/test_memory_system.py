"""
Memory System Integration Tests for LumosTrade

This module contains tests to validate the Memory System implementation.
"""

import asyncio
import json
import logging
import os
import sys
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the project root to the path so we can import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.memory_core import MemoryCore
from src.memory.assistant import MemoryAssistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Test data
TEST_AGENT_ID = "test_agent"
TEST_AGENT_METADATA = {
    "name": "TestAgent",
    "description": "Agent for memory system testing",
    "capabilities": ["test_capability"],
    "interests": ["test_topic"]
}

@pytest.fixture
async def memory_core():
    """Fixture to create a MemoryCore instance for testing"""
    # Use in-memory stores for testing
    core = MemoryCore(config={
        "use_redis": False,
        "enable_persistence": False,
        "debug_mode": True
    })
    yield core
    # Cleanup (if needed)

@pytest.fixture
async def memory_assistant(memory_core):
    """Fixture to create a MemoryAssistant instance for testing"""
    assistant = MemoryAssistant(memory_core)
    await assistant.initialize_for_agent(TEST_AGENT_ID, TEST_AGENT_METADATA)
    yield assistant

@pytest.mark.asyncio
async def test_agent_registration(memory_core):
    """Test agent registration functionality"""
    # Register an agent
    success = await memory_core.register_agent(TEST_AGENT_ID, TEST_AGENT_METADATA)
    assert success, "Agent registration should succeed"
    
    # Verify agent was registered
    agent = await memory_core.registry.get_agent(TEST_AGENT_ID)
    assert agent is not None, "Agent should be retrievable after registration"
    assert agent.name == TEST_AGENT_METADATA["name"], "Agent name should match"

@pytest.mark.asyncio
async def test_memory_storage_and_retrieval(memory_assistant):
    """Test storing and retrieving memories"""
    # Store a test memory
    test_content = {
        "key": "value",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    memory_id = await memory_assistant.remember(
        content=test_content,
        memory_type="test",
        tags=["test_tag"]
    )
    
    assert memory_id, "Memory storage should return a valid ID"
    
    # Retrieve the memory
    memories = await memory_assistant.recall_recent(limit=1)
    assert len(memories) == 1, "Should retrieve exactly one memory"
    assert memories[0]["content"]["key"] == "value", "Memory content should match"
    assert "test_tag" in memories[0]["tags"], "Memory tags should be preserved"

@pytest.mark.asyncio
async def test_message_bus(memory_core):
    """Test the message bus functionality"""
    # Register two agents
    await memory_core.register_agent("agent1", {
        "name": "Agent1", 
        "interests": ["topic1", "topic2"]
    })
    
    await memory_core.register_agent("agent2", {
        "name": "Agent2", 
        "interests": ["topic2", "topic3"]
    })
    
    # Subscribe agents to topics
    await memory_core.subscribe("agent1", ["topic1", "topic2"])
    await memory_core.subscribe("agent2", ["topic2", "topic3"])
    
    # Send a message on topic2 (both should receive)
    message_content = {"test": "message"}
    message_id = await memory_core.publish_message(
        sender_id="agent1",
        topic="topic2",
        content=message_content
    )
    
    assert message_id, "Message publish should return a valid ID"
    
    # Check that agent2 received the message
    agent2_assistant = MemoryAssistant(memory_core)
    await agent2_assistant.initialize_for_agent("agent2", {})
    
    context = await agent2_assistant.get_context()
    messages = context["recent_messages"]
    
    assert any(msg["content"]["test"] == "message" for msg in messages), \
        "Agent2 should receive the message on topic2"

@pytest.mark.asyncio
async def test_temporal_log(memory_core):
    """Test the temporal log functionality"""
    # Log some events
    await memory_core.temporal.log_event(
        agent_id="agent1",
        event_type="test_event",
        metadata={"test_key": "test_value"}
    )
    
    await memory_core.temporal.log_event(
        agent_id="agent2",
        event_type="another_event",
        metadata={"another_key": "another_value"}
    )
    
    # Retrieve recent events
    events = await memory_core.temporal.get_recent_events(limit=10)
    assert len(events) >= 2, "Should retrieve at least 2 events"
    
    # Check for specific event
    assert any(e["event_type"] == "test_event" for e in events), \
        "Should find the test_event in recent events"
    
    # Test time range query
    start_time = datetime.utcnow() - timedelta(minutes=5)
    end_time = datetime.utcnow() + timedelta(minutes=5)
    
    time_range_events = await memory_core.temporal.get_events_in_range(
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat()
    )
    
    assert len(time_range_events) >= 2, "Should retrieve events in time range"

@pytest.mark.asyncio
async def test_memory_context(memory_assistant):
    """Test retrieving agent context"""
    # Store some memories and messages
    await memory_assistant.remember(
        content={"test": "memory1"},
        memory_type="observation"
    )
    
    await memory_assistant.remember(
        content={"test": "memory2"},
        memory_type="decision"
    )
    
    await memory_assistant.send_message(
        topic="test_topic",
        content={"message": "test_message"}
    )
    
    # Get context
    context = await memory_assistant.get_context()
    
    assert "recent_memories" in context, "Context should include recent memories"
    assert len(context["recent_memories"]) >= 2, "Should have at least 2 memories"
    assert "recent_messages" in context, "Context should include recent messages"
    assert "timeline" in context, "Context should include timeline events"

@pytest.mark.asyncio
async def test_memory_persistence(tmp_path):
    """Test memory persistence across instances"""
    # Create a persistent memory core
    persistence_path = os.path.join(tmp_path, "memory_test")
    
    # First instance - store data
    core1 = MemoryCore(config={
        "persistence_path": persistence_path,
        "enable_persistence": True
    })
    
    # Register and store memory
    await core1.register_agent(TEST_AGENT_ID, TEST_AGENT_METADATA)
    
    assistant1 = MemoryAssistant(core1)
    await assistant1.initialize_for_agent(TEST_AGENT_ID, TEST_AGENT_METADATA)
    
    test_memory_id = await assistant1.remember(
        content={"persistence": "test"},
        memory_type="test_persistence"
    )
    
    # Force persistence
    # In the real implementation, this would be handled by the persistence mechanism
    # For testing, we're assuming it's written immediately
    
    # Create a second instance that should load from the same path
    core2 = MemoryCore(config={
        "persistence_path": persistence_path,
        "enable_persistence": True
    })
    
    assistant2 = MemoryAssistant(core2)
    await assistant2.initialize_for_agent(TEST_AGENT_ID, TEST_AGENT_METADATA)
    
    # Try to retrieve the memory
    # Note: This test might need adaptation based on the actual persistence implementation
    memories = await assistant2.recall("persistence")
    
    # Verify some memory was found
    # The exact behavior depends on the persistence implementation
    assert len(memories) > 0, "Should retrieve some memories from persistent storage"

@pytest.mark.asyncio
async def test_memory_analysis(memory_assistant):
    """Test memory analysis capabilities"""
    # Store multiple related memories
    for i in range(5):
        await memory_assistant.remember(
            content={
                "symbol": "BTC/USD",
                "pattern": f"pattern_{i}",
                "confidence": 0.5 + (i * 0.1)
            },
            memory_type="analysis",
            tags=["test_analysis"]
        )
    
    # Analyze history
    analysis = await memory_assistant.analyze_history(query="BTC/USD pattern")
    
    assert analysis["memory_count"] > 0, "Analysis should find memories"
    assert "insights" in analysis, "Analysis should include insights"
    assert analysis["agent_id"] == TEST_AGENT_ID, "Analysis should reference agent ID"

if __name__ == "__main__":
    # For manual test execution
    async def run_tests():
        core = MemoryCore(config={"debug_mode": True})
        assistant = MemoryAssistant(core)
        await assistant.initialize_for_agent(TEST_AGENT_ID, TEST_AGENT_METADATA)
        
        await test_agent_registration(core)
        await test_memory_storage_and_retrieval(assistant)
        await test_message_bus(core)
        await test_temporal_log(core)
        await test_memory_context(assistant)
        
        print("All manual tests completed")
    
    asyncio.run(run_tests())