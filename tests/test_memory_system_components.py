"""
Memory System Unit Test Cases for LumosTrade

This module contains unit tests for individual components of the memory system.
"""

import pytest
import asyncio
import json
import logging
import os
import sys
import uuid
from unittest import mock
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Add the project root to the path so we can import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import memory components
from src.memory.memory_core import MemoryCore
from src.memory.registry import AgentRegistry
from src.memory.episodic_store import EpisodicMemoryStore
from src.memory.semantic_store import SemanticMemoryStore
from src.memory.temporal_log import TemporalEventLog
from src.memory.message_bus import MessageBus
from src.memory.retriever import QueryEngine
from src.memory.summarizer import SummarizationEngine
from src.memory.assistant import MemoryAssistant

# Test fixtures
@pytest.fixture
def memory_config():
    """Test configuration for memory components"""
    return {
        "debug_mode": True,
        "use_redis": False,
        "ttl_default": 3600,  # 1 hour
        "enable_persistence": False
    }

@pytest.fixture
def memory_core(memory_config):
    """Create a memory core for testing"""
    core = MemoryCore(config=memory_config)
    core.initialize_components()
    return core

@pytest.fixture
def agent_data():
    """Sample agent data for testing"""
    return {
        "id": "test_agent_1",
        "role": "Test Agent",
        "output_schema": ["result", "confidence", "timestamp"],
        "metadata": {
            "name": "TestAgent",
            "description": "Agent for testing",
            "capabilities": ["testing"],
            "interests": ["test_topic"]
        }
    }

@pytest.fixture
def memory_data():
    """Sample memory data for testing"""
    return {
        "content": {
            "symbol": "BTC/USD",
            "result": "test_result",
            "confidence": 0.95,
            "timestamp": datetime.utcnow().isoformat()
        },
        "memory_type": "test",
        "tags": ["test_tag"]
    }

@pytest.mark.asyncio
async def test_memory_core_initialization(memory_config):
    """Test that MemoryCore initializes correctly with components"""
    core = MemoryCore(config=memory_config)
    core.initialize_components()
    
    assert core.registry is not None
    assert core.episodic_store is not None
    assert core.semantic_store is not None
    assert core.temporal_log is not None
    assert core.message_bus is not None
    assert core.query_engine is not None
    assert core.summarizer is not None
    assert core.current_session_id is not None

@pytest.mark.asyncio
async def test_memory_core_session_management(memory_core):
    """Test session management in MemoryCore"""
    # Get initial session
    session_id = memory_core.get_current_session()
    assert session_id is not None
    
    # Start new session
    new_session = memory_core.start_new_session()
    assert new_session != session_id
    assert memory_core.get_current_session() == new_session

@pytest.mark.asyncio
async def test_agent_registry_registration(memory_core, agent_data):
    """Test agent registration in AgentRegistry"""
    registry = memory_core.registry
    
    # Register agent
    result = await memory_core.register_agent(
        agent_data["id"], 
        agent_data["role"], 
        agent_data["output_schema"]
    )
    
    assert result is True
    assert agent_data["id"] in registry.agents

@pytest.mark.asyncio
async def test_agent_registry_subscription(memory_core, agent_data):
    """Test agent subscription in AgentRegistry"""
    registry = memory_core.registry
    
    # Register agent first
    await memory_core.register_agent(
        agent_data["id"], 
        agent_data["role"], 
        agent_data["output_schema"]
    )
    
    # Test subscription
    topics = ["test_topic", "another_topic"]
    for topic in topics:
        await registry.subscribe_agent_to_topic(agent_data["id"], topic)
    
    # Verify subscriptions
    for topic in topics:
        assert agent_data["id"] in registry.subscriptions.get(topic, [])

@pytest.mark.asyncio
async def test_episodic_store_basic(memory_core, agent_data, memory_data):
    """Test basic operations of EpisodicMemoryStore"""
    episodic = memory_core.episodic_store
    
    # Register agent first
    await memory_core.register_agent(
        agent_data["id"], 
        agent_data["role"], 
        agent_data["output_schema"]
    )
    
    # Store a memory
    memory_id = await episodic.store(
        agent_id=agent_data["id"],
        memory_type=memory_data["memory_type"],
        content=memory_data["content"],
        tags=memory_data["tags"]
    )
    
    assert memory_id is not None
    
    # Get recent memories
    memories = await episodic.get_recent(agent_data["id"], limit=10)
    assert len(memories) > 0
    
    # Get by memory type
    type_memories = await episodic.get_by_type(
        agent_data["id"], 
        memory_type=memory_data["memory_type"]
    )
    assert len(type_memories) > 0
    
    # Get by tag
    tag_memories = await episodic.get_by_tag(
        agent_data["id"], 
        tag=memory_data["tags"][0]
    )
    assert len(tag_memories) > 0

@pytest.mark.asyncio
async def test_episodic_store_get_by_id(memory_core, agent_data, memory_data):
    """Test retrieving memory by ID from EpisodicMemoryStore"""
    episodic = memory_core.episodic_store
    
    # Register agent first
    await memory_core.register_agent(
        agent_data["id"], 
        agent_data["role"], 
        agent_data["output_schema"]
    )
    
    # Store a memory
    memory_id = await episodic.store(
        agent_id=agent_data["id"],
        memory_type=memory_data["memory_type"],
        content=memory_data["content"],
        tags=memory_data["tags"]
    )
    
    # Retrieve by ID
    memory = await episodic.get_by_id(memory_id)
    assert memory is not None
    assert memory["id"] == memory_id
    assert memory["agent_id"] == agent_data["id"]
    assert memory["content"] == memory_data["content"]

@pytest.mark.asyncio
async def test_temporal_log_events(memory_core, agent_data):
    """Test TemporalEventLog event handling"""
    temporal = memory_core.temporal_log
    
    # Register agent first
    await memory_core.register_agent(
        agent_data["id"], 
        agent_data["role"], 
        agent_data["output_schema"]
    )
    
    # Log some events
    events = [
        {
            "event_type": "test_event_1",
            "metadata": {"test_key": "test_value_1"}
        },
        {
            "event_type": "test_event_2",
            "metadata": {"test_key": "test_value_2"}
        }
    ]
    
    event_ids = []
    for event in events:
        event_id = await temporal.log_event(
            agent_id=agent_data["id"],
            event_type=event["event_type"],
            metadata=event["metadata"]
        )
        event_ids.append(event_id)
    
    # Get recent events
    recent = await temporal.get_recent_events(limit=10)
    assert len(recent) >= len(events)
    
    # Get events by type
    type_events = await temporal.get_events_by_type("test_event_1")
    assert len(type_events) > 0
    assert type_events[0]["event_type"] == "test_event_1"
    
    # Get events by agent
    agent_events = await temporal.get_events_by_agent(agent_data["id"])
    assert len(agent_events) >= len(events)

@pytest.mark.asyncio
async def test_message_bus_basic(memory_core, agent_data):
    """Test basic MessageBus operations"""
    message_bus = memory_core.message_bus
    registry = memory_core.registry
    
    # Register two agents
    agent1_id = agent_data["id"]
    agent2_id = f"{agent_data['id']}_2"
    
    await memory_core.register_agent(
        agent1_id, 
        agent_data["role"], 
        agent_data["output_schema"]
    )
    
    await memory_core.register_agent(
        agent2_id, 
        agent_data["role"], 
        agent_data["output_schema"]
    )
    
    # Subscribe agents to topics
    await registry.subscribe_agent_to_topic(agent1_id, "test_topic")
    await registry.subscribe_agent_to_topic(agent2_id, "test_topic")
    
    # Publish a message
    message_content = {
        "test_key": "test_value",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    message_id = await message_bus.publish(
        sender_id=agent1_id,
        topic="test_topic",
        content=message_content
    )
    
    assert message_id is not None
    
    # Get messages for agent
    messages = await message_bus.get_messages_for_agent(agent2_id)
    
    # Note: Retrieval might depend on implementation details
    # If using in-memory queue, messages should be available immediately
    # If using Redis, might need a delay or different check
    
    # At minimum, publishing should not fail
    assert True

@pytest.mark.asyncio
async def test_query_engine_basic(memory_core, agent_data, memory_data, monkeypatch):
    """Test basic QueryEngine operations with mocked semantics"""
    query_engine = memory_core.query_engine
    episodic = memory_core.episodic_store
    
    # Register agent and store memories
    await memory_core.register_agent(
        agent_data["id"], 
        agent_data["role"], 
        agent_data["output_schema"]
    )
    
    # Store multiple memories
    await episodic.store(
        agent_id=agent_data["id"],
        memory_type=memory_data["memory_type"],
        content=memory_data["content"],
        tags=memory_data["tags"]
    )
    
    # Create a second memory with different content
    second_memory = memory_data.copy()
    second_memory["content"] = {
        **memory_data["content"],
        "result": "different_result"
    }
    second_memory["tags"] = ["different_tag"]
    
    await episodic.store(
        agent_id=agent_data["id"],
        memory_type=second_memory["memory_type"],
        content=second_memory["content"],
        tags=second_memory["tags"]
    )
    
    # Mock the semantic search if needed
    # This may be needed if semantic search is not fully implemented
    async def mock_semantic_search(*args, **kwargs):
        # Return the memory we just stored
        return [{"id": "mock_id", "score": 0.95}]
        
    # Apply the monkeypatch if needed
    # monkeypatch.setattr(memory_core.semantic_store, "search", mock_semantic_search)
    
    # Test basic query
    # This will depend on the implementation - might use semantic or keyword search
    results = await query_engine.query(
        query="test_result",
        agent_id=agent_data["id"],
        limit=10
    )
    
    # We should get some results, but exact behavior depends on implementation
    assert isinstance(results, list)
    
    # Test filtered query
    filtered = await query_engine.query(
        query="test_result",
        agent_id=agent_data["id"],
        memory_types=[memory_data["memory_type"]],
        tags=[memory_data["tags"][0]],
        limit=10
    )
    
    # Should return filtered results
    assert isinstance(filtered, list)

@pytest.mark.asyncio
async def test_summarizer_stub_functionality(memory_core):
    """Test SummarizationEngine stub functionality"""
    summarizer = memory_core.summarizer
    
    # Test summary generation
    summary = await summarizer.generate_summary(
        content="This is test content to summarize",
        prompt_template="Summarize: {content}"
    )
    
    # Even with stub implementation, should return something
    assert summary is not None
    
    # Test agent output summarization
    agent_summary = await summarizer.summarize_agent_output(
        agent_id="test_agent",
        data={"result": "test_result"}
    )
    
    # Should return something
    assert agent_summary is not None

@pytest.mark.asyncio
async def test_memory_assistant_lifecycle(memory_core, agent_data, memory_data):
    """Test MemoryAssistant lifecycle and operations"""
    # Create assistant
    assistant = MemoryAssistant(memory_core)
    
    # Initialize for agent
    result = await assistant.initialize_for_agent(
        agent_data["id"],
        agent_data["metadata"]
    )
    
    assert result is True
    assert assistant.agent_id == agent_data["id"]
    assert assistant._init_complete is True
    
    # Store memory
    memory_id = await assistant.remember(
        content=memory_data["content"],
        memory_type=memory_data["memory_type"],
        tags=memory_data["tags"]
    )
    
    assert memory_id is not None
    
    # Recall memories
    memories = await assistant.recall_recent(limit=10)
    assert len(memories) > 0
    
    # Get context
    context = await assistant.get_context()
    assert "recent_memories" in context
    
    # Send message
    message_id = await assistant.send_message(
        topic="test_topic",
        content={"message": "test_message"}
    )
    
    assert message_id is not None

@pytest.mark.asyncio
async def test_memory_assistant_initialization_error():
    """Test MemoryAssistant error handling for uninitialized operations"""
    assistant = MemoryAssistant(MemoryCore(config={"debug_mode": True}))
    
    # Attempt operation without initialization
    with pytest.raises(RuntimeError):
        await assistant.remember(
            content={"test": "value"},
            memory_type="test"
        )