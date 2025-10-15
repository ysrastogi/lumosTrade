#!/usr/bin/env python3
"""
Test persistent sessions for Athena memory manager.
"""

import asyncio
from src.agents.athena_workspace.memory_manager import AthenaMemoryManager

async def test_persistent_sessions():
    """Test that persistent sessions work across multiple manager instances"""
    
    print("=== Testing Persistent Sessions ===")
    
    # Test 1: Create first memory manager and store data
    print("\n1. Creating first memory manager...")
    manager1 = AthenaMemoryManager(agent_id="athena_agent", use_redis=True)
    await manager1.initialize()
    
    print(f"Manager 1 session ID: {manager1.memory_core.current_session_id}")
    
    # Store some test data
    test_data = {
        "symbol": "PERSISTENT_TEST",
        "price": 99999.99,
        "rsi": 45.0,
        "timestamp": "2025-10-14T22:00:00"
    }
    
    obs_id = await manager1.store_observation("PERSISTENT_TEST", test_data)
    print(f"Stored observation with ID: {obs_id}")
    
    # Store insight
    insight_id = await manager1.store_insight(
        symbol="PERSISTENT_TEST",
        insight_type="test_insight",
        insight="This is a persistent session test insight",
        confidence=0.9
    )
    print(f"Stored insight with ID: {insight_id}")
    
    # Test 2: Create second memory manager (simulating restart)
    print("\n2. Creating second memory manager (simulating restart)...")
    manager2 = AthenaMemoryManager(agent_id="athena_agent", use_redis=True)
    await manager2.initialize()
    
    print(f"Manager 2 session ID: {manager2.memory_core.current_session_id}")
    
    # Check if session IDs match
    if manager1.memory_core.current_session_id == manager2.memory_core.current_session_id:
        print("✅ Session IDs match - persistent sessions working!")
    else:
        print("❌ Session IDs don't match - persistent sessions not working")
        print(f"  Manager 1: {manager1.memory_core.current_session_id}")
        print(f"  Manager 2: {manager2.memory_core.current_session_id}")
    
    # Test 3: Try to retrieve data from second manager
    print("\n3. Retrieving data from second manager...")
    
    observations = await manager2.recall_recent_observations(symbol="PERSISTENT_TEST", limit=10)
    print(f"Retrieved {len(observations)} observations")
    
    insights = await manager2.recall_recent_insights(symbol="PERSISTENT_TEST", limit=10)
    print(f"Retrieved {len(insights)} insights")
    
    # Show retrieved data
    for obs in observations:
        print(f"  Observation: {obs.get('symbol')} - {obs.get('data', {}).get('price')}")
        
    for insight in insights:
        print(f"  Insight: {insight.get('symbol')} - {insight.get('insight')}")
    
    # Test 4: Also test retrieving older Athena data
    print("\n4. Retrieving all Athena observations and insights...")
    
    all_observations = await manager2.recall_recent_observations(limit=20)
    print(f"All observations: {len(all_observations)}")
    
    all_insights = await manager2.recall_recent_insights(limit=20)
    print(f"All insights: {len(all_insights)}")
    
    # Show symbols we have data for
    symbols = set()
    for obs in all_observations:
        symbols.add(obs.get('symbol', 'Unknown'))
    for insight in all_insights:
        symbols.add(insight.get('symbol', 'Unknown'))
    
    print(f"Symbols with data: {list(symbols)}")
    
    print("\n=== Persistent Session Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_persistent_sessions())