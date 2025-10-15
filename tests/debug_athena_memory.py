# Create a new test file: tests/debug_athena_memory.py
import asyncio
from src.agents.athena_workspace.memory_manager import AthenaMemoryManager
from src.memory.memory_core import MemoryCore
from src.memory.assistant import MemoryAssistant

async def debug_athena_memory():
    """Debug what memories Athena is actually storing and retrieving"""
    
    print("=== Debugging Athena Memory System ===")
    
    # Test 1: Use the same agent ID as Athena
    print("\n1. Testing with Athena's agent ID...")
    athena_memory = AthenaMemoryManager(agent_id="athena_agent", use_redis=True)
    await athena_memory.initialize()
    
    # Check if we can access the memory core directly
    print(f"Memory core session: {athena_memory.memory_core.current_session_id}")
    
    # Test 2: Query the memory assistant directly for athena_agent
    print("\n2. Querying memory assistant for athena_agent...")
    try:
        # Get ALL memories for athena_agent (no type filter)
        all_memories = await athena_memory.memory_assistant.recall_recent(limit=50)
        print(f"Found {len(all_memories)} total memories for athena_agent")
        
        # Group by memory type
        memory_types = {}
        for memory in all_memories:
            mem_type = memory.get('memory_type', 'unknown')
            if mem_type not in memory_types:
                memory_types[mem_type] = []
            memory_types[mem_type].append(memory)
        
        print("Memory types found:")
        for mem_type, memories in memory_types.items():
            print(f"  {mem_type}: {len(memories)} memories")
            if memories:
                latest = memories[0]
                print(f"    Latest: {latest.get('timestamp')} - {latest.get('data', {}).get('symbol', 'N/A')}")
    
    except Exception as e:
        print(f"Error querying memories: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Query specific memory types
    print("\n3. Testing specific memory type queries...")
    
    try:
        # Market observations
        observations = await athena_memory.memory_assistant.recall_recent(
            limit=10, 
            memory_types=["market_observation"]
        )
        print(f"Market observations: {len(observations)}")
        
        # Market insights
        insights = await athena_memory.memory_assistant.recall_recent(
            limit=10,
            memory_types=["market_insight"] 
        )
        print(f"Market insights: {len(insights)}")
        
        # Market analysis
        analysis = await athena_memory.memory_assistant.recall_recent(
            limit=10,
            memory_types=["market_analysis"]
        )
        print(f"Market analysis: {len(analysis)}")
        
    except Exception as e:
        print(f"Error with specific queries: {e}")
    
    # Test 4: Test Athena's memory manager methods directly
    print("\n4. Testing AthenaMemoryManager methods...")
    
    try:
        # Test recall methods
        obs_via_manager = await athena_memory.recall_recent_observations(limit=10)
        print(f"Observations via manager: {len(obs_via_manager)}")
        
        insights_via_manager = await athena_memory.recall_recent_insights(limit=10)
        print(f"Insights via manager: {len(insights_via_manager)}")
        
        # Test with symbol filter
        r10_obs = await athena_memory.recall_recent_observations(symbol="R_10", limit=10)
        print(f"R_10 observations: {len(r10_obs)}")
        
        r10_insights = await athena_memory.recall_recent_insights(symbol="R_10", limit=10)
        print(f"R_10 insights: {len(r10_insights)}")
        
    except Exception as e:
        print(f"Error with manager methods: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Manual storage and retrieval test
    print("\n5. Manual storage test...")
    
    try:
        # Store a test observation
        test_data = {
            "symbol": "TEST_SYMBOL",
            "price": 12345.67,
            "rsi": 65.5,
            "timestamp": "2025-10-14T21:30:00"
        }
        
        stored_id = await athena_memory.store_observation("TEST_SYMBOL", test_data)
        print(f"Stored test observation: {stored_id}")
        
        # Try to retrieve it immediately
        test_observations = await athena_memory.recall_recent_observations(symbol="TEST_SYMBOL", limit=5)
        print(f"Retrieved test observations: {len(test_observations)}")
        
        if test_observations:
            print(f"Test observation data: {test_observations[0]}")
        
    except Exception as e:
        print(f"Error with manual test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    asyncio.run(debug_athena_memory())