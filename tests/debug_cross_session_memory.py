#!/usr/bin/env python3
"""
Debug script to access all Athena memories across sessions.
"""

import asyncio
import redis
from datetime import datetime
from src.agents.athena_workspace.memory_manager import AthenaMemoryManager
from src.memory.memory_core import MemoryCore
from src.memory.assistant import MemoryAssistant

async def debug_cross_session_memories():
    """Debug memories across all sessions"""
    
    print("=== Debugging Cross-Session Athena Memories ===")
    
    # Method 1: Direct Redis access to find all memories
    print("\n1. Direct Redis query for all athena_agent memories...")
    try:
        r = redis.Redis(host='localhost', port=6380, db=0)  # Your Redis port
        
        # Find all memory keys for athena_agent
        memory_keys = r.keys("memory:athena_agent:*")
        print(f"Found {len(memory_keys)} memory keys for athena_agent")
        
        # Show some examples
        for i, key in enumerate(memory_keys[:5]):
            key_str = key.decode('utf-8') if isinstance(key, bytes) else str(key)
            print(f"  Key {i+1}: {key_str}")
            
            # Get the memory data
            memory_data = r.get(key)
            if memory_data:
                import json
                try:
                    data = json.loads(memory_data.decode('utf-8'))
                    print(f"    Type: {data.get('memory_type')}")
                    print(f"    Session: {data.get('session_id')}")
                    print(f"    Symbol: {data.get('data', {}).get('symbol', 'N/A')}")
                except Exception as e:
                    print(f"    Error parsing: {e}")
        
    except Exception as e:
        print(f"Redis error: {e}")
    
    # Method 2: Query episodic store directly
    print("\n2. Direct episodic store query...")
    try:
        from src.memory.episodic_store import EpisodicMemoryStore
        
        episodic_store = EpisodicMemoryStore(use_redis=True)
        
        # Get all memories for athena_agent (across all sessions)
        all_memories = await episodic_store.get_memories_by_agent("athena_agent", limit=50)
        print(f"Found {len(all_memories)} memories via episodic store")
        
        # Group by session and type
        sessions = {}
        for memory in all_memories:
            session_id = memory.get('session_id', 'unknown')
            mem_type = memory.get('memory_type', 'unknown')
            
            if session_id not in sessions:
                sessions[session_id] = {}
            if mem_type not in sessions[session_id]:
                sessions[session_id][mem_type] = []
            sessions[session_id][mem_type].append(memory)
        
        print("Memories by session:")
        for session_id, types in sessions.items():
            print(f"  Session {session_id}:")
            for mem_type, memories in types.items():
                print(f"    {mem_type}: {len(memories)} memories")
                # Show latest memory details
                if memories:
                    latest = memories[0]
                    symbol = latest.get('data', {}).get('symbol', 'N/A')
                    timestamp = latest.get('timestamp', 'N/A')
                    print(f"      Latest: {symbol} at {timestamp}")
        
    except Exception as e:
        print(f"Episodic store error: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 3: Create a memory manager that can access any session
    print("\n3. Testing cross-session memory access...")
    try:
        # Create a memory manager
        memory_manager = AthenaMemoryManager(agent_id="athena_agent", use_redis=True)
        await memory_manager.initialize()
        
        # Try to modify the memory assistant to query across sessions
        # This is a hack to bypass session isolation
        original_recall = memory_manager.memory_assistant.recall_recent
        
        async def cross_session_recall(limit=10, memory_types=None):
            """Modified recall that searches across all sessions"""
            try:
                # Get all memories for this agent directly from episodic store
                episodic_store = memory_manager.memory_core.episodic_store
                agent_memories = await episodic_store.get_memories_by_agent(
                    memory_manager.agent_id, 
                    limit=limit
                )
                
                # Filter by memory types if specified
                if memory_types:
                    agent_memories = [
                        m for m in agent_memories 
                        if m.get('memory_type') in memory_types
                    ]
                
                return agent_memories
                
            except Exception as e:
                print(f"Cross-session recall error: {e}")
                return []
        
        # Use the cross-session recall
        all_observations = await cross_session_recall(
            limit=20, 
            memory_types=["market_observation"]
        )
        print(f"Cross-session observations: {len(all_observations)}")
        
        all_insights = await cross_session_recall(
            limit=20,
            memory_types=["market_insight"]
        )
        print(f"Cross-session insights: {len(all_insights)}")
        
        # Show some details
        for obs in all_observations[:3]:
            data = obs.get('data', {})
            print(f"  Observation: {data.get('symbol')} at {obs.get('timestamp')}")
            
        for insight in all_insights[:3]:
            data = insight.get('data', {})
            print(f"  Insight: {data.get('symbol')} - {data.get('insight_type')}")
        
    except Exception as e:
        print(f"Cross-session test error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Cross-Session Debug Complete ===")

if __name__ == "__main__":
    asyncio.run(debug_cross_session_memories())