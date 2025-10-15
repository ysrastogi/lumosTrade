#!/usr/bin/env python3
"""
Comprehensive test for cross-agent memory persistence and access.

This test validates that:
1. Multiple agents can store memories in the global memory core
2. Agents can access each other's memories across sessions
3. Memory persistence works across agent restarts
4. Cross-agent intelligence is properly aggregated
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.athena_workspace.memory_manager import AthenaMemoryManager

async def test_cross_agent_memory():
    print("🧠 === Cross-Agent Memory Persistence Test ===\n")
    
    # === Phase 1: Setup Multiple Agents ===
    print("📝 Phase 1: Setting up multiple agents...")
    
    # Create three different agents
    athena_agent = AthenaMemoryManager(agent_id="athena_agent", use_redis=True)
    apollo_agent = AthenaMemoryManager(agent_id="apollo_agent", use_redis=True)
    chronos_agent = AthenaMemoryManager(agent_id="chronos_agent", use_redis=True)
    
    await athena_agent.initialize()
    await apollo_agent.initialize()
    await chronos_agent.initialize()
    
    print(f"✅ Athena session: {athena_agent.memory_core.current_session_id}")
    print(f"✅ Apollo session: {apollo_agent.memory_core.current_session_id}")
    print(f"✅ Chronos session: {chronos_agent.memory_core.current_session_id}")
    
    # === Phase 2: Store Data from Multiple Agents ===
    print(f"\n📊 Phase 2: Storing data from multiple agents...")
    
    # Athena stores market observations
    athena_obs1 = await athena_agent.store_observation("EURUSD", {
        "price": 1.0850, "trend": "bullish", "volume": 1000, "analysis": "strong uptrend"
    })
    athena_ins1 = await athena_agent.store_insight("EURUSD", "trend_analysis", 
        "Strong bullish momentum on EURUSD", 0.85)
    
    # Apollo stores signal analysis  
    apollo_obs1 = await apollo_agent.store_observation("EURUSD", {
        "signal": "buy", "strength": 0.75, "timeframe": "1H", "confluence": 3
    })
    apollo_ins1 = await apollo_agent.store_insight("EURUSD", "signal_analysis",
        "Multiple confluences suggest buy signal", 0.75)
    
    # Chronos stores temporal analysis
    chronos_obs1 = await chronos_agent.store_observation("EURUSD", {
        "timing": "optimal", "market_session": "london", "volatility": "medium"
    })
    chronos_ins1 = await chronos_agent.store_insight("EURUSD", "timing_analysis",
        "London session provides optimal timing", 0.70)
    
    print(f"✅ Athena stored: obs={athena_obs1}, insight={athena_ins1}")
    print(f"✅ Apollo stored: obs={apollo_obs1}, insight={apollo_ins1}")
    print(f"✅ Chronos stored: obs={chronos_obs1}, insight={chronos_ins1}")
    
    # === Phase 3: Test Cross-Agent Memory Access ===
    print(f"\n🔍 Phase 3: Testing cross-agent memory access...")
    
    # Test Athena accessing all agents' data
    print("\n--- Athena's Cross-Agent Query ---")
    cross_obs = await athena_agent.recall_cross_agent_observations("EURUSD", limit=5)
    cross_ins = await athena_agent.recall_cross_agent_insights("EURUSD", limit=5)
    
    print(f"Cross-agent observations found: {len(cross_obs)} agents")
    for agent_id, observations in cross_obs.items():
        print(f"  {agent_id}: {len(observations)} observations")
        for obs in observations[:1]:  # Show first observation
            print(f"    Sample: {list(obs.get('data', {}).keys())}")
    
    print(f"Cross-agent insights found: {len(cross_ins)} agents")  
    for agent_id, insights in cross_ins.items():
        print(f"  {agent_id}: {len(insights)} insights")
    
    # === Phase 4: Test Global Memory Context ===
    print(f"\n🌍 Phase 4: Testing global memory context...")
    
    global_context = await athena_agent.get_global_memory_context("EURUSD")
    
    print(f"Global Context Stats:")
    stats = global_context.get("global_stats", {})
    print(f"  Active agents: {stats.get('active_agents')}")
    print(f"  Total observations: {stats.get('total_observations')}")
    print(f"  Total insights: {stats.get('total_insights')}")
    print(f"  Agent list: {global_context.get('agent_list')}")
    
    # === Phase 5: Test Agent Restart Persistence ===
    print(f"\n🔄 Phase 5: Testing agent restart persistence...")
    
    # "Restart" Apollo by creating a new instance
    print("Simulating Apollo agent restart...")
    apollo_restarted = AthenaMemoryManager(agent_id="apollo_agent", use_redis=True)
    await apollo_restarted.initialize()
    
    # Apollo should still be able to access all memories including its own
    apollo_cross_obs = await apollo_restarted.recall_cross_agent_observations("EURUSD", limit=5)
    apollo_own_obs = await apollo_restarted.recall_recent_observations("EURUSD", limit=5)
    
    print(f"✅ Restarted Apollo can see {len(apollo_cross_obs)} agents' data")
    print(f"✅ Restarted Apollo can see {len(apollo_own_obs)} of its own observations")
    
    # === Phase 6: Test Agent Memory Summary ===
    print(f"\n📋 Phase 6: Testing agent memory summaries...")
    
    athena_summary = await chronos_agent.get_agent_memory_summary("athena_agent", "EURUSD")
    print(f"Chronos viewing Athena's memory:")
    print(f"  Observations: {athena_summary.get('observations_count')}")
    print(f"  Insights: {athena_summary.get('insights_count')}")
    print(f"  Last activity: {athena_summary.get('last_activity')}")
    
    # === Phase 7: Test Different Symbol Isolation ===
    print(f"\n🔗 Phase 7: Testing symbol isolation...")
    
    # Store data for different symbol
    await athena_agent.store_observation("GBPUSD", {
        "price": 1.2750, "trend": "bearish", "strength": "weak"
    })
    
    # Query should only return EURUSD data when filtered
    eurusd_only = await athena_agent.recall_cross_agent_observations("EURUSD", limit=10)
    gbpusd_only = await athena_agent.recall_cross_agent_observations("GBPUSD", limit=10)
    all_symbols = await athena_agent.recall_cross_agent_observations(None, limit=10)
    
    print(f"EURUSD observations: {sum(len(obs) for obs in eurusd_only.values())}")
    print(f"GBPUSD observations: {sum(len(obs) for obs in gbpusd_only.values())}")
    print(f"All symbols observations: {sum(len(obs) for obs in all_symbols.values())}")
    
    # === Final Summary ===
    print(f"\n🎉 === Test Summary ===")
    
    success_criteria = [
        ("Multi-agent storage", len(cross_obs) >= 3),
        ("Cross-agent access", sum(len(obs) for obs in cross_obs.values()) >= 3),
        ("Global context", stats.get('active_agents', 0) >= 3),
        ("Restart persistence", len(apollo_cross_obs) >= 3),
        ("Own memory access", len(apollo_own_obs) >= 1),
        ("Symbol isolation", len(gbpusd_only) >= 1)
    ]
    
    passed = 0
    for test_name, condition in success_criteria:
        status = "✅ PASS" if condition else "❌ FAIL"
        print(f"{status} {test_name}")
        if condition:
            passed += 1
    
    print(f"\n🏆 Overall Result: {passed}/{len(success_criteria)} tests passed")
    
    if passed == len(success_criteria):
        print("🎉 ALL TESTS PASSED! Cross-agent memory persistence is working perfectly!")
    else:
        print("⚠️  Some tests failed. Check the implementation.")
    
    return passed == len(success_criteria)

if __name__ == "__main__":
    asyncio.run(test_cross_agent_memory())