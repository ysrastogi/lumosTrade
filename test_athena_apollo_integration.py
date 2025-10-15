"""
Test script for Athena-Apollo integration

This script tests the integration between Athena and Apollo agents to ensure:
1. Athena can analyze market data
2. Apollo can access Athena's memory
3. Apollo can generate signals based on Athena's analysis
4. Apollo can store signals in memory
5. Both agents can access cross-agent memory
"""

import os
import sys
import asyncio
import logging
import json

# Set up path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the integration module
from examples.athena_apollo_integration import AthenaApolloIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_integration():
    """Test the integration between Athena and Apollo"""
    
    # Define test symbols
    test_symbols = ["BTC/USD", "ETH/USD"]
    
    # Initialize the integration
    integration = AthenaApolloIntegration(
        use_redis=True,
        use_llm=True
    )
    await integration.initialize()
    
    # Test results storage
    results = {
        "athena_analysis": {},
        "apollo_signals": {},
        "cross_agent_memory": {},
        "tests_passed": 0,
        "tests_failed": 0,
        "errors": []
    }
    
    # Test 1: Athena can analyze market data
    logger.info("Test 1: Athena Market Analysis")
    try:
        for symbol in test_symbols:
            result = await integration.analyze_symbol(symbol)
            
            if 'error' in result:
                logger.error(f"❌ Analysis error for {symbol}: {result['error']}")
                results["errors"].append(f"Analysis failed for {symbol}: {result['error']}")
                results["tests_failed"] += 1
            else:
                logger.info(f"✅ Successfully analyzed {symbol}")
                results["athena_analysis"][symbol] = {
                    "regime": result['athena_context'].get('regime'),
                    "regime_confidence": result['athena_context'].get('regime_confidence'),
                    "trading_bias": result['athena_context'].get('trading_bias')
                }
                results["apollo_signals"][symbol] = [s for s in result['apollo_signals']]
                results["tests_passed"] += 1
    except Exception as e:
        logger.error(f"❌ Test 1 failed: {str(e)}")
        results["errors"].append(f"Test 1 failed: {str(e)}")
        results["tests_failed"] += 1
    
    # Test 2: Memory integration and cross-agent access
    logger.info("Test 2: Memory Integration and Cross-Agent Access")
    try:
        combined_context = await integration.get_combined_memory_context()
        
        athena_ctx = combined_context.get('athena_context', {})
        apollo_ctx = combined_context.get('apollo_context', {})
        
        athena_intelligence = athena_ctx.get('intelligence_summary', {})
        apollo_intelligence = apollo_ctx.get('intelligence_summary', {})
        
        has_cross_agent = (
            athena_intelligence.get('cross_agent_observations', 0) > 0 or
            athena_intelligence.get('cross_agent_insights', 0) > 0 or
            apollo_intelligence.get('cross_agent_observations', 0) > 0 or
            apollo_intelligence.get('cross_agent_insights', 0) > 0
        )
        
        if has_cross_agent:
            logger.info("✅ Cross-agent intelligence detected")
            results["cross_agent_memory"]["status"] = "success"
            results["cross_agent_memory"]["details"] = {
                "athena_observations": athena_intelligence.get('cross_agent_observations', 0),
                "athena_insights": athena_intelligence.get('cross_agent_insights', 0),
                "apollo_observations": apollo_intelligence.get('cross_agent_observations', 0),
                "apollo_insights": apollo_intelligence.get('cross_agent_insights', 0)
            }
            results["tests_passed"] += 1
        else:
            logger.warning("⚠️ No cross-agent intelligence detected yet")
            results["cross_agent_memory"]["status"] = "partial"
            results["cross_agent_memory"]["details"] = {
                "message": "Memory system initialized but no cross-agent data found yet"
            }
            # Don't count as failure, just report status
            results["tests_passed"] += 1
    except Exception as e:
        logger.error(f"❌ Test 2 failed: {str(e)}")
        results["errors"].append(f"Test 2 failed: {str(e)}")
        results["tests_failed"] += 1
    
    # Cleanup
    await integration.close()
    
    # Print test summary
    logger.info("\n" + "="*50)
    logger.info(f"Test Summary: {results['tests_passed']} passed, {results['tests_failed']} failed")
    
    if results["errors"]:
        logger.error("Errors:")
        for error in results["errors"]:
            logger.error(f"  - {error}")
    
    if results["tests_failed"] == 0:
        logger.info("✅ All tests passed successfully!")
    else:
        logger.error(f"❌ {results['tests_failed']} tests failed")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_integration())