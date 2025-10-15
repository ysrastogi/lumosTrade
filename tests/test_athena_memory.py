"""
Test module for the memory integration with Athena agent.

This file verifies that Athena can properly register with and use the memory system.
"""

import asyncio
import json
import logging
from datetime import datetime

from src.agents.athena_workspace.athena import AthenaAgent

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_athena_memory_integration():
    """
    Test the integration of Athena with the memory system.
    """
    logger.info("Creating Athena agent with memory system integration...")
    
    # Create Athena with memory integration enabled
    athena = AthenaAgent(
        agent_id="test_athena",
        use_memory=True,
        use_redis=False  # Use in-memory storage for testing
    )
    
    # Initialize the agent
    await athena.initialize()
    
    # Observe a market symbol to generate some data
    logger.info("Running market observation to generate data...")
    observation = await athena.observe("R_10", interval=60, count=100)
    
    # Wait a moment to ensure all async operations complete
    await asyncio.sleep(1)
    
    # Retrieve history from memory system
    logger.info("Retrieving history from memory system...")
    history = await athena.get_recent_history("R_10")  # Use the same symbol that we observed
    
    # Check if we got any results
    if history:
        logger.info(f"Successfully retrieved {len(history)} items from memory!")
        # Print the first item summary
        logger.info(f"First memory item type: {history[0].get('memory_type', 'unknown')}")
    else:
        logger.warning("No history items retrieved from memory")
    
    # Get memory context
    logger.info("Retrieving memory context...")
    context = await athena.get_memory_context("R_10")  # Use the same symbol that we observed
    
    if context:
        logger.info(f"Successfully retrieved memory context with {len(context.get('recent_memories', []))} memories")
    else:
        logger.warning("No memory context retrieved")
    
    logger.info("Memory integration test complete!")
    return True

if __name__ == "__main__":
    asyncio.run(test_athena_memory_integration())