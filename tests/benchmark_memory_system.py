"""
Memory System Performance Benchmark Tests for LumosTrade

This script benchmarks the performance of the memory system under various loads.
"""

import asyncio
import json
import logging
import os
import random
import string
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

# Add the project root to the path so we can import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import memory components
from src.memory.memory_core import MemoryCore
from src.memory.assistant import MemoryAssistant

class MemorySystemBenchmark:
    """Benchmark for the LumosTrade Memory System"""
    
    def __init__(self, use_redis: bool = True):
        """
        Initialize the benchmark.
        
        Args:
            use_redis: Whether to use Redis for the benchmark
        """
        self.use_redis = use_redis
        self.memory_core = None
        self.results = {
            "store_memory": [],
            "retrieve_memory": [],
            "message_publish": [],
            "query_execution": [],
            "context_retrieval": []
        }
    
    def _generate_random_agent(self, index: int) -> Dict[str, Any]:
        """Generate random agent data"""
        capabilities = ["analysis", "prediction", "trading", "monitoring", "research"]
        interests = ["market_data", "news", "social_media", "technical_indicators", "fundamentals"]
        
        return {
            "id": f"bench_agent_{index}",
            "role": f"Benchmark Agent {index}",
            "output_schema": ["result", "confidence", "timestamp"],
            "metadata": {
                "name": f"BenchAgent{index}",
                "description": f"Benchmark test agent {index}",
                "capabilities": random.sample(capabilities, k=random.randint(1, len(capabilities))),
                "interests": random.sample(interests, k=random.randint(1, len(interests)))
            }
        }
    
    def _generate_random_memory(self) -> Dict[str, Any]:
        """Generate random memory content"""
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD"]
        memory_types = ["observation", "analysis", "decision", "insight"]
        tags = ["technical", "fundamental", "sentiment", "on-chain", "market"]
        
        # Generate random text
        text = ''.join(random.choice(string.ascii_letters) for _ in range(100))
        
        return {
            "content": {
                "symbol": random.choice(symbols),
                "text": text,
                "value": random.random(),
                "timestamp": datetime.utcnow().isoformat()
            },
            "memory_type": random.choice(memory_types),
            "tags": random.sample(tags, k=random.randint(1, 3))
        }
    
    async def setup(self):
        """Set up benchmark environment"""
        # Create memory core with benchmark configuration
        self.memory_core = MemoryCore(
            config={
                "use_redis": self.use_redis,
                "debug_mode": True,
                "enable_persistence": False
            }
        )
        
        # Initialize components
        self.memory_core.initialize_components()
        logger.info("Benchmark environment setup complete")
        
        # Register agents for benchmarking
        self.agents = [self._generate_random_agent(i) for i in range(10)]
        
        for agent in self.agents:
            await self.memory_core.register_agent(
                agent["id"], 
                agent["role"], 
                agent["output_schema"]
            )
        
        logger.info(f"Registered {len(self.agents)} benchmark agents")
    
    async def benchmark_memory_storage(self, n_memories: int = 100):
        """
        Benchmark memory storage performance.
        
        Args:
            n_memories: Number of memories to store
        """
        logger.info(f"Benchmarking memory storage with {n_memories} memories...")
        
        # Create memory assistant
        agent = self.agents[0]
        assistant = MemoryAssistant(self.memory_core)
        await assistant.initialize_for_agent(agent["id"], agent["metadata"])
        
        total_time = 0
        memory_ids = []
        
        for i in range(n_memories):
            memory = self._generate_random_memory()
            
            start_time = time.time()
            memory_id = await assistant.remember(
                content=memory["content"],
                memory_type=memory["memory_type"],
                tags=memory["tags"]
            )
            elapsed = time.time() - start_time
            
            total_time += elapsed
            memory_ids.append(memory_id)
            
            # Record individual operation time
            self.results["store_memory"].append(elapsed)
        
        avg_time = total_time / n_memories
        logger.info(f"Memory storage: {avg_time:.6f} seconds per operation (total: {total_time:.6f}s)")
        
        return memory_ids
    
    async def benchmark_memory_retrieval(self, n_queries: int = 100):
        """
        Benchmark memory retrieval performance.
        
        Args:
            n_queries: Number of retrieval operations to perform
        """
        logger.info(f"Benchmarking memory retrieval with {n_queries} queries...")
        
        # Ensure we have some memories stored
        memory_ids = await self.benchmark_memory_storage(100)
        
        # Create memory assistant
        agent = self.agents[0]
        assistant = MemoryAssistant(self.memory_core)
        await assistant.initialize_for_agent(agent["id"], agent["metadata"])
        
        total_time = 0
        
        for i in range(n_queries):
            start_time = time.time()
            memories = await assistant.recall_recent(limit=10)
            elapsed = time.time() - start_time
            
            total_time += elapsed
            
            # Record individual operation time
            self.results["retrieve_memory"].append(elapsed)
        
        avg_time = total_time / n_queries
        logger.info(f"Memory retrieval: {avg_time:.6f} seconds per operation (total: {total_time:.6f}s)")
    
    async def benchmark_message_publishing(self, n_messages: int = 100):
        """
        Benchmark message publishing performance.
        
        Args:
            n_messages: Number of messages to publish
        """
        logger.info(f"Benchmarking message publishing with {n_messages} messages...")
        
        # Create memory assistants for sender and receiver
        sender = self.agents[0]
        receiver = self.agents[1]
        
        # Subscribe receiver to topics
        topics = ["bench_topic_1", "bench_topic_2", "bench_topic_3"]
        for topic in topics:
            await self.memory_core.registry.subscribe_agent_to_topic(receiver["id"], topic)
        
        # Create sender assistant
        sender_assistant = MemoryAssistant(self.memory_core)
        await sender_assistant.initialize_for_agent(sender["id"], sender["metadata"])
        
        total_time = 0
        
        for i in range(n_messages):
            # Generate random message
            topic = random.choice(topics)
            message = {
                "type": "benchmark",
                "sequence": i,
                "data": ''.join(random.choice(string.ascii_letters) for _ in range(50)),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            start_time = time.time()
            message_id = await sender_assistant.send_message(topic=topic, content=message)
            elapsed = time.time() - start_time
            
            total_time += elapsed
            
            # Record individual operation time
            self.results["message_publish"].append(elapsed)
        
        avg_time = total_time / n_messages
        logger.info(f"Message publishing: {avg_time:.6f} seconds per operation (total: {total_time:.6f}s)")
    
    async def benchmark_query_execution(self, n_queries: int = 50):
        """
        Benchmark query execution performance.
        
        Args:
            n_queries: Number of queries to execute
        """
        logger.info(f"Benchmarking query execution with {n_queries} queries...")
        
        # Ensure we have some memories stored
        memory_ids = await self.benchmark_memory_storage(100)
        
        # Create query terms
        query_terms = [
            "technical analysis",
            "price pattern",
            "market sentiment",
            "trading signal",
            "BTC price prediction"
        ]
        
        total_time = 0
        
        for i in range(n_queries):
            query = random.choice(query_terms)
            agent = random.choice(self.agents)
            
            start_time = time.time()
            results = await self.memory_core.query_engine.query(
                query=query,
                agent_id=agent["id"],
                limit=10
            )
            elapsed = time.time() - start_time
            
            total_time += elapsed
            
            # Record individual operation time
            self.results["query_execution"].append(elapsed)
        
        avg_time = total_time / n_queries
        logger.info(f"Query execution: {avg_time:.6f} seconds per operation (total: {total_time:.6f}s)")
    
    async def benchmark_context_retrieval(self, n_contexts: int = 50):
        """
        Benchmark context retrieval performance.
        
        Args:
            n_contexts: Number of context retrievals to perform
        """
        logger.info(f"Benchmarking context retrieval with {n_contexts} operations...")
        
        # Ensure we have some memories and messages
        await self.benchmark_memory_storage(100)
        await self.benchmark_message_publishing(50)
        
        # Create memory assistants
        assistants = []
        for i in range(min(n_contexts, len(self.agents))):
            assistant = MemoryAssistant(self.memory_core)
            await assistant.initialize_for_agent(
                self.agents[i]["id"], 
                self.agents[i]["metadata"]
            )
            assistants.append(assistant)
        
        total_time = 0
        
        for i in range(n_contexts):
            # Select a random assistant
            assistant = random.choice(assistants)
            
            start_time = time.time()
            context = await assistant.get_context(window_size=10)
            elapsed = time.time() - start_time
            
            total_time += elapsed
            
            # Record individual operation time
            self.results["context_retrieval"].append(elapsed)
        
        avg_time = total_time / n_contexts
        logger.info(f"Context retrieval: {avg_time:.6f} seconds per operation (total: {total_time:.6f}s)")
    
    async def run_all_benchmarks(self):
        """Run all benchmark tests"""
        await self.setup()
        
        await self.benchmark_memory_storage(100)
        await self.benchmark_memory_retrieval(100)
        await self.benchmark_message_publishing(100)
        await self.benchmark_query_execution(50)
        await self.benchmark_context_retrieval(50)
        
        logger.info("All benchmarks completed")
        
        return self.results
    
    def print_results(self):
        """Print benchmark results"""
        print("\n" + "="*60)
        print(" "*15 + "MEMORY SYSTEM BENCHMARK RESULTS")
        print("="*60 + "\n")
        
        for operation, times in self.results.items():
            if not times:
                continue
                
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"Operation: {operation}")
            print(f"  Samples: {len(times)}")
            print(f"  Average time: {avg_time:.6f} seconds")
            print(f"  Min time: {min_time:.6f} seconds")
            print(f"  Max time: {max_time:.6f} seconds")
            print(f"  Throughput: {1/avg_time:.2f} ops/second")
            print()
        
        print("="*60)

async def main():
    """Main entry point for memory system benchmarks"""
    print("Starting LumosTrade Memory System Benchmarks...")
    
    # Run benchmarks without Redis
    benchmark = MemorySystemBenchmark(use_redis=False)
    results = await benchmark.run_all_benchmarks()
    benchmark.print_results()
    
    # Check if Redis is available and run with Redis
    try:
        import redis
        r = redis.Redis.from_url("redis://localhost:6379/0")
        r.ping()
        
        print("\nRedis is available. Running benchmarks with Redis...")
        redis_benchmark = MemorySystemBenchmark(use_redis=True)
        redis_results = await redis_benchmark.run_all_benchmarks()
        redis_benchmark.print_results()
    except (ImportError, redis.exceptions.ConnectionError):
        print("\nRedis is not available. Skipping benchmarks with Redis.")
    
    print("\nMemory System benchmarking complete!")

if __name__ == "__main__":
    asyncio.run(main())