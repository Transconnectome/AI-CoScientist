"""Performance benchmark script for RAG optimizations."""

import asyncio
import time
import statistics
from typing import List, Dict
import argparse

from src.services.knowledge_base.search import KnowledgeBaseSearch
from src.services.knowledge_base.search_optimized import OptimizedKnowledgeBaseSearch
from src.services.knowledge_base.vector_store import VectorStore
from src.services.knowledge_base.vector_store_optimized import OptimizedVectorStore
from src.services.knowledge_base.embedding import EmbeddingService
from src.services.knowledge_base.embedding_optimized import OptimizedEmbeddingService
from src.services.knowledge_base.query_cache import QueryCache
from src.core.database import get_db


# Test queries for benchmarking
TEST_QUERIES = [
    "machine learning algorithms",
    "deep neural networks",
    "natural language processing",
    "computer vision techniques",
    "reinforcement learning",
    "supervised learning methods",
    "unsupervised clustering",
    "transfer learning",
    "attention mechanisms",
    "transformer architecture"
]


async def benchmark_original_search(queries: List[str], iterations: int = 3) -> Dict:
    """Benchmark original search implementation.

    Args:
        queries: List of test queries
        iterations: Number of iterations per query

    Returns:
        Dictionary with benchmark results
    """
    print("\n=== Benchmarking Original Search ===")

    # Initialize services
    vector_store = VectorStore()
    embedding_service = EmbeddingService()

    async with get_db() as db:
        search_service = KnowledgeBaseSearch(
            vector_store=vector_store,
            embedding_service=embedding_service,
            db=db
        )

        times = []

        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")

            for query in queries:
                start = time.time()
                await search_service.semantic_search(query, top_k=10)
                elapsed = time.time() - start
                times.append(elapsed * 1000)  # Convert to ms

    return {
        "total_queries": len(queries) * iterations,
        "avg_time_ms": statistics.mean(times),
        "median_time_ms": statistics.median(times),
        "min_time_ms": min(times),
        "max_time_ms": max(times),
        "std_dev_ms": statistics.stdev(times) if len(times) > 1 else 0
    }


async def benchmark_optimized_search(queries: List[str], iterations: int = 3) -> Dict:
    """Benchmark optimized search implementation.

    Args:
        queries: List of test queries
        iterations: Number of iterations per query

    Returns:
        Dictionary with benchmark results
    """
    print("\n=== Benchmarking Optimized Search ===")

    # Initialize optimized services
    vector_store = OptimizedVectorStore()
    embedding_service = OptimizedEmbeddingService()
    query_cache = QueryCache()

    async with get_db() as db:
        search_service = OptimizedKnowledgeBaseSearch(
            vector_store=vector_store,
            embedding_service=embedding_service,
            db=db,
            cache=query_cache
        )

        times = []
        cache_hits = 0
        cache_misses = 0

        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")

            for query in queries:
                start = time.time()
                results, metrics = await search_service.semantic_search(
                    query, top_k=10, use_cache=True
                )
                elapsed = time.time() - start

                times.append(elapsed * 1000)  # Convert to ms

                if metrics.cache_hit:
                    cache_hits += 1
                else:
                    cache_misses += 1

        # Get comprehensive stats
        stats = search_service.get_stats()

        return {
            "total_queries": len(queries) * iterations,
            "avg_time_ms": statistics.mean(times),
            "median_time_ms": statistics.median(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "std_dev_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "cache_hit_rate": cache_hits / (cache_hits + cache_misses),
            "stats": stats
        }


async def benchmark_batch_embedding(batch_sizes: List[int] = [1, 8, 16, 32, 64]) -> Dict:
    """Benchmark embedding batch processing.

    Args:
        batch_sizes: List of batch sizes to test

    Returns:
        Dictionary with benchmark results
    """
    print("\n=== Benchmarking Batch Embedding ===")

    service = OptimizedEmbeddingService()

    results = {}

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")

        # Generate test texts
        texts = [f"test text {i}" for i in range(batch_size)]

        times = []
        for _ in range(5):  # 5 iterations
            start = time.time()
            await service.encode_batch(texts, batch_size=batch_size, use_cache=False)
            elapsed = time.time() - start
            times.append(elapsed * 1000)

        results[batch_size] = {
            "avg_time_ms": statistics.mean(times),
            "throughput": batch_size / (statistics.mean(times) / 1000)  # items/sec
        }

    return results


async def benchmark_concurrent_queries(num_concurrent: List[int] = [1, 5, 10, 20]) -> Dict:
    """Benchmark concurrent query handling.

    Args:
        num_concurrent: List of concurrent query counts

    Returns:
        Dictionary with benchmark results
    """
    print("\n=== Benchmarking Concurrent Queries ===")

    vector_store = OptimizedVectorStore()
    embedding_service = OptimizedEmbeddingService()
    query_cache = QueryCache()

    async with get_db() as db:
        search_service = OptimizedKnowledgeBaseSearch(
            vector_store=vector_store,
            embedding_service=embedding_service,
            db=db,
            cache=query_cache
        )

        results = {}

        for num in num_concurrent:
            print(f"Testing {num} concurrent queries")

            times = []
            for _ in range(3):  # 3 iterations
                start = time.time()

                # Launch concurrent queries
                tasks = [
                    search_service.semantic_search(
                        TEST_QUERIES[i % len(TEST_QUERIES)],
                        top_k=10,
                        use_cache=False
                    )
                    for i in range(num)
                ]

                await asyncio.gather(*tasks)

                elapsed = time.time() - start
                times.append(elapsed * 1000)

            results[num] = {
                "avg_time_ms": statistics.mean(times),
                "queries_per_second": num / (statistics.mean(times) / 1000)
            }

        return results


def print_results(original: Dict, optimized: Dict):
    """Print comparison results.

    Args:
        original: Original implementation results
        optimized: Optimized implementation results
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 70)

    print("\n📊 Query Performance:")
    print(f"{'Metric':<30} {'Original':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-" * 70)

    metrics = [
        ("Average time", "avg_time_ms", "ms"),
        ("Median time", "median_time_ms", "ms"),
        ("Min time", "min_time_ms", "ms"),
        ("Max time", "max_time_ms", "ms"),
        ("Std deviation", "std_dev_ms", "ms")
    ]

    for label, key, unit in metrics:
        orig_val = original[key]
        opt_val = optimized[key]
        improvement = ((orig_val - opt_val) / orig_val) * 100

        print(f"{label:<30} {orig_val:>10.2f} {unit:<4} {opt_val:>10.2f} {unit:<4} {improvement:>10.1f}%")

    if "cache_hit_rate" in optimized:
        print(f"\n💾 Cache Hit Rate: {optimized['cache_hit_rate']*100:.1f}%")

    print(f"\n⚡ Overall Speedup: {original['avg_time_ms'] / optimized['avg_time_ms']:.2f}x")


async def run_full_benchmark():
    """Run full benchmark suite."""
    print("🚀 Starting RAG Performance Benchmark Suite")
    print("=" * 70)

    # 1. Compare original vs optimized
    print("\n1️⃣  Comparing Original vs Optimized Search")
    original_results = await benchmark_original_search(TEST_QUERIES, iterations=3)
    optimized_results = await benchmark_optimized_search(TEST_QUERIES, iterations=3)
    print_results(original_results, optimized_results)

    # 2. Batch embedding benchmark
    print("\n2️⃣  Batch Embedding Performance")
    batch_results = await benchmark_batch_embedding()
    print("\nBatch Processing Results:")
    print(f"{'Batch Size':<15} {'Avg Time (ms)':<20} {'Throughput (items/s)':<20}")
    print("-" * 55)
    for size, metrics in batch_results.items():
        print(f"{size:<15} {metrics['avg_time_ms']:<20.2f} {metrics['throughput']:<20.2f}")

    # 3. Concurrent query benchmark
    print("\n3️⃣  Concurrent Query Handling")
    concurrent_results = await benchmark_concurrent_queries()
    print("\nConcurrent Query Results:")
    print(f"{'Concurrent':<15} {'Avg Time (ms)':<20} {'Queries/sec':<20}")
    print("-" * 55)
    for num, metrics in concurrent_results.items():
        print(f"{num:<15} {metrics['avg_time_ms']:<20.2f} {metrics['queries_per_second']:<20.2f}")

    # 4. Print final stats
    if "stats" in optimized_results:
        print("\n📈 Detailed Performance Statistics:")
        stats = optimized_results["stats"]

        print("\nSearch Stats:")
        for key, value in stats["search"].items():
            print(f"  {key}: {value}")

        print("\nCache Stats:")
        for key, value in stats["cache"].items():
            print(f"  {key}: {value}")

        print("\nEmbedding Stats:")
        for key, value in stats["embedding"].items():
            print(f"  {key}: {value}")

        print("\nVector Store Stats:")
        for key, value in stats["vector_store"].items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("✅ Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RAG performance optimizations")
    parser.add_argument("--queries", type=int, default=10, help="Number of test queries")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per query")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")

    args = parser.parse_args()

    if args.full:
        asyncio.run(run_full_benchmark())
    else:
        # Run quick comparison
        async def quick_benchmark():
            queries = TEST_QUERIES[:args.queries]
            original = await benchmark_original_search(queries, args.iterations)
            optimized = await benchmark_optimized_search(queries, args.iterations)
            print_results(original, optimized)

        asyncio.run(quick_benchmark())
