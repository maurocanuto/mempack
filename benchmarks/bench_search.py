#!/usr/bin/env python3
"""Benchmark script for MemPack search performance."""

import random
import statistics
import sys
import time
from pathlib import Path

# Add mempack to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mempack import MemPackEncoder, MemPackRetriever
from mempack.config import MemPackConfig


def generate_test_data(num_documents: int = 1000) -> list:
    """Generate test documents.
    
    Args:
        num_documents: Number of documents to generate
        
    Returns:
        List of document dictionaries
    """
    topics = [
        "artificial intelligence", "machine learning", "deep learning",
        "quantum computing", "cryptography", "algorithms",
        "data structures", "programming", "software engineering",
        "computer science", "mathematics", "statistics",
        "neural networks", "natural language processing",
        "computer vision", "robotics", "blockchain",
        "cloud computing", "distributed systems", "databases"
    ]
    
    documents = []
    for i in range(num_documents):
        topic = random.choice(topics)
        content = f"""
        Document {i+1}: {topic.title()}
        
        This is a comprehensive document about {topic}. It covers various aspects including:
        - Fundamental concepts and principles
        - Practical applications and use cases
        - Recent developments and trends
        - Challenges and future directions
        
        The field of {topic} has evolved significantly over the years, with many breakthroughs
        and innovations that have shaped the current landscape. Researchers and practitioners
        continue to push the boundaries of what's possible in this domain.
        
        Key areas of focus include theoretical foundations, practical implementations,
        performance optimization, and real-world applications across different industries.
        """
        
        documents.append({
            "text": content,
            "meta": {
                "id": i + 1,
                "topic": topic,
                "category": random.choice(["research", "tutorial", "news", "review"]),
                "tags": [topic, random.choice(["beginner", "intermediate", "advanced"])],
            }
        })
    
    return documents


def benchmark_build(documents: list, config: MemPackConfig) -> dict:
    """Benchmark knowledge pack building.
    
    Args:
        documents: List of documents
        config: MemPack configuration
        
    Returns:
        Build statistics
    """
    print(f"Building knowledge pack with {len(documents)} documents...")
    
    # Create encoder
    encoder = MemPackEncoder(config=config)
    
    # Add documents
    start_time = time.time()
    for doc in documents:
        encoder.add_text(doc["text"], doc["meta"])
    add_time = time.time() - start_time
    
    # Build knowledge pack
    start_time = time.time()
    stats = encoder.build(
        pack_path="benchmark_kb.mpack",
        ann_path="benchmark_kb.ann",
    )
    build_time = time.time() - start_time
    
    return {
        "documents": len(documents),
        "chunks": stats.chunks,
        "vectors": stats.vectors,
        "add_time": add_time,
        "build_time": build_time,
        "total_time": add_time + build_time,
        "pack_size": Path("benchmark_kb.mpack").stat().st_size,
        "ann_size": Path("benchmark_kb.ann").stat().st_size,
    }


def benchmark_search(queries: list, top_k: int = 10, num_runs: int = 5) -> dict:
    """Benchmark search performance.
    
    Args:
        queries: List of search queries
        top_k: Number of results per query
        num_runs: Number of benchmark runs
        
    Returns:
        Search statistics
    """
    print(f"Benchmarking search with {len(queries)} queries, {num_runs} runs...")
    
    with MemPackRetriever(
        pack_path="benchmark_kb.mpack",
        ann_path="benchmark_kb.ann",
        ef_search=64,
    ) as retriever:
        
        search_times = []
        
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")
            
            run_times = []
            for query in queries:
                start_time = time.time()
                hits = retriever.search(query, top_k=top_k)
                search_time = time.time() - start_time
                run_times.append(search_time)
            
            search_times.extend(run_times)
        
        # Calculate statistics
        search_times_ms = [t * 1000 for t in search_times]
        
        return {
            "queries": len(queries),
            "runs": num_runs,
            "total_searches": len(search_times),
            "avg_time_ms": statistics.mean(search_times_ms),
            "median_time_ms": statistics.median(search_times_ms),
            "p95_time_ms": sorted(search_times_ms)[int(0.95 * len(search_times_ms))],
            "p99_time_ms": sorted(search_times_ms)[int(0.99 * len(search_times_ms))],
            "min_time_ms": min(search_times_ms),
            "max_time_ms": max(search_times_ms),
        }


def main():
    """Run the benchmark."""
    print("MemPack Performance Benchmark")
    print("=" * 50)
    
    # Configuration
    config = MemPackConfig()
    config.chunking.chunk_size = 300
    config.chunking.chunk_overlap = 50
    config.embedding.model = "all-MiniLM-L6-v2"
    config.embedding.batch_size = 32
    config.index.hnsw.M = 32
    config.index.hnsw.ef_construction = 200
    config.index.hnsw.ef_search = 64
    
    # Generate test data
    print("Generating test data...")
    documents = generate_test_data(num_documents=500)
    
    # Test queries
    test_queries = [
        "artificial intelligence applications",
        "machine learning algorithms",
        "quantum computing principles",
        "cryptography and security",
        "data structures and algorithms",
        "neural networks and deep learning",
        "natural language processing",
        "computer vision techniques",
        "blockchain technology",
        "cloud computing architecture",
        "distributed systems design",
        "database optimization",
        "software engineering practices",
        "mathematical foundations",
        "statistical analysis methods",
    ]
    
    # Benchmark build
    print("\n" + "=" * 50)
    build_stats = benchmark_build(documents, config)
    
    print(f"\nBuild Results:")
    print(f"  Documents: {build_stats['documents']}")
    print(f"  Chunks: {build_stats['chunks']}")
    print(f"  Vectors: {build_stats['vectors']}")
    print(f"  Add time: {build_stats['add_time']:.2f}s")
    print(f"  Build time: {build_stats['build_time']:.2f}s")
    print(f"  Total time: {build_stats['total_time']:.2f}s")
    print(f"  Pack size: {build_stats['pack_size']:,} bytes")
    print(f"  ANN size: {build_stats['ann_size']:,} bytes")
    print(f"  Total size: {build_stats['pack_size'] + build_stats['ann_size']:,} bytes")
    
    # Benchmark search
    print("\n" + "=" * 50)
    search_stats = benchmark_search(test_queries, top_k=10, num_runs=3)
    
    print(f"\nSearch Results:")
    print(f"  Queries: {search_stats['queries']}")
    print(f"  Runs: {search_stats['runs']}")
    print(f"  Total searches: {search_stats['total_searches']}")
    print(f"  Average time: {search_stats['avg_time_ms']:.2f}ms")
    print(f"  Median time: {search_stats['median_time_ms']:.2f}ms")
    print(f"  P95 time: {search_stats['p95_time_ms']:.2f}ms")
    print(f"  P99 time: {search_stats['p99_time_ms']:.2f}ms")
    print(f"  Min time: {search_stats['min_time_ms']:.2f}ms")
    print(f"  Max time: {search_stats['max_time_ms']:.2f}ms")
    
    # Performance targets
    print(f"\n" + "=" * 50)
    print("Performance Targets:")
    print(f"  Target p50: ≤ 40ms")
    print(f"  Target p95: ≤ 120ms")
    print(f"  Actual p50: {search_stats['median_time_ms']:.2f}ms")
    print(f"  Actual p95: {search_stats['p95_time_ms']:.2f}ms")
    
    if search_stats['median_time_ms'] <= 40:
        print("  ✓ P50 target met!")
    else:
        print("  ✗ P50 target not met")
    
    if search_stats['p95_time_ms'] <= 120:
        print("  ✓ P95 target met!")
    else:
        print("  ✗ P95 target not met")
    
    # Cleanup
    print(f"\nCleaning up benchmark files...")
    Path("benchmark_kb.mpack").unlink(missing_ok=True)
    Path("benchmark_kb.ann").unlink(missing_ok=True)
    
    print("Benchmark completed!")


if __name__ == "__main__":
    main()
