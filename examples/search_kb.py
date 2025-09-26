#!/usr/bin/env python3
"""Example: Search a knowledge pack."""

import sys
from pathlib import Path

# Add mempack to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mempack import MemPackRetriever


def main():
    """Search the example knowledge pack."""
    print("MemPack Knowledge Pack Search")
    print("=" * 40)
    
    # Check if knowledge pack exists
    pack_path = Path("example_kb.mpack")
    ann_path = Path("example_kb.ann")
    
    if not pack_path.exists() or not ann_path.exists():
        print("Error: Knowledge pack not found!")
        print("Please run build_kb.py first to create the example knowledge pack.")
        return
    
    # Create retriever
    print("Loading knowledge pack...")
    with MemPackRetriever(
        pack_path=pack_path,
        ann_path=ann_path,
        ef_search=32,  # Lower for faster search
    ) as retriever:
        
        # Example queries
        queries = [
            "What is quantum computing?",
            "How does machine learning work?",
            "Python programming best practices",
            "algorithms and data structures",
            "artificial intelligence applications",
        ]
        
        print(f"\nSearching with {len(queries)} queries...\n")
        
        for i, query in enumerate(queries, 1):
            print(f"Query {i}: {query}")
            print("-" * 50)
            
            # Search
            hits = retriever.search(query, top_k=3)
            
            if not hits:
                print("No results found.\n")
                continue
            
            # Display results
            for j, hit in enumerate(hits, 1):
                print(f"{j}. Score: {hit.score:.3f}")
                print(f"   Source: {hit.meta.get('source', 'unknown')}")
                print(f"   Topic: {hit.meta.get('topic', 'unknown')}")
                print(f"   Text: {hit.text[:150]}...")
                print()
            
            print()
        
        # Get statistics
        stats = retriever.get_stats()
        print(f"Search Statistics:")
        print(f"  Total searches: {stats.total_searches}")
        print(f"  Average search time: {stats.avg_search_ms:.2f}ms")
        print(f"  Cache hits: {stats.cache_hits}")
        print(f"  Cache misses: {stats.cache_misses}")


if __name__ == "__main__":
    main()
