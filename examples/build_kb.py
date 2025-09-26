#!/usr/bin/env python3
"""Example: Build a knowledge pack from text files."""

import sys
from pathlib import Path

# Add mempack to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mempack import MemPackEncoder
from mempack.config import MemPackConfig


def main():
    """Build a knowledge pack from example data."""
    print("MemPack Knowledge Pack Builder")
    print("=" * 40)
    
    # Create some example data
    example_texts = [
        {
            "text": """
            # Introduction to Quantum Computing
            
            Quantum computing is a revolutionary approach to computation that leverages the principles of quantum mechanics. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits that can exist in superposition states.
            
            The key advantages of quantum computing include:
            - Exponential speedup for certain algorithms
            - Ability to solve problems intractable for classical computers
            - Potential applications in cryptography, optimization, and simulation
            """,
            "meta": {
                "source": "quantum_computing.md",
                "topic": "quantum computing",
                "tags": ["technology", "computing", "quantum"],
            }
        },
        {
            "text": """
            # Machine Learning Fundamentals
            
            Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It has three main types:
            
            1. Supervised Learning: Learning with labeled training data
            2. Unsupervised Learning: Finding patterns in unlabeled data
            3. Reinforcement Learning: Learning through interaction with an environment
            
            Popular algorithms include linear regression, decision trees, neural networks, and support vector machines.
            """,
            "meta": {
                "source": "machine_learning.md",
                "topic": "machine learning",
                "tags": ["AI", "algorithms", "data science"],
            }
        },
        {
            "text": """
            # Python Programming Tips
            
            Python is a versatile programming language known for its simplicity and readability. Here are some best practices:
            
            - Use meaningful variable names
            - Follow PEP 8 style guidelines
            - Write comprehensive docstrings
            - Use type hints for better code documentation
            - Leverage list comprehensions for concise code
            - Handle exceptions properly with try-except blocks
            """,
            "meta": {
                "source": "python_tips.md",
                "topic": "programming",
                "tags": ["python", "programming", "best practices"],
            }
        }
    ]
    
    # Create configuration
    config = MemPackConfig()
    config.chunking.chunk_size = 200
    config.chunking.chunk_overlap = 50
    config.embedding.model = "all-MiniLM-L6-v2"
    config.index.hnsw.M = 16
    config.index.hnsw.ef_construction = 100
    
    # Create encoder
    encoder = MemPackEncoder(config=config)
    
    # Add texts
    print("Adding texts to knowledge pack...")
    for text_data in example_texts:
        encoder.add_text(text_data["text"], text_data["meta"])
    
    print(f"Added {len(encoder.chunks)} chunks")
    
    # Build knowledge pack
    print("\nBuilding knowledge pack...")
    stats = encoder.build(
        pack_path="example_kb.mpack",
        ann_path="example_kb.ann",
    )
    
    print(f"\nBuild completed!")
    print(f"Chunks: {stats.chunks}")
    print(f"Vectors: {stats.vectors}")
    print(f"Build time: {stats.build_time_ms:.2f}ms")
    print(f"Files created:")
    print(f"  - example_kb.mpack")
    print(f"  - example_kb.ann")


if __name__ == "__main__":
    main()
