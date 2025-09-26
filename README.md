# MemPack

A portable, fast knowledge pack with two-file ANN memory for semantic search and retrieval.

## Overview

MemPack is a Python library that packages text chunks + metadata + integrity info into **one container file** (`.mpack`) and a **separate ANN index** (`.ann`). It's designed for portability, deterministic random access, fast semantic retrieval, and clean APIs.

At its heart, mempack is a knowledge container that works like a hybrid between a structured archive and a vector database:

- Container file **(.mpack)** – Holds compressed text chunks, metadata, and integrity checks.

- Index file **(.ann)** – Stores a memory-mappable Approximate Nearest Neighbor (ANN) index (e.g., HNSW) for fast retrieval.

This separation ensures that data remains portable, compact, and deterministic, while the index is directly mmap-able for lightning-fast loading and search.

### Why MemPack?

- **Two-file format**: Clean separation of data (`.mpack`) and index (`.ann`)
- **Fast retrieval**: Sub-100ms vector search with HNSW indexing
- **Portable**: No database dependencies, works with just files
- **Integrity**: Built-in checksums and optional ECC error correction
- **Memory efficient**: Memory-mappable index with block caching

## Quick Start

### Installation

```bash
pip install mempack
```

### Basic Usage

```python
from mempack import MemPackEncoder, MemPackRetriever

# Build a knowledge pack
encoder = MemPackEncoder(chunk_size=300, chunk_overlap=50)
encoder.add_text("# Introduction\nQuantum computers use qubits...", 
                 meta={"source": "notes/quantum.md"})
encoder.build(pack_path="kb.mpack", ann_path="kb.ann")

# Search the knowledge pack
retriever = MemPackRetriever(pack_path="kb.mpack", ann_path="kb.ann")
hits = retriever.search("quantum computing", top_k=5)
for hit in hits:
    print(f"Score: {hit.score:.3f}")
    print(f"Source: {hit.meta.get('source')}")
    print(f"Text: {hit.text[:120]}...")
    print()
```

### CLI Usage

```bash
# Build from a folder of markdown/text files
mempack build --src ./notes --out ./kb \
  --chunk-size 300 --chunk-overlap 50 \
  --embed-model all-MiniLM-L6-v2

# Search the knowledge pack
mempack search --kb ./kb --q "quantum computing" --topk 5

# Verify integrity
mempack verify --kb ./kb
```

## Two-File Format

### `kb.mpack` — Container File
- **Header**: Magic bytes, version, flags, section offsets
- **Config**: Embedding model, dimensions, compression settings
- **TOC**: Chunk metadata, block information, optional tag index
- **Blocks**: Compressed text chunks (Zstd by default)
- **Checksums**: Per-block integrity verification
- **ECC**: Optional Reed-Solomon error correction

### `kb.ann` — ANN Index File
- **Header**: Magic bytes, algorithm (HNSW), dimensions, parameters
- **Payload**: Memory-mappable HNSW graph structure
- **IDs**: Mapping from vector IDs to chunk IDs

## Performance

- **Search latency**: p50 ≤ 40ms, p95 ≤ 120ms (1M vectors, 384-dim, HNSW)
- **Block fetch**: ≤ 1.5ms typical (zstd decompression)
- **Memory usage**: Efficient block caching with LRU eviction

## API Reference

### MemPackEncoder

```python
class MemPackEncoder:
    def __init__(
        self,
        *,
        compressor: str = "zstd",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        embedding_backend: Optional[EmbeddingBackend] = None,
        index_type: str = "hnsw",
        index_params: Optional[dict] = None,
        ecc: Optional[dict] = None,
        progress: bool = True,
    ): ...

    def add_text(self, text: str, meta: Optional[dict] = None) -> None: ...
    def add_chunks(self, chunks: list[dict] | list[str]) -> None: ...
    def build(
        self,
        *,
        pack_path: str,
        ann_path: str,
        embed_batch_size: int = 64,
        workers: int = 0
    ) -> BuildStats: ...
```

### MemPackRetriever

```python
class MemPackRetriever:
    def __init__(
        self,
        *,
        pack_path: str,
        ann_path: str,
        embedding_backend: Optional[EmbeddingBackend] = None,
        mmap: bool = True,
        block_cache_size: int = 1024,
        io_batch_size: int = 64,
        ef_search: int = 64,
        prefetch: bool = True,
    ): ...

    def search(self, query: str, top_k: int = 5, filter_meta: Optional[dict] = None) -> list[SearchHit]: ...
    def get_chunk_by_id(self, chunk_id: int) -> dict: ...
    def stats(self) -> RetrieverStats: ...
```

## Configuration

### HNSW Parameters

- `M`: Number of bi-directional links (default: 32)
- `efConstruction`: Size of dynamic candidate list (default: 200)
- `efSearch`: Size of dynamic candidate list during search (default: 64)

### Compression

- `zstd`: Fast compression with good ratio (default)
- `deflate`: Standard gzip compression
- `none`: No compression

### Chunking

- `chunk_size`: Target chunk size in characters (default: 300)
- `chunk_overlap`: Overlap between chunks (default: 50)

## Integrity & Error Correction

MemPack includes built-in integrity checking with XXH3 checksums per block. Optional Reed-Solomon error correction can be enabled:

```python
encoder = MemPackEncoder(ecc={"k": 10, "m": 2})  # 10 data + 2 parity blocks
```

## Development

### Setup

```bash
git clone https://github.com/mempack/mempack
cd mempack
pip install -e ".[dev]"
```

### Testing

```bash
make test
```

### Linting

```bash
make lint
```

### Benchmarks

```bash
make bench
```

## License

MIT License - see LICENSE file for details.

## Roadmap

- [ ] IVF-PQ backend for ultra-large corpora
- [ ] Quantized vectors (int8) support
- [ ] Streaming append API
- [ ] HTTP server for remote access
- [ ] More embedding backends (OpenAI, Vertex AI)
