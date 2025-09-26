"""Command-line interface for MemPack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .builder import MemPackEncoder
from .chat import MemPackChat
from .config import MemPackConfig, get_default_config
from .errors import MemPackError
from .logging import cli_logger, setup_logger
from .retriever import MemPackRetriever

# Initialize Typer app
app = typer.Typer(
    name="mempack",
    help="MemPack: A portable, fast knowledge pack with two-file ANN memory",
    no_args_is_help=True,
)

# Initialize Rich console
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Set up logging for CLI."""
    level = "DEBUG" if verbose else "INFO"
    setup_logger("mempack", level=level)


@app.command()
def build(
    src: Path = typer.Option(..., "--src", "-s", help="Source directory or file"),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory"),
    chunk_size: int = typer.Option(300, "--chunk-size", help="Chunk size in characters"),
    chunk_overlap: int = typer.Option(50, "--chunk-overlap", help="Chunk overlap in characters"),
    embed_model: str = typer.Option("all-MiniLM-L6-v2", "--embed-model", help="Embedding model"),
    index_type: str = typer.Option("hnsw", "--index", help="Index type (hnsw)"),
    M: int = typer.Option(32, "--M", help="HNSW M parameter"),
    efc: int = typer.Option(200, "--efc", help="HNSW ef_construction parameter"),
    batch_size: int = typer.Option(64, "--batch-size", help="Embedding batch size"),
    workers: int = typer.Option(0, "--workers", help="Number of worker threads (0=auto)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Build a MemPack knowledge pack from source files."""
    setup_logging(verbose)
    
    try:
        # Create output directory
        out.mkdir(parents=True, exist_ok=True)
        
        # Set up paths
        pack_path = out / "kb.mpack"
        ann_path = out / "kb.ann"
        
        # Create configuration
        config = get_default_config()
        config.chunking.chunk_size = chunk_size
        config.chunking.chunk_overlap = chunk_overlap
        config.embedding.model = embed_model
        config.embedding.batch_size = batch_size
        config.index.type = index_type
        config.index.hnsw.M = M
        config.index.hnsw.ef_construction = efc
        config.workers = workers
        
        # Create encoder
        encoder = MemPackEncoder(config=config)
        
        # Add source files
        if src.is_file():
            console.print(f"Adding file: {src}")
            encoder.add_file(src)
        elif src.is_dir():
            console.print(f"Adding directory: {src}")
            encoder.add_directory(src)
        else:
            console.print(f"[red]Error: {src} is not a file or directory[/red]")
            raise typer.Exit(1)
        
        # Build knowledge pack
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Building knowledge pack...", total=None)
            
            stats = encoder.build(
                pack_path=pack_path,
                ann_path=ann_path,
                embed_batch_size=batch_size,
                workers=workers,
            )
        
        # Display results
        console.print(f"\n[green]✓ Knowledge pack built successfully![/green]")
        console.print(f"Pack file: {pack_path}")
        console.print(f"Index file: {ann_path}")
        console.print(f"Chunks: {stats.chunks}")
        console.print(f"Vectors: {stats.vectors}")
        console.print(f"Build time: {stats.build_time_ms:.2f}ms")
        
    except MemPackError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def search(
    kb: Path = typer.Option(..., "--kb", "-k", help="Knowledge pack directory"),
    query: str = typer.Option(..., "--query", "-q", help="Search query"),
    topk: int = typer.Option(5, "--topk", help="Number of results"),
    ef_search: int = typer.Option(64, "--ef-search", help="HNSW ef_search parameter"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Search a MemPack knowledge pack."""
    setup_logging(verbose)
    
    try:
        # Set up paths
        if kb.is_dir():
            pack_path = kb / "kb.mpack"
            ann_path = kb / "kb.ann"
        else:
            # If kb is a file, use it directly
            pack_path = kb
            ann_path = kb.with_suffix('.ann')
        
        if not pack_path.exists() or not ann_path.exists():
            console.print(f"[red]Error: Knowledge pack not found: {pack_path} or {ann_path}[/red]")
            raise typer.Exit(1)
        
        # Create retriever
        with MemPackRetriever(
            pack_path=pack_path,
            ann_path=ann_path,
            ef_search=ef_search,
        ) as retriever:
            # Search
            hits = retriever.search(query, top_k=topk)
            
            if not hits:
                console.print("[yellow]No results found[/yellow]")
                return
            
            # Display results
            table = Table(title=f"Search Results for: {query}")
            table.add_column("Score", justify="right", style="cyan")
            table.add_column("ID", justify="right", style="magenta")
            table.add_column("Source", style="green")
            table.add_column("Text", style="white")
            
            for hit in hits:
                source = hit.meta.get("source", "unknown")
                text = hit.text[:100] + "..." if len(hit.text) > 100 else hit.text
                table.add_row(
                    f"{hit.score:.3f}",
                    str(hit.id),
                    source,
                    text,
                )
            
            console.print(table)
            
    except MemPackError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def verify(
    kb: Path = typer.Option(..., "--kb", "-k", help="Knowledge pack directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Verify the integrity of a MemPack knowledge pack."""
    setup_logging(verbose)
    
    try:
        # Set up paths
        if kb.is_dir():
            pack_path = kb / "kb.mpack"
            ann_path = kb / "kb.ann"
        else:
            # If kb is a file, use it directly
            pack_path = kb
            ann_path = kb.with_suffix('.ann')
        
        if not pack_path.exists() or not ann_path.exists():
            console.print(f"[red]Error: Knowledge pack not found: {pack_path} or {ann_path}[/red]")
            raise typer.Exit(1)
        
        # Create retriever
        with MemPackRetriever(
            pack_path=pack_path,
            ann_path=ann_path,
        ) as retriever:
            # Verify
            is_valid = retriever.verify()
            
            if is_valid:
                console.print("[green]✓ Knowledge pack is valid[/green]")
                
                # Display stats
                pack_stats = retriever.get_pack_stats()
                index_stats = retriever.get_index_stats()
                
                console.print(f"Chunks: {pack_stats['total_chunks']}")
                console.print(f"Blocks: {pack_stats['total_blocks']}")
                console.print(f"Vectors: {index_stats['current_elements']}")
                console.print(f"Dimensions: {index_stats['dimensions']}")
            else:
                console.print("[red]✗ Knowledge pack is corrupted[/red]")
                raise typer.Exit(1)
                
    except MemPackError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def info(
    kb: Path = typer.Option(..., "--kb", "-k", help="Knowledge pack directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Display information about a MemPack knowledge pack."""
    setup_logging(verbose)
    
    try:
        # Set up paths
        if kb.is_dir():
            pack_path = kb / "kb.mpack"
            ann_path = kb / "kb.ann"
        else:
            # If kb is a file, use it directly
            pack_path = kb
            ann_path = kb.with_suffix('.ann')
        
        if not pack_path.exists() or not ann_path.exists():
            console.print(f"[red]Error: Knowledge pack not found: {pack_path} or {ann_path}[/red]")
            raise typer.Exit(1)
        
        # Create retriever
        with MemPackRetriever(
            pack_path=pack_path,
            ann_path=ann_path,
        ) as retriever:
            # Get information
            pack_stats = retriever.get_pack_stats()
            index_stats = retriever.get_index_stats()
            pack_config = retriever.pack_reader.get_config()
            
            # Display information
            console.print(f"[bold]Knowledge Pack Information[/bold]")
            console.print(f"Pack file: {pack_path}")
            console.print(f"Index file: {ann_path}")
            console.print()
            
            console.print(f"[bold]Content[/bold]")
            console.print(f"Chunks: {pack_stats['total_chunks']}")
            console.print(f"Blocks: {pack_stats['total_blocks']}")
            console.print(f"Vectors: {index_stats['current_elements']}")
            console.print(f"Dimensions: {index_stats['dimensions']}")
            console.print()
            
            console.print(f"[bold]Configuration[/bold]")
            console.print(f"Embedding model: {pack_config.get('embedding_model', 'unknown')}")
            console.print(f"Chunk size: {pack_config.get('chunk_size', 'unknown')}")
            console.print(f"Chunk overlap: {pack_config.get('chunk_overlap', 'unknown')}")
            console.print(f"Compressor: {pack_config.get('compressor', 'unknown')}")
            console.print()
            
            console.print(f"[bold]File Sizes[/bold]")
            console.print(f"Pack file: {pack_path.stat().st_size:,} bytes")
            console.print(f"Index file: {ann_path.stat().st_size:,} bytes")
            console.print(f"Total: {pack_path.stat().st_size + ann_path.stat().st_size:,} bytes")
            
    except MemPackError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def export(
    kb: Path = typer.Option(..., "--kb", "-k", help="Knowledge pack directory"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file"),
    format: str = typer.Option("jsonl", "--format", "-f", help="Output format (jsonl, json)"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit number of chunks"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Export chunks from a MemPack knowledge pack."""
    setup_logging(verbose)
    
    try:
        # Set up paths
        if kb.is_dir():
            pack_path = kb / "kb.mpack"
            ann_path = kb / "kb.ann"
        else:
            # If kb is a file, use it directly
            pack_path = kb
            ann_path = kb.with_suffix('.ann')
        
        if not pack_path.exists() or not ann_path.exists():
            console.print(f"[red]Error: Knowledge pack not found: {pack_path} or {ann_path}[/red]")
            raise typer.Exit(1)
        
        # Create retriever
        with MemPackRetriever(
            pack_path=pack_path,
            ann_path=ann_path,
        ) as retriever:
            # Get all chunks
            chunks = retriever.pack_reader.search_chunks()
            
            if limit:
                chunks = chunks[:limit]
            
            # Export
            if format == "jsonl":
                with open(output, 'w', encoding='utf-8') as f:
                    for chunk in chunks:
                        data = {
                            "id": chunk.id,
                            "text": chunk.text,
                            "meta": chunk.meta.__dict__,
                        }
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
            elif format == "json":
                data = []
                for chunk in chunks:
                    data.append({
                        "id": chunk.id,
                        "text": chunk.text,
                        "meta": chunk.meta.__dict__,
                    })
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                console.print(f"[red]Error: Unsupported format: {format}[/red]")
                raise typer.Exit(1)
            
            console.print(f"[green]✓ Exported {len(chunks)} chunks to {output}[/green]")
            
    except MemPackError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def chat(
    kb: Path = typer.Option(..., "--kb", "-k", help="Knowledge pack directory"),
    query: str = typer.Option(..., "--query", "-q", help="Chat query"),
    context_chunks: int = typer.Option(8, "--context-chunks", help="Number of chunks to use as context"),
    max_context_length: int = typer.Option(2000, "--max-context", help="Maximum context length in characters"),
    ef_search: int = typer.Option(64, "--ef-search", help="HNSW ef_search parameter"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Chat with a MemPack knowledge pack using context retrieval."""
    setup_logging(verbose)
    
    try:
        # Find .mpack and .ann files
        if kb.is_dir():
            mpack_file = kb / f"{kb.name}.mpack"
            ann_file = kb / f"{kb.name}.ann"
        else:
            # If kb is a file, use it directly
            mpack_file = kb
            ann_file = kb.with_suffix('.ann')
        
        if not mpack_file.exists():
            console.print(f"[red]Error: MemPack file not found: {mpack_file}[/red]")
            raise typer.Exit(1)
        
        if not ann_file.exists():
            console.print(f"[red]Error: ANN file not found: {ann_file}[/red]")
            raise typer.Exit(1)
        
        # Initialize retriever
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading knowledge pack...", total=None)
            
            retriever = MemPackRetriever(
                pack_path=mpack_file,
                ann_path=ann_file,
                ef_search=ef_search,
            )
            
            # Initialize chat interface
            chat = MemPackChat(
                retriever=retriever,
                context_chunks=context_chunks,
                max_context_length=max_context_length,
            )
            
            progress.update(task, description="Processing query...")
            
            # Get response
            response = chat.chat(query)
            
            progress.update(task, description="Complete!")
        
        # Display response
        console.print(f"\n[bold blue]Query:[/bold blue] {query}")
        console.print(f"[bold green]Response:[/bold green] {response}")
        
        # Show context sources if verbose
        if verbose:
            # Get the context from the last search
            hits = chat.retriever.search(query, top_k=context_chunks)
            console.print(f"\n[bold yellow]Context sources ({len(hits)} chunks):[/bold yellow]")
            for i, hit in enumerate(hits, 1):
                console.print(f"  {i}. [dim]ID {hit.id} (score: {hit.score:.3f})[/dim]: {hit.text[:100]}...")
        
    except MemPackError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
