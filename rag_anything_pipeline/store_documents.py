#!/usr/bin/env python3
"""
RAG-Anything Document Storage Pipeline

This script processes PDF documents and stores them in RAG-Anything's storage.
Run this ONCE per document. The processed documents are cached for future use.

Usage:
    python store_documents.py [--filename FILENAME] [--force]

Example:
    python store_documents.py --filename "attention.pdf"
    python store_documents.py --filename "attention.pdf" --force  # Reprocess even if cached
"""

# %%
# Imports and Setup
import asyncio
import argparse
from pathlib import Path
from typing import Optional

# Import RAG-Anything
from raganything import RAGAnything, RAGAnythingConfig
import nest_asyncio

# Import project config and models
from ai_eval.config import global_config as glob
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Apply nest_asyncio for async support
nest_asyncio.apply()


# %%
# Initialize Models
def initialize_models():
    """Initialize chat and embedding models based on configuration."""
    print("=" * 80)
    print("INITIALIZING MODELS")
    print("=" * 80)

    if glob.MODEL_PROVIDER == "openai":
        chat_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_retries=2,
        )
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
        )
        print("✓ Using OpenAI models")
    else:
        chat_model = ChatOllama(
            model="llama3.1:latest",
            temperature=0.1,
            max_retries=2,
        )
        embedding_model = OllamaEmbeddings(
            model="nomic-embed-text",
        )
        print(f"✓ Using Ollama models (provider: {glob.MODEL_PROVIDER})")

    return chat_model, embedding_model


# %%
# Create RAG-Anything Configuration
def create_rag_anything_config(
    working_dir: str = "./rag_storage",
    parser: str = "mineru",
    parse_method: str = "auto",
) -> RAGAnythingConfig:
    """Create RAG-Anything configuration with optimized settings."""
    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=parser,
        parse_method=parse_method,
        parser_output_dir="./output",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        display_content_stats=True,
    )
    return config


# %%
# Initialize RAG-Anything
def initialize_rag_anything(
    chat_model, embedding_model, config: RAGAnythingConfig
) -> RAGAnything:
    """Initialize RAG-Anything with model functions."""
    print("=" * 80)
    print("INITIALIZING RAG-ANYTHING")
    print("=" * 80)

    # Create LLM function wrapper (async for LightRAG)
    async def llm_func(prompt, **kwargs):
        """Async LLM function for RAG-Anything."""
        # LightRAG expects async, but langchain LLMs are sync
        # We'll run them in a thread pool to make them async
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, chat_model.invoke, prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)
    
    # Add .func attribute for LightRAG's decorator pattern
    llm_func.func = llm_func

    # Create embedding function wrapper with embedding_dim attribute
    # Get embedding dimension first
    sample_embedding = embedding_model.embed_documents(["test"])
    embedding_dim = len(sample_embedding[0])
    
    # Create an async function for LightRAG
    async def embedding_func(texts):
        """Async embedding function for RAG-Anything."""
        if isinstance(texts, str):
            texts = [texts]
        # LightRAG expects async, but langchain embeddings are sync
        # We'll run them in a thread pool to make them async
        import asyncio
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, embedding_model.embed_documents, texts)
        return embeddings
    
    # Add embedding_dim as an attribute to the function
    embedding_func.embedding_dim = embedding_dim
    # Add .func attribute for LightRAG's decorator pattern
    embedding_func.func = embedding_func

    # Initialize RAG-Anything
    rag_anything = RAGAnything(
        config=config,
        llm_model_func=llm_func,
        embedding_func=embedding_func,
    )

    print("✓ RAG-Anything initialized successfully!")
    print(f"  Working directory: {config.working_dir}")
    print(f"  Parser: {config.parser}")
    print(f"  Parse method: {config.parse_method}")

    return rag_anything


# %%
# Process Document (Async)
async def process_document(
    rag_anything: RAGAnything,
    file_path: str,
    output_dir: str,
    parse_method: str = "auto",
) -> None:
    """Process a document and store it in RAG-Anything storage."""
    await rag_anything.process_document_complete(
        file_path=file_path,
        output_dir=output_dir,
        parse_method=parse_method,
        display_stats=True,
    )


# %%
# Main Storage Function
def store_document(
    filename: str,
    force_reprocess: bool = False,
    output_dir: Optional[str] = None,
) -> bool:
    """
    Process and store a document in RAG-Anything storage.

    Args:
        filename: Name of the PDF file in the data directory
        force_reprocess: If True, reprocess even if already cached
        output_dir: Output directory for processed documents (default: ./output/raganything_processed)

    Returns:
        True if processing was successful, False if skipped (already processed)
    """
    print("=" * 80)
    print("RAG-ANYTHING DOCUMENT STORAGE")
    print("=" * 80)
    print(f"Document: {filename}")
    print()

    # Setup paths
    file_path = Path(glob.DATA_PKG_DIR) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if output_dir is None:
        output_dir = "./output/raganything_processed"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_marker = output_path / ".processed"

    # Check if already processed
    if processed_marker.exists() and not force_reprocess:
        print("✓ Document already processed and stored!")
        print(f"  Storage location: {output_dir}")
        print("  RAG-Anything working directory: ./rag_storage")
        print("\n  To reprocess, run with --force flag")
        return False

    if force_reprocess and processed_marker.exists():
        print("⚠️  Force reprocessing enabled. Removing existing cache...")
        processed_marker.unlink()

    # Initialize models
    chat_model, embedding_model = initialize_models()
    print()

    # Create configuration
    config = create_rag_anything_config()

    # Initialize RAG-Anything
    rag_anything = initialize_rag_anything(chat_model, embedding_model, config)
    print()

    # Process document
    print("=" * 80)
    print("PROCESSING DOCUMENT")
    print("=" * 80)
    print(f"File: {file_path}")
    print(f"Output directory: {output_dir}")
    print("Extracting: text, tables, formulas, and images...")
    print("\nThis may take several minutes...")
    print()

    try:
        asyncio.run(
            process_document(
                rag_anything=rag_anything,
                file_path=str(file_path),
                output_dir=output_dir,
                parse_method="auto",
            )
        )

        # Mark as processed
        processed_marker.touch()

        print()
        print("=" * 80)
        print("✓ DOCUMENT PROCESSING COMPLETE!")
        print("=" * 80)
        print(f"  Processed content stored in: {output_dir}")
        print(f"  RAG-Anything storage: {config.working_dir}")
        print("\n  You can now use query_documents.py to query the stored documents.")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback

        traceback.print_exc()
        raise


# %%
# Command-Line Interface
def main():
    """Main entry point."""
    # Check if running in interactive environment (Jupyter/IPython)
    # Skip argument parsing in interactive mode to avoid conflicts
    try:
        # Check if we're in IPython/Jupyter
        import sys

        if "ipykernel" in sys.modules or "IPython" in sys.modules:
            # Running in Jupyter/IPython - skip argument parsing
            print(
                "Running in interactive mode. Use the example cell below or call store_document() directly."
            )
            return 0
    except Exception:
        # Not in IPython - continue with argument parsing
        pass

    parser = argparse.ArgumentParser(
        description="Process and store documents in RAG-Anything storage"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="attention.pdf",
        help="Name of the PDF file in the data directory (default: attention.pdf)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if document is already cached",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed documents (default: ./output/raganything_processed)",
    )

    args = parser.parse_args()

    try:
        store_document(
            filename=args.filename,
            force_reprocess=args.force,
            output_dir=args.output_dir,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1

    return 0


# %%
# Execute Main (when running as script)
if __name__ == "__main__":
    exit(main())

# %%
# Example: Run storage interactively
# Uncomment and modify the following to run in Jupyter/VS Code:
# filename = "attention.pdf"  # Change to your PDF filename
# force_reprocess = False  # Set to True to reprocess even if cached
# store_document(filename=filename, force_reprocess=force_reprocess)
