#!/usr/bin/env python3
"""
RAG-Anything Document Query Pipeline

This script queries documents stored in RAG-Anything storage and generates answers.
You can run this multiple times with different queries.

Usage:
    python query_documents.py [--query "Your question here"] [--interactive]

Examples:
    python query_documents.py --query "How do I use patterns in Allplan?"
    python query_documents.py --interactive  # Interactive mode for multiple queries
"""

# %%
# Imports and Setup
import os
import asyncio
import argparse
from pathlib import Path
from typing import Optional, List

# Import RAG-Anything
from raganything import RAGAnything, RAGAnythingConfig
from ai_eval.resources.rag_template import RAGAnythingRAG
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
    if glob.MODEL_PROVIDER == "openai":
        chat_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_retries=2,
        )
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
        )
    else:
        chat_model = ChatOllama(
            model="llama3.1:latest",
            temperature=0.1,
            max_retries=2,
        )
        embedding_model = OllamaEmbeddings(
            model="nomic-embed-text",
        )

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
        display_content_stats=False,  # Don't show stats during queries
    )
    return config


# %%
# Initialize RAG-Anything Instance
def initialize_rag_anything(
    chat_model, embedding_model, config: RAGAnythingConfig
) -> RAGAnything:
    """Initialize RAG-Anything with model functions."""

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
        embeddings = await loop.run_in_executor(
            None, embedding_model.embed_documents, texts
        )
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

    return rag_anything


# %%
# Verify Storage
def verify_storage(output_dir: str = "./output/raganything_processed") -> bool:
    """Verify that documents have been processed and stored."""
    processed_marker = Path(output_dir) / ".processed"

    if not processed_marker.exists():
        print("=" * 80)
        print("❌ ERROR: Documents not processed yet!")
        print("=" * 80)
        print(f"\nStorage location: {output_dir}")
        print("\nPlease run store_documents.py first to process and store documents.")
        print("\nExample:")
        print("  python store_documents.py --filename 'your_document.pdf'")
        return False

    return True


# %%
# Create RAG Instance for Querying
def create_rag_instance(
    chat_model, embedding_model, config: RAGAnythingConfig
) -> RAGAnythingRAG:
    """Create a RAG-Anything RAG instance for querying."""
    # Initialize RAG-Anything
    rag_anything = initialize_rag_anything(chat_model, embedding_model, config)

    # Create RAG instance
    rag_anything_rag = RAGAnythingRAG(
        llm=chat_model,
        documents=None,
        k=3,  # Number of documents to retrieve
        rag_anything_instance=rag_anything,
        processed_docs_cache=None,
    )

    return rag_anything_rag


# %%
# Query Document Function
def query_document(
    query: str,
    rag_instance: RAGAnythingRAG,
    show_context: bool = True,
) -> tuple[str, List]:
    """
    Query stored documents and get an answer.

    Args:
        query: The question to ask
        rag_instance: Initialized RAG-Anything RAG instance
        show_context: Whether to show retrieved context documents

    Returns:
        Tuple of (answer, list of relevant documents)
    """
    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    print(f"Question: {query}\n")

    # Retrieve relevant documents
    print("Retrieving relevant documents...")
    try:
        relevant_docs = rag_instance.retrieve(question=query)
        print(f"✓ Retrieved {len(relevant_docs)} documents")

        if show_context:
            print("\nRetrieved Context:")
            print("-" * 80)
            for i, doc in enumerate(relevant_docs, 1):
                preview = (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                )
                print(f"\n[{i}] {preview}")
                print(f"    Source: {doc.metadata.get('source', 'unknown')}")
            print("-" * 80)
            print()

    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        import traceback

        traceback.print_exc()
        raise

    # Generate answer
    print("Generating answer...")
    try:
        answer, relevant_docs = rag_instance.answer(question=query)

        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        print(f"\n✓ Used {len(relevant_docs)} relevant documents for context")

        return answer, relevant_docs

    except Exception as e:
        print(f"❌ Error during answer generation: {e}")
        import traceback

        traceback.print_exc()
        raise


# %%
# Interactive Query Mode
def interactive_mode(rag_instance: RAGAnythingRAG):
    """Run in interactive mode for multiple queries."""
    print("=" * 80)
    print("INTERACTIVE QUERY MODE")
    print("=" * 80)
    print("Enter your questions (type 'quit' or 'exit' to stop)")
    print()

    while True:
        try:
            query = input("Query: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            if not query:
                continue

            print()
            query_document(query, rag_instance, show_context=False)
            print("\n" + "-" * 80 + "\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


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
                "Running in interactive mode. Use the example cell below or call query functions directly."
            )
            return 0
    except Exception:
        # Not in IPython - continue with argument parsing
        pass

    parser = argparse.ArgumentParser(
        description="Query documents stored in RAG-Anything storage"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="The question to ask about the documents",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode for multiple queries",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/raganything_processed",
        help="Output directory where processed documents are stored",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default="./rag_storage",
        help="RAG-Anything working directory (default: ./rag_storage)",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved context documents",
    )

    args = parser.parse_args()

    # Verify storage
    if not verify_storage(args.output_dir):
        return 1

    # Initialize models
    print("Initializing models...")
    chat_model, embedding_model = initialize_models()

    # Create configuration
    config = create_rag_anything_config(working_dir=args.working_dir)

    # Create RAG instance
    print("Initializing RAG-Anything...")
    rag_instance = create_rag_instance(chat_model, embedding_model, config)
    print("✓ Ready to query!\n")

    # Run query or interactive mode
    if args.interactive:
        interactive_mode(rag_instance)
    elif args.query:
        query_document(args.query, rag_instance, show_context=args.show_context)
    else:
        # Default query
        default_query = "How can I control the height of a pattern element in the layout to be a specific size, independent of the drawing scale, in Allplan 2020?"
        print("No query provided. Using default query:\n")
        query_document(default_query, rag_instance, show_context=args.show_context)

    return 0


# %%
# Execute Main (when running as script)
if __name__ == "__main__":
    exit(main())

# %%
# Example: Query documents interactively
# Uncomment and modify the following to run in Jupyter/VS Code:
#
# # Step 1: Verify storage
# output_dir = "./output/raganything_processed"
# if not verify_storage(output_dir):
#     print("Please run store_documents.py first!")
#
# # Step 2: Initialize
# chat_model, embedding_model = initialize_models()
# config = create_rag_anything_config()
# rag_instance = create_rag_instance(chat_model, embedding_model, config)
#
# # Step 3: Query
# query = "How do I control pattern height in Allplan?"
# answer, docs = query_document(query, rag_instance, show_context=True)


# %%
