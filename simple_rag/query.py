#!/usr/bin/env python3
"""
Simple RAG-Anything Query

Queries stored documents and returns answers.

Usage:
    python query.py "Your question here"
    python query.py "How do I use patterns in Allplan?"
"""

import sys
import asyncio
import warnings
from pathlib import Path

# Suppress google-cloud-storage deprecation warning
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="google.cloud.aiplatform"
)

from raganything import RAGAnything, RAGAnythingConfig
import nest_asyncio
from ai_eval.resources.rag_template import RAGAnythingRAG
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq

# Import Gemini and Anthropic (with fallback if not installed)
try:
    from langchain_google_vertexai import ChatVertexAI
except ImportError:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI as ChatVertexAI
    except ImportError:
        ChatVertexAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

# Import config - handle both running from parent directory and from simple_rag directory
try:
    from simple_rag import config
except ImportError:
    # If running from simple_rag directory, import config directly
    import config

nest_asyncio.apply()


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in file paths."""
    # Replace special characters with underscores
    return model_name.replace("/", "_").replace(":", "_").replace(" ", "_")


def get_model_paths(
    chat_model_name: str = None,
    embedding_model_name: str = None,
    parser_name: str = None,
) -> tuple[str, str]:
    """Generate storage and output paths based on model names and parser."""
    chat_model_name = chat_model_name or config.query_chat_model
    embedding_model_name = embedding_model_name or config.query_embedding_model
    parser_name = parser_name or config.parser

    # Sanitize model names and parser for file paths
    chat_safe = sanitize_model_name(chat_model_name)
    embedding_safe = sanitize_model_name(embedding_model_name)
    parser_safe = sanitize_model_name(parser_name)

    model_dir = f"{chat_safe}_{embedding_safe}_{parser_safe}"
    working_dir = f"./rag_storage_{model_dir}"
    output_dir = f"./output_{model_dir}/raganything_processed"

    return working_dir, output_dir


def get_models(chat_provider: str = None, embedding_provider: str = None):
    """Initialize chat and embedding models from config."""
    # Use config values if not provided
    chat_provider = chat_provider or config.query_chat_provider
    embedding_provider = embedding_provider or config.query_embedding_provider
    print(f"Using chat provider: {chat_provider}")
    print(f"Using embedding provider: {embedding_provider}")

    # Get LLM settings from config (query can use slightly higher temperature for natural responses)
    temperature = getattr(
        config, "llm_temperature", 0.1
    )  # Slightly higher for querying
    max_tokens = getattr(config, "llm_max_tokens", 4000)
    max_retries = getattr(config, "llm_max_retries", 3)

    # Initialize chat model
    if chat_provider == "openai":
        chat_model = ChatOpenAI(
            model=config.query_chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    elif chat_provider == "groq":
        chat_model = ChatGroq(
            model=config.query_chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif chat_provider == "gemini":
        if ChatVertexAI is None:
            raise ImportError(
                "Gemini models require langchain-google-vertexai or langchain-google-genai. "
                "Install with: pip install langchain-google-vertexai"
            )
        chat_model = ChatVertexAI(
            model=config.query_chat_model,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    elif chat_provider == "anthropic":
        if ChatAnthropic is None:
            raise ImportError(
                "Anthropic models require langchain-anthropic. "
                "Install with: pip install langchain-anthropic"
            )
        chat_model = ChatAnthropic(
            model=config.query_chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:  # ollama
        chat_model = ChatOllama(
            model=config.query_chat_model,
            temperature=temperature,
            num_predict=max_tokens,
        )

    # Initialize embedding model
    if embedding_provider == "openai":
        embedding_model = OpenAIEmbeddings(model=config.query_embedding_model)
    else:  # ollama
        embedding_model = OllamaEmbeddings(model=config.query_embedding_model)

    return chat_model, embedding_model


def create_llm_func(chat_model):
    """Create async LLM function wrapper."""

    async def llm_func(prompt, **kwargs):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, chat_model.invoke, prompt)
        return response.content if hasattr(response, "content") else str(response)

    llm_func.func = llm_func
    return llm_func


def create_embedding_func(embedding_model):
    """Create async embedding function wrapper."""
    sample_embedding = embedding_model.embed_documents(["test"])
    embedding_dim = len(sample_embedding[0])

    async def embedding_func(texts):
        if isinstance(texts, str):
            texts = [texts]
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, embedding_model.embed_documents, texts)

    embedding_func.embedding_dim = embedding_dim
    embedding_func.func = embedding_func
    return embedding_func


def query(
    query_text: str,
    chat_provider: str = None,
    embedding_provider: str = None,
    working_dir: str = None,
    show_retrieved_docs: bool = True,
) -> tuple[str, list]:
    """
    Query stored documents and return answer.

    Args:
        query_text: The question to ask
        chat_provider: Chat model provider (openai, groq, gemini, anthropic, or ollama)
        embedding_provider: Embedding model provider (openai or ollama)
        working_dir: RAG-Anything working directory (auto-generated from config if not provided)
        show_retrieved_docs: Whether to print retrieved document contents (default: True)

    Returns:
        Tuple of (answer string, list of relevant Document objects)
    """
    # Get model paths from config
    if working_dir is None:
        if config.query_rag_storage_path:
            working_dir = config.query_rag_storage_path
        else:
            working_dir, _ = get_model_paths(
                config.query_chat_model, config.query_embedding_model, config.parser
            )

    # Check if storage exists
    storage_path = Path(working_dir)
    if not storage_path.exists() or not list(storage_path.glob("*.json")):
        raise FileNotFoundError(
            f"Storage not found at {working_dir}. "
            "Please run store.py first to process documents."
        )

    # Initialize models from config
    chat_model, embedding_model = get_models(chat_provider, embedding_provider)

    # Check embedding dimension before initializing RAG-Anything
    # This helps catch dimension mismatches early
    embedding_func = create_embedding_func(embedding_model)
    expected_dim = embedding_func.embedding_dim

    # Check if storage exists and has a dimension mismatch
    try:
        import json

        # Check vector database files for stored dimension
        vdb_files = list(storage_path.glob("vdb_*.json"))
        if vdb_files:
            # Try to read the first vdb file to check dimension
            with open(vdb_files[0], "r") as f:
                vdb_data = json.load(f)
                if "embedding_dim" in vdb_data:
                    stored_dim = vdb_data["embedding_dim"]
                    if stored_dim != expected_dim:
                        raise ValueError(
                            f"Embedding dimension mismatch!\n"
                            f"  Storage was created with: {stored_dim} dimensions\n"
                            f"  Current embedding model has: {expected_dim} dimensions\n"
                            f"  Embedding model: {config.query_embedding_model}\n\n"
                            f"Solution: Delete the storage directory and recreate it:\n"
                            f"  rm -rf {working_dir}\n"
                            f"  python store.py attention.pdf"
                        )
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        # If we can't check, proceed and let RAG-Anything handle it
        pass

    # Create RAG-Anything config
    parser_output_base = f"./output_{sanitize_model_name(config.query_chat_model)}_{sanitize_model_name(config.query_embedding_model)}_{sanitize_model_name(config.parser)}"
    rag_config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=config.parser,
        parse_method="auto",
        parser_output_dir=parser_output_base,
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Initialize RAG-Anything with better error handling
    try:
        rag_anything = RAGAnything(
            config=rag_config,
            llm_model_func=create_llm_func(chat_model),
            embedding_func=embedding_func,
        )
    except AssertionError as e:
        if "Embedding dim mismatch" in str(e):
            raise ValueError(
                f"Embedding dimension mismatch detected!\n"
                f"  Expected: {expected_dim} dimensions\n"
                f"  Storage has: Different dimension (see error above)\n"
                f"  Embedding model: {config.query_embedding_model}\n\n"
                f"Solution: Delete the storage directory and recreate it:\n"
                f"  rm -rf {working_dir}\n"
                f"  python store.py attention.pdf"
            ) from e
        raise

    # Create RAG instance
    rag_instance = RAGAnythingRAG(
        llm=chat_model,
        documents=None,
        k=3,
        rag_anything_instance=rag_anything,
        processed_docs_cache=None,
    )

    # Query and get answer
    answer, relevant_docs = rag_instance.answer(question=query_text)

    # Print retrieved documents if requested
    if show_retrieved_docs:
        print("\n" + "=" * 80)
        print("RETRIEVED DOCUMENTS")
        print("=" * 80)
        print(f"Retrieved {len(relevant_docs)} document(s):\n")

        for i, doc in enumerate(relevant_docs, 1):
            print(f"[Document {i}]")
            print("-" * 80)

            # Print content (truncate if too long)
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            max_length = 500  # Show first 500 characters
            if len(content) > max_length:
                print(content[:max_length] + "...")
                print(f"\n[Content truncated: {len(content)} total characters]")
            else:
                print(content)

            # Print metadata if available
            if hasattr(doc, "metadata") and doc.metadata:
                print("\nMetadata:")
                for key, value in doc.metadata.items():
                    print(f"  {key}: {value}")

            print("-" * 80)
            print()

    return answer, relevant_docs


def main():
    """Main entry point."""

    query_text = "List all authors of the paper 'Attention is All You Need' with their institutional email addresses as published in the paper. This is publicly available academic information from the published paper."
    # Use values from config.py

    try:
        answer, relevant_docs = query(query_text, show_retrieved_docs=True)
        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        print(f"\n✓ Used {len(relevant_docs)} relevant document(s) for context")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
