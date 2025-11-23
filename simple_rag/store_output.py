#!/usr/bin/env python3
"""
Document Parsing and Output Generation

Parses PDF documents and generates output files for RAG-Anything processing.
This is the first step - run this to parse documents and generate output files.

Usage:
    python store_output.py <filename>
    python store_output.py attention.pdf
    python store_output.py --batch doc1.pdf doc2.pdf doc3.pdf
"""

import asyncio
import re
import logging
import logging.config
import warnings
import os
import argparse
import time
from pathlib import Path
import sys

# Ensure venv bin is in PATH for subprocess calls (like mineru)
venv_bin = Path(sys.executable).parent
if str(venv_bin) not in os.environ["PATH"]:
    os.environ["PATH"] = f"{venv_bin}:{os.environ['PATH']}"

from functools import lru_cache
from typing import List, Optional

from dotenv import load_dotenv
from raganything import RAGAnything, RAGAnythingConfig
import nest_asyncio
from ai_eval.config import global_config as glob
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq

# Import Gemini and Anthropic (with fallback if not installed)
try:
    # from langchain_google_vertexai import ChatVertexAI
    pass
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

load_dotenv()

nest_asyncio.apply()


def configure_logging(log_dir: str = None):
    """
    Configure logging with both console and file output.
    Based on RAG-Anything example best practices.
    """
    # Get log directory path from environment variable or use provided/default
    log_dir = log_dir or os.getenv("LOG_DIR", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.abspath(os.path.join(log_dir, "rag_output.log"))

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "": {  # Root logger
                    "handlers": ["console", "file"],
                    "level": "INFO",
                },
            },
        }
    )

    print(f"📝 Log file: {log_file_path}\n")


# Create a custom filter to suppress specific warning messages
class WarningFilter(logging.Filter):
    """Filter out repetitive LLM format warnings that are handled automatically."""

    def filter(self, record):
        # Suppress LLM format error warnings (they're fixed automatically)
        if "LLM output format error" in record.getMessage():
            return False
        if "Entity extraction error: empty description" in record.getMessage():
            return False
        if "Using regex fallback for JSON parsing" in record.getMessage():
            return False
        return True


# Apply filter to common library loggers
for logger_name in ["", "raganything", "lightrag", "__main__"]:
    logger = logging.getLogger(logger_name)
    logger.addFilter(WarningFilter())

# Also suppress Python warnings
warnings.filterwarnings("ignore", message=".*LLM output format error.*")
warnings.filterwarnings("ignore", message=".*Entity extraction error.*")
warnings.filterwarnings("ignore", message=".*Using regex fallback.*")


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in file paths."""
    # Replace special characters with underscores
    return model_name.replace("/", "_").replace(":", "_").replace(" ", "_")


def get_llm_settings_suffix() -> str:
    """Generate a suffix string from LLM settings for folder names."""
    temperature = getattr(config, "llm_temperature", 0.0)
    max_tokens = getattr(config, "llm_max_tokens", 4000)
    max_retries = getattr(config, "llm_max_retries", 3)

    # Format max_tokens: 4000 -> "4k", 2000 -> "2k", etc.
    if max_tokens >= 1000:
        max_tokens_str = f"{max_tokens // 1000}k"
    else:
        max_tokens_str = str(max_tokens)

    return f"temp{temperature}_max{max_tokens_str}_retry{max_retries}"


def get_output_path(
    chat_model_name: str = None,
    embedding_model_name: str = None,
    parser_name: str = None,
) -> str:
    """Generate output path based on model names, parser, and LLM settings."""
    chat_model_name = chat_model_name or config.chat_model
    embedding_model_name = embedding_model_name or config.embedding_model
    parser_name = config.parser
    print(f"Using parser: {parser_name}")
    print(f"Using chat model: {chat_model_name}")
    print(f"Using embedding model: {embedding_model_name}")

    # Sanitize model names and parser for file paths
    chat_safe = sanitize_model_name(chat_model_name)
    embedding_safe = sanitize_model_name(embedding_model_name)
    parser_safe = sanitize_model_name(parser_name)

    # Get LLM settings suffix
    llm_suffix = get_llm_settings_suffix()

    # Get LLM settings for display
    temperature = getattr(config, "llm_temperature", 0.0)
    max_tokens = getattr(config, "llm_max_tokens", 4000)
    max_retries = getattr(config, "llm_max_retries", 3)

    model_dir = f"{chat_safe}_{embedding_safe}_{parser_safe}_{llm_suffix}"
    output_dir = f"./output_{model_dir}/raganything_processed"

    print(
        f"LLM settings: temp={temperature}, max_tokens={max_tokens}, retries={max_retries}"
    )
    print(f"Output directory: {output_dir}")

    return output_dir


def get_models(chat_provider: str = None, embedding_provider: str = None):
    """
    Initialize chat and embedding models from config.
    Enhanced with vision model support for multimodal processing.
    """
    # Use config values if not provided
    chat_provider = config.chat_provider
    embedding_provider = config.embedding_provider
    print(f"Using chat provider: {chat_provider}")
    print(f"Using embedding provider: {embedding_provider}")

    # Get LLM settings from config (with defaults)
    temperature = getattr(config, "llm_temperature", 0.0)
    max_tokens = getattr(config, "llm_max_tokens", 4000)
    max_retries = getattr(config, "llm_max_retries", 3)

    print(
        f"LLM settings: temperature={temperature}, max_tokens={max_tokens}, max_retries={max_retries}"
    )

    # Initialize chat model with optimized settings for structured output
    if chat_provider == "openai":
        chat_model = ChatOpenAI(
            model=config.chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        # Vision model for multimodal content (use gpt-4o for better vision capabilities)
        vision_model = ChatOpenAI(
            model="gpt-4o",  # Correct vision model
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    elif chat_provider == "groq":
        chat_model = ChatGroq(
            model=config.chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        vision_model = chat_model  # Groq doesn't have separate vision model
    elif chat_provider == "gemini":
        if ChatVertexAI is None:
            raise ImportError(
                "Gemini models require langchain-google-vertexai or langchain-google-genai. "
                "Install with: pip install langchain-google-vertexai"
            )
        chat_model = ChatVertexAI(
            model=config.chat_model,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        vision_model = chat_model  # Gemini models support vision natively
    elif chat_provider == "anthropic":
        if ChatAnthropic is None:
            raise ImportError(
                "Anthropic models require langchain-anthropic. "
                "Install with: pip install langchain-anthropic"
            )
        chat_model = ChatAnthropic(
            model=config.chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        vision_model = chat_model  # Claude supports vision natively
    else:  # ollama
        chat_model = ChatOllama(
            model=config.chat_model,
            temperature=temperature,
            num_predict=max_tokens,  # Ollama uses num_predict instead of max_tokens
        )
        vision_model = chat_model

    # Initialize embedding model
    if embedding_provider == "openai":
        embedding_model = OpenAIEmbeddings(model=config.embedding_model)
    else:  # ollama
        embedding_model = OllamaEmbeddings(model=config.embedding_model)

    return chat_model, vision_model, embedding_model


def create_llm_func(chat_model):
    """Create async LLM function wrapper with enhanced error handling and caching."""

    # Cache for repeated prompts (helps with multimodal processing)
    prompt_cache = {}

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        """Async LLM function with retry logic and format correction."""
        # Check cache first (for identical prompts)
        cache_key = hash(prompt)
        if cache_key in prompt_cache:
            return prompt_cache[cache_key]

        max_attempts = 2  # Reduced from 3 since we have format fixing
        last_error = None

        for attempt in range(max_attempts):
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, chat_model.invoke, prompt)
                content = (
                    response.content if hasattr(response, "content") else str(response)
                )

                # Basic validation
                if (
                    content
                    and content.strip()
                    and not content.strip().startswith("I'm ready")
                ):
                    # Cache successful response
                    prompt_cache[cache_key] = content
                    return content
                elif attempt < max_attempts - 1:
                    await asyncio.sleep(0.3 * (attempt + 1))
                    continue
                else:
                    return content

            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.3 * (attempt + 1))
                    continue
                else:
                    raise

        if last_error:
            raise last_error
        return ""

    llm_func.func = llm_func
    return llm_func


def create_vision_func(vision_model):
    """
    Create async vision model function for multimodal content processing.
    Based on RAG-Anything example best practices.
    """

    async def vision_func(
        prompt,
        system_prompt=None,
        history_messages=[],
        image_data=None,
        messages=None,
        **kwargs,
    ):
        """Vision model function supporting both single image and multimodal formats."""
        loop = asyncio.get_event_loop()

        # If messages format is provided (for multimodal VLM enhanced query), use it directly
        if messages:
            response = await loop.run_in_executor(
                None, lambda: vision_model.invoke(messages)
            )
        # Traditional single image format
        elif image_data:
            # Format message for vision model
            message_content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
            response = await loop.run_in_executor(
                None,
                lambda: vision_model.invoke(
                    [{"role": "user", "content": message_content}]
                ),
            )
        # Pure text format
        else:
            response = await loop.run_in_executor(None, vision_model.invoke, prompt)

        return response.content if hasattr(response, "content") else str(response)

    vision_func.func = vision_func
    return vision_func


def create_embedding_func(embedding_model):
    """Create async embedding function wrapper with caching."""
    # Get embedding dimension once
    sample_embedding = embedding_model.embed_documents(["test"])
    embedding_dim = len(sample_embedding[0])

    # Simple cache for repeated embeddings
    embedding_cache = {}

    async def embedding_func(texts):
        if isinstance(texts, str):
            texts = [texts]

        # Check cache for each text
        uncached_texts = []
        cached_results = {}
        for i, text in enumerate(texts):
            cache_key = hash(text)
            if cache_key in embedding_cache:
                cached_results[i] = embedding_cache[cache_key]
            else:
                uncached_texts.append((i, text))

        # Get embeddings for uncached texts
        if uncached_texts:
            loop = asyncio.get_event_loop()
            uncached_only = [text for _, text in uncached_texts]
            new_embeddings = await loop.run_in_executor(
                None, embedding_model.embed_documents, uncached_only
            )

            # Cache new embeddings
            for (i, text), embedding in zip(uncached_texts, new_embeddings):
                cache_key = hash(text)
                embedding_cache[cache_key] = embedding
                cached_results[i] = embedding

        # Reconstruct results in original order
        return [cached_results[i] for i in range(len(texts))]

    embedding_func.embedding_dim = embedding_dim
    embedding_func.func = embedding_func
    return embedding_func


async def parse_document(
    filename: str,
    chat_provider: str = None,
    embedding_provider: str = None,
    output_dir: str = None,
    show_progress: bool = True,
):
    """
    Parse a document and generate output files.
    This is step 1 - generates parsed output that can be used by store_rag.py

    Args:
        filename: Name of the file to process
        chat_provider: Chat model provider (openai, groq, etc.)
        embedding_provider: Embedding model provider
        output_dir: Output directory for processed files
        show_progress: Whether to show progress information
    """
    start_time = time.time()

    # Setup paths
    print(f"DEBUG: glob.DATA_PKG_DIR = {glob.DATA_PKG_DIR}")
    file_path = Path(glob.DATA_PKG_DIR) / filename
    print(f"DEBUG: file_path = {file_path}")
    if not file_path.exists():
        print(f"DEBUG: File does not exist at {file_path}")
        # Try looking in ../data
        alt_path = Path("../data") / filename
        if alt_path.exists():
            print(f"DEBUG: Found file at {alt_path}")
            file_path = alt_path
        else:
            raise FileNotFoundError(f"Document not found: {file_path}")

    # Get model names and generate paths
    chat_provider = chat_provider or config.chat_provider
    embedding_provider = embedding_provider or config.embedding_provider
    default_output_dir = get_output_path(
        config.chat_model, config.embedding_model, config.parser
    )

    # Use provided output_dir or generate from model names
    if output_dir is None:
        output_dir = default_output_dir

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize models from config
    chat_model, vision_model, embedding_model = get_models(
        chat_provider, embedding_provider
    )

    # Create RAG-Anything config for parsing only
    # Use a temporary working_dir since we're only generating output
    llm_suffix = get_llm_settings_suffix()
    parser_output_base = f"./output_{sanitize_model_name(config.chat_model)}_{sanitize_model_name(config.embedding_model)}_{sanitize_model_name(config.parser)}_{llm_suffix}"
    temp_working_dir = f"./rag_storage_temp_{sanitize_model_name(config.chat_model)}_{sanitize_model_name(config.embedding_model)}_{sanitize_model_name(config.parser)}_{llm_suffix}"

    rag_config = RAGAnythingConfig(
        working_dir=temp_working_dir,  # Temporary working dir for parsing only
        parser=config.parser,
        parse_method="auto",
        parser_output_dir=parser_output_base,
        # Multimodal processing - ENABLED for best quality
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        # BEST PRACTICE: Enhanced context extraction for better understanding
        context_window=2,  # Increased from default 1 - include 2 pages before/after
        context_mode="page",  # Page-based context for document structure
        max_context_tokens=3000,  # Increased from default 2000 for richer context
        include_headers=True,  # Include document headers for structure
        include_captions=True,  # Include image/table captions
        context_filter_content_types=[
            "text",
            "image",
            "table",
        ],  # Include ALL content types
    )

    # Initialize RAGAnything with vision model support
    rag_anything = RAGAnything(
        config=rag_config,
        llm_model_func=create_llm_func(chat_model),
        vision_model_func=create_vision_func(
            vision_model
        ),  # Enhanced with vision support
        embedding_func=create_embedding_func(embedding_model),
    )

    # Process document - this will parse and generate output files
    print(f"Parsing {filename} and generating output files...")
    print(f"Output will be saved to: {output_dir}")
    await rag_anything.process_document_complete(
        file_path=str(file_path),
        output_dir=output_dir,
        parse_method="auto",
    )

    elapsed_time = time.time() - start_time
    print(f"✓ Document parsed successfully! (took {elapsed_time:.2f} seconds)")
    print(f"✓ Output files saved to: {output_dir}")
    print(f"\nNext step: Run store_rag.py to store the parsed output in RAG storage")


async def parse_documents_batch(
    filenames: List[str],
    chat_provider: str = None,
    embedding_provider: str = None,
    output_dir: str = None,
):
    """
    Parse multiple documents in batch.

    Args:
        filenames: List of filenames to process
        chat_provider: Chat model provider
        embedding_provider: Embedding model provider
        output_dir: Output directory for processed files
    """
    print(f"\n{'=' * 60}")
    print(f"BATCH PARSING: {len(filenames)} documents")
    print(f"{'=' * 60}\n")

    start_time = time.time()
    successful = []
    failed = []

    for i, filename in enumerate(filenames, 1):
        print(f"\n[{i}/{len(filenames)}] Parsing: {filename}")
        print("-" * 40)

        try:
            await parse_document(
                filename,
                chat_provider=chat_provider,
                embedding_provider=embedding_provider,
                output_dir=output_dir,
                show_progress=True,
            )
            successful.append(filename)
        except Exception as e:
            print(f"❌ Error parsing {filename}: {str(e)}")
            failed.append((filename, str(e)))

    total_time = time.time() - start_time

    # Print summary
    print(f"\n{'=' * 60}")
    print("BATCH PARSING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total documents: {len(filenames)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success rate: {len(successful) / len(filenames) * 100:.1f}%")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per document: {total_time / len(filenames):.2f} seconds")

    if successful:
        print(f"\n✅ Successfully parsed:")
        for filename in successful:
            print(f"   - {filename}")

    if failed:
        print(f"\n❌ Failed to parse:")
        for filename, error in failed:
            print(f"   - {filename}: {error}")


if __name__ == "__main__":
    # Configure logging first
    configure_logging(log_dir="./logs")

    parser = argparse.ArgumentParser(
        description="Parse PDF documents and generate output files for RAG storage"
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="Allplan_2020_Manual_2_pages.pdf",
        help="Name of the PDF file to parse (default: Allplan_2020_Manual.pdf)",
    )
    parser.add_argument(
        "--batch",
        nargs="+",
        help="Process multiple files in batch",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for parsed files (default: auto-generated from model config)",
    )

    args = parser.parse_args()

    if args.batch:
        asyncio.run(parse_documents_batch(args.batch, output_dir=args.output_dir))
    else:
        asyncio.run(parse_document(args.filename, output_dir=args.output_dir))
