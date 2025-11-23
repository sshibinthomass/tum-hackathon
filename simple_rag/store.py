#!/usr/bin/env python3
"""
Simple RAG-Anything Document Storage

Processes and stores PDF documents in RAG-Anything storage.
Run this once per document.

Usage:
    python store.py <filename>
    python store.py attention.pdf
"""

import asyncio
import re
import logging
import warnings
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from raganything import RAGAnything, RAGAnythingConfig
import nest_asyncio
from ai_eval.config import global_config as glob
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

load_dotenv()

nest_asyncio.apply()

# Configure logging to reduce noise from format warnings
logging.basicConfig(level=logging.INFO)


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


# Apply filter to root logger and common library loggers
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


def get_model_paths(
    chat_model_name: str = None,
    embedding_model_name: str = None,
    parser_name: str = None,
) -> tuple[str, str]:
    """Generate storage and output paths based on model names, parser, and LLM settings."""
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
    working_dir = f"./rag_storage_{model_dir}"
    output_dir = f"./output_{model_dir}/raganything_processed"

    print(
        f"LLM settings: temp={temperature}, max_tokens={max_tokens}, retries={max_retries}"
    )
    print(f"Storage directory: {working_dir}")

    return working_dir, output_dir


def get_models(chat_provider: str = None, embedding_provider: str = None):
    """Initialize chat and embedding models from config."""
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
    elif chat_provider == "groq":
        chat_model = ChatGroq(
            model=config.chat_model,
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
            model=config.chat_model,
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
            model=config.chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:  # ollama
        chat_model = ChatOllama(
            model=config.chat_model,
            temperature=temperature,
            num_predict=max_tokens,  # Ollama uses num_predict instead of max_tokens
        )

    # Initialize embedding model
    if embedding_provider == "openai":
        embedding_model = OpenAIEmbeddings(model=config.embedding_model)
    else:  # ollama
        embedding_model = OllamaEmbeddings(model=config.embedding_model)

    return chat_model, embedding_model


def fix_llm_output_format(content: str) -> str:
    """
    Post-process LLM output to fix common format errors per RAG-Anything requirements.

    Based on RAG-Anything GitHub repository format specifications:
    - Entities: entity<|#|>name<|#|>type<|#|>description (4 fields, single line)
    - Relations: relation<|#|>source<|#|>target<|#|>keywords<|#|>description (5 fields, single line)
    - Each record MUST be on a SINGLE line
    - <|#|> is ONLY used to separate fields WITHIN a line, NOT to separate records
    - End with <|COMPLETE|>

    Fixes:
    1. Strips introductory/placeholder text
    2. Fixes malformed delimiters (e.g., <|location|> -> <|#|>location<|#|>)
    3. Converts newline-separated records to proper single-line format
    4. Fixes wrong field counts (6/5 for relations, 3/4 for entities)
    5. Ensures proper <|#|> delimiter usage
    6. Adds missing descriptions
    7. Removes empty/invalid records
    """
    if not content or not content.strip():
        return "<|COMPLETE|>"

    # Remove common placeholder/introductory text
    content = content.strip()
    skip_patterns = [
        "I'm ready",
        "Please provide",
        "I will follow",
        "I will adhere",
        "I can extract",
    ]
    for pattern in skip_patterns:
        if content.startswith(pattern):
            return "<|COMPLETE|>"

    # If content is just <|COMPLETE|>, return it (don't try to fix it)
    if content.strip() == "<|COMPLETE|>":
        return "<|COMPLETE|>"

    # Split by lines and process
    lines = content.split("\n")
    fixed_lines = []
    current_record = []  # For handling multi-line records
    has_valid_content = False  # Track if we found any valid entities/relations

    for line in lines:
        line = line.strip()
        if not line:
            # Empty line - if we have a current record, try to finalize it
            if current_record:
                fixed_line = _try_fix_multi_line_record(current_record)
                if fixed_line:
                    fixed_lines.append(fixed_line)
                    has_valid_content = True
                current_record = []
            continue

        # Handle completion signal
        if line == "<|COMPLETE|>":
            # Finalize any pending record
            if current_record:
                fixed_line = _try_fix_multi_line_record(current_record)
                if fixed_line:
                    fixed_lines.append(fixed_line)
                    has_valid_content = True
                current_record = []
            continue

        # Fix malformed delimiters (e.g., <|location|> -> <|#|>location<|#|>)
        line = _fix_malformed_delimiters(line)

        # Check if line has proper format with <|#|>
        if "<|#|>" in line:
            # Finalize any pending multi-line record
            if current_record:
                fixed_line = _try_fix_multi_line_record(current_record)
                if fixed_line:
                    fixed_lines.append(fixed_line)
                current_record = []

            # Process this line
            fixed_line = _fix_formatted_line(line)
            if fixed_line:
                fixed_lines.append(fixed_line)
                has_valid_content = True
        else:
            # Line without <|#|> - might be part of a multi-line record
            # or a malformed record
            if line.startswith("entity") or line.startswith("relation"):
                # Start of a new record without delimiters
                if current_record:
                    # Finalize previous record
                    fixed_line = _try_fix_multi_line_record(current_record)
                    if fixed_line:
                        fixed_lines.append(fixed_line)
                        has_valid_content = True
                current_record = [line]
            elif current_record:
                # Continuation of multi-line record
                current_record.append(line)
            # Otherwise, skip non-entity/relation lines

    # Finalize any remaining record
    if current_record:
        fixed_line = _try_fix_multi_line_record(current_record)
        if fixed_line:
            fixed_lines.append(fixed_line)
            has_valid_content = True

    # If we found valid content, return the fixed version
    if has_valid_content:
        result = "\n".join(fixed_lines)
        if result and not result.endswith("<|COMPLETE|>"):
            result += "\n<|COMPLETE|>"
        return result
    else:
        # No valid content found - preserve original content instead of stripping it
        # This prevents the format correction from removing valid LLM output
        # that we just couldn't parse properly
        original_stripped = content.strip()
        if original_stripped and original_stripped != "<|COMPLETE|>":
            # Return original content with <|COMPLETE|> if not present
            if not original_stripped.endswith("<|COMPLETE|>"):
                return original_stripped + "\n<|COMPLETE|>"
            return original_stripped
        else:
            # Truly empty or just completion signal
            return "<|COMPLETE|>"


def _fix_malformed_delimiters(line: str) -> str:
    """Fix malformed delimiters like <|location|> to <|#|>location<|#|>."""
    # Pattern: <|word|> should be <|#|>word<|#|>
    pattern = r"<\|([^|]+)\|>"
    replacement = r"<|#|>\1<|#|>"

    # But don't replace <|#|> itself
    if "<|#|>" in line:
        return line

    return re.sub(pattern, replacement, line)


def _fix_formatted_line(line: str) -> str:
    """Fix a line that already has <|#|> delimiters."""
    parts = line.split("<|#|>")

    # Fix entity lines (should have 4 fields: entity, name, type, description)
    if line.startswith("entity<|#|>"):
        if len(parts) == 4:
            # Check for empty description - add default if missing
            if parts[3].strip():
                return line
            else:
                # Add default description instead of skipping
                entity_name = parts[1].strip() if len(parts) > 1 else "Unknown"
                entity_type = parts[2].strip() if len(parts) > 2 else "Other"
                return f"entity<|#|>{entity_name}<|#|>{entity_type}<|#|>A {entity_type.lower()} entity named {entity_name}."
        elif len(parts) == 3:
            # Missing description - add a placeholder
            entity_name = parts[1].strip() if len(parts) > 1 else "Unknown"
            entity_type = parts[2].strip() if len(parts) > 2 else "Other"
            return f"entity<|#|>{entity_name}<|#|>{entity_type}<|#|>A {entity_type.lower()} entity."
        elif len(parts) > 4:
            # Too many fields - merge extra fields into description
            entity_name = parts[1].strip() if len(parts) > 1 else "Unknown"
            entity_type = parts[2].strip() if len(parts) > 2 else "Other"
            description = " ".join(parts[3:]).strip()
            if description:
                return f"entity<|#|>{entity_name}<|#|>{entity_type}<|#|>{description}"
            else:
                return f"entity<|#|>{entity_name}<|#|>{entity_type}<|#|>A {entity_type.lower()} entity."
        else:
            # Too few fields - skip
            return None

    # Fix relation lines (should have 5 fields: relation, source, target, keywords, description)
    elif line.startswith("relation<|#|>"):
        if len(parts) == 5:
            # Check for empty description
            if parts[4].strip():
                return line
            else:
                # Add a default description if missing
                source = parts[1].strip() if len(parts) > 1 else "Unknown"
                target = parts[2].strip() if len(parts) > 2 else "Unknown"
                keywords = parts[3].strip() if len(parts) > 3 else "relationship"
                return f"relation<|#|>{source}<|#|>{target}<|#|>{keywords}<|#|>{source} is related to {target}."
        elif len(parts) == 6:
            # Common error: 6 fields instead of 5
            # Usually the keywords field is split - merge fields 3 and 4
            source = parts[1].strip() if len(parts) > 1 else "Unknown"
            target = parts[2].strip() if len(parts) > 2 else "Unknown"
            keywords = (
                f"{parts[3].strip()}, {parts[4].strip()}"
                if len(parts) > 4 and parts[3].strip() and parts[4].strip()
                else (parts[3].strip() if len(parts) > 3 else "relationship")
            )
            description = (
                parts[5].strip()
                if len(parts) > 5
                else f"{source} is related to {target}."
            )
            if not description:
                description = f"{source} is related to {target}."
            return (
                f"relation<|#|>{source}<|#|>{target}<|#|>{keywords}<|#|>{description}"
            )
        elif len(parts) > 6:
            # Too many fields - merge extra into description
            source = parts[1].strip() if len(parts) > 1 else "Unknown"
            target = parts[2].strip() if len(parts) > 2 else "Unknown"
            keywords = parts[3].strip() if len(parts) > 3 else "relationship"
            description = " ".join(parts[4:]).strip()
            if not description:
                description = f"{source} is related to {target}."
            return (
                f"relation<|#|>{source}<|#|>{target}<|#|>{keywords}<|#|>{description}"
            )
        elif len(parts) == 4:
            # Missing description - add default
            source = parts[1].strip() if len(parts) > 1 else "Unknown"
            target = parts[2].strip() if len(parts) > 2 else "Unknown"
            keywords = parts[3].strip() if len(parts) > 3 else "relationship"
            return f"relation<|#|>{source}<|#|>{target}<|#|>{keywords}<|#|>{source} is related to {target}."
        else:
            # Too few fields - skip
            return None

    # Unknown format - skip
    return None


def _try_fix_multi_line_record(lines: list[str]) -> str:
    """Try to fix a record that was split across multiple lines."""
    # Join lines and try to extract fields
    combined = " ".join(lines).strip()

    # Try to detect entity or relation
    if combined.startswith("entity"):
        # Try to extract: entity name type description
        # This is a fallback - ideally should have <|#|> delimiters
        parts = combined.split(None, 3)  # Split into max 4 parts
        if len(parts) >= 3:
            entity_name = parts[1] if len(parts) > 1 else "Unknown"
            entity_type = parts[2] if len(parts) > 2 else "Other"
            description = (
                parts[3] if len(parts) > 3 else f"A {entity_type.lower()} entity."
            )
            return f"entity<|#|>{entity_name}<|#|>{entity_type}<|#|>{description}"
    elif combined.startswith("relation"):
        # Try to extract: relation source target keywords description
        parts = combined.split(None, 4)  # Split into max 5 parts
        if len(parts) >= 3:
            source = parts[1] if len(parts) > 1 else "Unknown"
            target = parts[2] if len(parts) > 2 else "Unknown"
            keywords = parts[3] if len(parts) > 3 else "relationship"
            description = (
                parts[4] if len(parts) > 4 else f"{source} is related to {target}."
            )
            return (
                f"relation<|#|>{source}<|#|>{target}<|#|>{keywords}<|#|>{description}"
            )

    return None


def create_llm_func(chat_model):
    """Create async LLM function wrapper with enhanced error handling and caching."""

    # Cache for repeated prompts (helps with multimodal processing)
    prompt_cache = {}

    async def llm_func(prompt, **kwargs):
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

                # Apply format correction to fix all common errors
                # This handles most of the warnings we see in the logs
                content = fix_llm_output_format(content)

                # Basic validation: ensure response is not empty or just a placeholder
                if (
                    content
                    and content.strip()
                    and not content.strip().startswith("I'm ready")
                ):
                    # Cache successful response
                    prompt_cache[cache_key] = content
                    return content
                elif attempt < max_attempts - 1:
                    # If response looks incomplete, retry with shorter backoff
                    await asyncio.sleep(0.3 * (attempt + 1))
                    continue
                else:
                    # Last attempt, return what we got
                    return content

            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.3 * (attempt + 1))
                    continue
                else:
                    # Last attempt failed, raise the error
                    raise

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        return ""

    llm_func.func = llm_func
    return llm_func


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


def store_document(
    filename: str,
    chat_provider: str = None,
    embedding_provider: str = None,
    output_dir: str = None,
):
    """Process and store a document."""
    # Setup paths
    file_path = Path(glob.DATA_PKG_DIR) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    # Get model names and generate paths
    chat_provider = chat_provider or config.chat_provider
    embedding_provider = embedding_provider or config.embedding_provider
    working_dir, default_output_dir = get_model_paths(
        config.chat_model, config.embedding_model, config.parser
    )

    # Use provided output_dir or generate from model names
    if output_dir is None:
        output_dir = default_output_dir

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize models from config
    chat_model, embedding_model = get_models(chat_provider, embedding_provider)

    # Create RAG-Anything config
    # Include LLM settings in parser_output_base for consistency
    llm_suffix = get_llm_settings_suffix()
    parser_output_base = f"./output_{sanitize_model_name(config.chat_model)}_{sanitize_model_name(config.embedding_model)}_{sanitize_model_name(config.parser)}_{llm_suffix}"

    rag_config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=config.parser,
        parse_method="auto",
        parser_output_dir=parser_output_base,
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Initialize RAG-Anything
    rag_anything = RAGAnything(
        config=rag_config,
        llm_model_func=create_llm_func(chat_model),
        embedding_func=create_embedding_func(embedding_model),
    )

    # Process document
    print(f"Processing {filename}...")
    asyncio.run(
        rag_anything.process_document_complete(
            file_path=str(file_path),
            output_dir=output_dir,
            parse_method="auto",
        )
    )
    print("✓ Document stored successfully!")


if __name__ == "__main__":
    filename = "Allplan_2020_Manual.pdf"
    # Use values from config.py
    store_document(filename)
