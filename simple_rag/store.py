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
import sys
from pathlib import Path

from dotenv import load_dotenv
from raganything import RAGAnything, RAGAnythingConfig
import nest_asyncio
from ai_eval.config import global_config as glob
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

nest_asyncio.apply()


def get_models(model_provider: str = "ollama"):
    """Initialize chat and embedding models."""
    if model_provider == "openai":
        chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    elif model_provider == "groq":
        chat_model = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        chat_model = ChatOllama(model="qwen2.5vl:3b", temperature=0.1)
        embedding_model = OllamaEmbeddings(model="bge-m3")
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


def store_document(
    filename: str,
    model_provider: str = "ollama",
    output_dir: str = "./output_simple/raganything_processed",
):
    """Process and store a document."""
    # Setup paths
    file_path = Path(glob.DATA_PKG_DIR) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize models
    chat_model, embedding_model = get_models(model_provider)

    # Create config
    config = RAGAnythingConfig(
        working_dir="./rag_storage_simple",
        parser="mineru",
        parse_method="auto",
        parser_output_dir="./output_simple",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Initialize RAG-Anything
    rag_anything = RAGAnything(
        config=config,
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
    filename = "attention.pdf"
    model_provider = "ollama"
    store_document(filename, model_provider=model_provider)
