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
from pathlib import Path

from raganything import RAGAnything, RAGAnythingConfig
import nest_asyncio
from ai_eval.resources.rag_template import RAGAnythingRAG
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq

nest_asyncio.apply()


def get_models(model_provider: str = "ollama"):
    """Initialize chat and embedding models."""
    if model_provider == "openai":
        chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        embedding_model = OllamaEmbeddings(model="bge-m3")
    elif model_provider == "groq":
        chat_model = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.1)
        embedding_model = OllamaEmbeddings(model="bge-m3")
    else:
        chat_model = ChatOllama(model="llama3.1:latest", temperature=0.1)
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


def query(
    query_text: str, model_provider: str = "ollama", working_dir: str = "./rag_storage_simple"
) -> str:
    """
    Query stored documents and return answer.

    Args:
        query_text: The question to ask
        model_provider: Model provider (openai, groq, or ollama)
        working_dir: RAG-Anything working directory

    Returns:
        Answer string
    """
    # Check if storage exists
    storage_path = Path(working_dir)
    if not storage_path.exists() or not list(storage_path.glob("*.json")):
        raise FileNotFoundError(
            f"Storage not found at {working_dir}. "
            "Please run store.py first to process documents."
        )

    # Initialize models
    chat_model, embedding_model = get_models(model_provider)

    # Create config
    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser="mineru",
        parse_method="auto",
        parser_output_dir="./output",
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

    # Create RAG instance
    rag_instance = RAGAnythingRAG(
        llm=chat_model,
        documents=None,
        k=3,
        rag_anything_instance=rag_anything,
        processed_docs_cache=None,
    )

    # Query and get answer
    answer, _ = rag_instance.answer(question=query_text)
    return answer


def main():
    """Main entry point."""

    query_text = "Can you give more details about Vaswani also give his email address?"
    model_provider = "openai"

    try:
        answer = query(query_text, model_provider=model_provider)
        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)
        print("=" * 80)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
