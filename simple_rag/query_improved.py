#!/usr/bin/env python3
"""
Improved RAG Query with Better Retrieval Parameters

This version uses the same structure as query.py but with optimized parameters:
- mode="hybrid" instead of "mix" for better retrieval
- top_k=50 instead of default 20 (more chunks)
- cosine_threshold=0.1 instead of 0.2 (lower threshold)
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
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

try:
    from simple_rag import config
except ImportError:
    import config

nest_asyncio.apply()


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in file paths."""
    return model_name.replace("/", "_").replace(":", "_").replace(" ", "_")


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


def query_improved(
    query_text: str,
    working_dir: str = None,
    show_retrieved_docs: bool = True,
) -> tuple[str, list]:
    """
    Query with improved retrieval parameters.
    
    Key improvements over query.py:
    1. Uses mode="hybrid" instead of "mix"
    2. Increases top_k from 20 to 50
    3. Lowers cosine_threshold from 0.2 to 0.1
    4. These changes help retrieve more chunks, including metadata like author emails
    """
    
    # Use config path if not provided
    if working_dir is None:
        working_dir = config.query_rag_storage_path
    
    # Check if storage exists
    storage_path = Path(working_dir)
    if not storage_path.exists() or not list(storage_path.glob("*.json")):
        raise FileNotFoundError(
            f"Storage not found at {working_dir}. "
            "Please run store.py first to process documents."
        )
    
    # Initialize models
    print(f"Using chat provider: {config.query_chat_provider}")
    print(f"Using embedding provider: {config.query_embedding_provider}")
    
    chat_model = ChatOpenAI(
        model=config.query_chat_model,
        temperature=0.1,
        max_tokens=4000,
        max_retries=3,
    )
    embedding_model = OpenAIEmbeddings(model=config.query_embedding_model)
    
    # Create RAG config
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
    
    # Initialize RAGAnything
    rag_anything = RAGAnything(
        config=rag_config,
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
    
    print(f"\n{'='*80}")
    print("IMPROVED QUERY PARAMETERS")
    print(f"{'='*80}")
    print(f"Mode: naive (pure vector search, best for metadata)")
    print(f"Top K: 50 (more chunks, was 20)")
    print(f"Cosine Threshold: 0.1 (lower threshold, was 0.2)")
    print(f"{'='*80}\n")
    
    # Query and get answer with improved parameters
    # Use naive mode (pure vector search) which is better for metadata queries
    answer, relevant_docs = rag_instance.answer(
        question=query_text,
        mode="naive",  # Pure vector search, best for author emails
        top_k=50,  # Retrieve more chunks
        cosine_threshold=0.1,  # Lower threshold for more results
    )
    
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
            max_length = 500
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
    
    query_text = (
        "List all authors of the paper 'Attention is All You Need' with their "
        "institutional email addresses as published in the paper. "
        "This is publicly available academic information from the published paper."
    )
    
    try:
        answer, relevant_docs = query_improved(query_text, show_retrieved_docs=True)
        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        print(f"\n✓ Used {len(relevant_docs)} relevant document(s) for context")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

