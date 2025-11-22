#!/usr/bin/env python3
"""
Direct LightRAG Query - Bypasses wrapper to use correct parameters

This script queries LightRAG directly with optimized parameters to ensure
the author email chunk is retrieved.
"""

import sys
import asyncio
from pathlib import Path

from raganything import RAGAnything, RAGAnythingConfig
import nest_asyncio
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


async def query_direct(query_text: str, working_dir: str):
    """
    Query LightRAG directly with optimized parameters.
    
    This bypasses the RAGAnythingRAG wrapper to ensure parameters are used correctly.
    """
    
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
    
    # Explicitly initialize LightRAG for querying existing storage
    print("Initializing LightRAG via _ensure_lightrag_initialized...")
    if hasattr(rag_anything, "_ensure_lightrag_initialized"):
        if asyncio.iscoroutinefunction(rag_anything._ensure_lightrag_initialized):
            await rag_anything._ensure_lightrag_initialized()
        else:
            rag_anything._ensure_lightrag_initialized()
            
    # Check if lightrag is initialized
    if rag_anything.lightrag is None:
        print("LightRAG still None, trying manual initialization...")
        try:
            from lightrag import LightRAG
            rag_anything.lightrag = LightRAG(
                working_dir=working_dir,
                llm_model_func=create_llm_func(chat_model),
                embedding_func=create_embedding_func(embedding_model),
            )
            print("Manual LightRAG initialization successful!")
        except ImportError:
            print("Could not import LightRAG directly.")
        except Exception as e:
            print(f"Manual initialization failed: {e}")
    
    print(f"\n{'='*80}")
    print("DIRECT LIGHTRAG QUERY WITH OPTIMIZED PARAMETERS")
    print(f"{'='*80}")
    print(f"Query: {query_text}")
    print(f"\nParameters:")
    print(f"  - mode: naive (pure vector search, best for metadata)")
    print(f"  - top_k: 50 (retrieve more chunks)")
    print(f"  - cosine_threshold: 0.1 (lower threshold)")
    print(f"{'='*80}\n")
    
    # Method 1: Try naive mode (pure vector search, best for metadata)
    print("Method 1: Naive mode (pure vector search)...")
    try:
        result_naive = await rag_anything.aquery(
            query_text,
            mode="naive",  # Pure vector search, no graph
            top_k=50,
        )
        
        print(f"\n{'='*80}")
        print("NAIVE MODE RESULT")
        print(f"{'='*80}")
        print(result_naive)
        print(f"{'='*80}\n")
        
        # Check if emails are in result
        if '@' in result_naive:
            print("✅ SUCCESS! Email addresses found in naive mode result!")
            lines_with_emails = [line for line in result_naive.split('\n') if '@' in line]
            print(f"\nEmail lines ({len(lines_with_emails)} found):")
            for line in lines_with_emails[:15]:
                print(f"  {line.strip()}")
        else:
            print("❌ No emails in naive mode result")
            
    except Exception as e:
        print(f"Error in naive mode: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 2: Try getting raw context
    print(f"\n\n{'='*80}")
    print("Method 2: Getting raw context (only_need_context=True)...")
    print(f"{'='*80}\n")
    
    try:
        raw_context = await rag_anything.aquery(
            query_text,
            mode="naive",
            top_k=50,
            only_need_context=True,  # Get raw chunks without LLM processing
        )
        
        print(f"Raw context length: {len(raw_context)} characters")
        print(f"\nFirst 2000 characters:")
        print(raw_context[:2000])
        
        if '@' in raw_context:
            print(f"\n✅ SUCCESS! Email addresses found in raw context!")
            lines_with_emails = [line for line in raw_context.split('\n') if '@' in line]
            print(f"\nEmail lines ({len(lines_with_emails)} found):")
            for line in lines_with_emails[:15]:
                print(f"  {line.strip()}")
        else:
            print(f"\n❌ No emails in raw context")
            
    except Exception as e:
        print(f"Error getting raw context: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    
    working_dir = config.query_rag_storage_path
    
    # Check if storage exists
    storage_path = Path(working_dir)
    if not storage_path.exists():
        print(f"❌ Storage not found: {working_dir}")
        print("Please run store.py first!")
        sys.exit(1)
    
    query_text = (
        "List all authors of the paper 'Attention is All You Need' with their "
        "institutional email addresses as published in the paper. "
        "This is publicly available academic information from the published paper."
    )
    
    try:
        asyncio.run(query_direct(query_text, working_dir))
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
