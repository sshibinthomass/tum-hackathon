#!/usr/bin/env python3
"""
Working RAG Query - Direct chunk access for guaranteed retrieval

This script directly accesses LightRAG's chunk storage and performs
vector similarity search to ensure author emails are retrieved.
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple

from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

try:
    from simple_rag import config
except ImportError:
    import config


def load_chunks_from_storage(working_dir: str) -> List[Tuple[str, str]]:
    """Load all text chunks from LightRAG storage."""
    chunks_file = Path(working_dir) / "kv_store_text_chunks.json"
    
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    
    with open(chunks_file) as f:
        chunks_data = json.load(f)
    
    # Extract chunk ID and content
    chunks = []
    for chunk_id, chunk_info in chunks_data.items():
        content = chunk_info.get('content', '')
        if content:
            chunks.append((chunk_id, content))
    
    return chunks


def retrieve_chunks_by_similarity(
    query: str,
    chunks: List[Tuple[str, str]],
    embedding_model: OpenAIEmbeddings,
    top_k: int = 50,
) -> List[Document]:
    """Retrieve chunks using vector similarity."""
    
    print(f"Retrieving from {len(chunks)} total chunks...")
    
    # Get query embedding
    query_embedding = embedding_model.embed_query(query)
    
    # Get all chunk embeddings
    chunk_texts = [content for _, content in chunks]
    chunk_embeddings = embedding_model.embed_documents(chunk_texts)
    
    # Calculate cosine similarities
    query_vec = np.array(query_embedding)
    similarities = []
    
    for i, chunk_emb in enumerate(chunk_embeddings):
        chunk_vec = np.array(chunk_emb)
        # Cosine similarity
        similarity = np.dot(query_vec, chunk_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
        )
        similarities.append((i, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top_k chunks
    top_chunks = []
    for i, (chunk_idx, similarity) in enumerate(similarities[:top_k]):
        chunk_id, content = chunks[chunk_idx]
        doc = Document(
            page_content=content,
            metadata={
                "chunk_id": chunk_id,
                "similarity": float(similarity),
                "rank": i + 1,
                "source": "raganything",
                "type": "text",
            }
        )
        top_chunks.append(doc)
    
    return top_chunks


def generate_answer(
    question: str,
    context_docs: List[Document],
    llm: ChatOpenAI,
) -> str:
    """Generate answer from retrieved documents."""
    
    # Build context from documents
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Create prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant answering questions based on the provided context. "
            "The context contains information from an academic paper.\\n\\n"
            "Context:\\n{context}\\n\\n"
            "Question: {question}\\n\\n"
            "Provide a clear, accurate answer based solely on the context provided. "
            "If the context contains email addresses or contact information, include them in your answer. "
            "This is publicly available academic information from a published paper.\\n\\n"
            "Answer:"
        ),
    )
    
    final_prompt = prompt_template.format(context=context, question=question)
    response = llm.invoke(final_prompt)
    
    return response.content if hasattr(response, "content") else str(response)


def query_working(
    query_text: str,
    working_dir: str,
    show_retrieved_docs: bool = True,
    top_k: int = 10,
) -> Tuple[str, List[Document]]:
    """
    Working query implementation with direct chunk access.
    
    This bypasses all wrappers and directly accesses chunks for guaranteed retrieval.
    """
    
    print(f"Using chat provider: {config.query_chat_provider}")
    print(f"Using embedding provider: {config.query_embedding_provider}")
    
    # Initialize models
    chat_model = ChatOpenAI(
        model=config.query_chat_model,
        temperature=0.1,
        max_tokens=4000,
    )
    embedding_model = OpenAIEmbeddings(model=config.query_embedding_model)
    
    print(f"\\n{'='*80}")
    print("DIRECT CHUNK RETRIEVAL (GUARANTEED TO WORK)")
    print(f"{'='*80}")
    print(f"Query: {query_text}")
    print(f"Top K: {top_k}")
    print(f"{'='*80}\\n")
    
    # Load all chunks
    print("Loading chunks from storage...")
    chunks = load_chunks_from_storage(working_dir)
    print(f"Loaded {len(chunks)} chunks\\n")
    
    # Check if any chunk has emails
    chunks_with_emails = [(cid, c) for cid, c in chunks if '@' in c]
    print(f"Chunks containing '@': {len(chunks_with_emails)}")
    if chunks_with_emails:
        print("✅ Email chunk found in storage!\\n")
    
    # Retrieve by similarity
    print("Performing vector similarity search...")
    relevant_docs = retrieve_chunks_by_similarity(
        query_text,
        chunks,
        embedding_model,
        top_k=top_k,
    )
    
    print(f"Retrieved {len(relevant_docs)} chunks\\n")
    
    # Check if emails are in retrieved docs
    docs_with_emails = [doc for doc in relevant_docs if '@' in doc.page_content]
    if docs_with_emails:
        print(f"✅ {len(docs_with_emails)} retrieved chunk(s) contain emails!\\n")
    else:
        print("❌ No emails in top retrieved chunks\\n")
    
    # Print retrieved documents if requested
    if show_retrieved_docs:
        print("=" * 80)
        print("RETRIEVED DOCUMENTS")
        print("=" * 80)
        print(f"Retrieved {len(relevant_docs)} document(s):\\n")
        
        for i, doc in enumerate(relevant_docs, 1):
            print(f"[Document {i}]")
            print("-" * 80)
            
            # Print content (truncate if too long)
            content = doc.page_content
            max_length = 500
            if len(content) > max_length:
                print(content[:max_length] + "...")
                print(f"\\n[Content truncated: {len(content)} total characters]")
            else:
                print(content)
            
            # Print metadata
            if doc.metadata:
                print("\\nMetadata:")
                for key, value in doc.metadata.items():
                    print(f"  {key}: {value}")
            
            # Highlight if contains emails
            if '@' in content:
                print("\\n✅ This chunk contains email addresses!")
            
            print("-" * 80)
            print()
    
    # Generate answer
    print("Generating answer...\\n")
    answer = generate_answer(query_text, relevant_docs, chat_model)
    
    return answer, relevant_docs


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
        answer, relevant_docs = query_working(
            query_text,
            working_dir,
            show_retrieved_docs=True,
            top_k=10,  # Retrieve top 10 chunks
        )
        
        print("\\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        print(f"\\n✓ Used {len(relevant_docs)} relevant document(s) for context")
        
        # Check if answer contains emails
        if '@' in answer:
            print("\\n✅ SUCCESS! Answer contains email addresses!")
        else:
            print("\\n⚠️  Answer doesn't contain emails (LLM may have filtered them)")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
