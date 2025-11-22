#!/usr/bin/env python3
"""Test script for querying RAG-Anything documents."""

import sys
sys.path.insert(0, '/Users/qtf4195/tum-hackathon')

from rag_anything_pipeline.query_documents import (
    initialize_models,
    create_rag_anything_config,
    create_rag_instance,
    query_document,
    verify_storage,
)

# Verify storage
if not verify_storage():
    print("Please run store_documents.py first!")
    sys.exit(1)

# Initialize
print("Initializing models...")
chat_model, embedding_model = initialize_models()

config = create_rag_anything_config()

print("Initializing RAG-Anything...")
rag_instance = create_rag_instance(chat_model, embedding_model, config)
print("✓ Ready to query!\n")

# Query
query = "What is the attention mechanism?"
answer, docs = query_document(query, rag_instance, show_context=True)

print("\n✓ Query completed successfully!")
