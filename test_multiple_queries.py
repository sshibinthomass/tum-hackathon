#!/usr/bin/env python3
"""Test script for querying RAG-Anything documents with multiple queries."""

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

# Test multiple queries
queries = [
    "What is the Transformer architecture?",
    "How does self-attention work?",
]

for query in queries:
    print("\n" + "="*80)
    answer, docs = query_document(query, rag_instance, show_context=False)
    print("\n")

print("\n✓ All queries completed successfully!")
