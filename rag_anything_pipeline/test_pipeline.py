#!/usr/bin/env python3
"""
RAG-Anything Pipeline - Complete End-to-End Test

This script tests the entire RAG pipeline in a single run:
1. Stores a PDF document
2. Queries the stored document with multiple questions
3. Displays results

Usage:
    python test_pipeline.py
    python test_pipeline.py --filename "your_document.pdf"
    python test_pipeline.py --force  # Force reprocess even if cached
"""

import argparse
import sys
from pathlib import Path

# Import from store_documents
from store_documents import (
    store_document,
    initialize_models as init_models_store,
)

# Import from query_documents
from query_documents import (
    initialize_models as init_models_query,
    create_rag_anything_config,
    create_rag_instance,
    query_document,
)


def run_complete_pipeline(
    filename: str = "attention.pdf",
    force_reprocess: bool = False,
    queries: list = None,
):
    """
    Run the complete RAG pipeline: store + query.
    
    Args:
        filename: PDF filename to process
        force_reprocess: Force reprocessing even if cached
        queries: List of queries to test (default: predefined test queries)
    """
    print("=" * 80)
    print("RAG-ANYTHING PIPELINE - COMPLETE TEST")
    print("=" * 80)
    print()
    
    # Default test queries if none provided
    if queries is None:
        queries = [
            "What is the attention mechanism?",
            "What is the Transformer architecture?",
            "How does self-attention work?",
        ]
    
    # =========================================================================
    # STEP 1: STORE DOCUMENT
    # =========================================================================
    print("STEP 1: STORING DOCUMENT")
    print("-" * 80)
    
    try:
        was_processed = store_document(
            filename=filename,
            force_reprocess=force_reprocess,
        )
        
        if was_processed:
            print("\n✓ Document stored successfully!")
        else:
            print("\n✓ Document already stored (using cached version)")
        
    except Exception as e:
        print(f"\n❌ Error storing document: {e}")
        return False
    
    print()
    
    # =========================================================================
    # STEP 2: INITIALIZE QUERY SYSTEM
    # =========================================================================
    print("STEP 2: INITIALIZING QUERY SYSTEM")
    print("-" * 80)
    
    try:
        # Initialize models
        chat_model, embedding_model = init_models_query()
        
        # Create config
        config = create_rag_anything_config()
        
        # Create RAG instance
        rag_instance = create_rag_instance(chat_model, embedding_model, config)
        
        print("✓ Query system initialized successfully!")
        
    except Exception as e:
        print(f"\n❌ Error initializing query system: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # =========================================================================
    # STEP 3: RUN QUERIES
    # =========================================================================
    print("STEP 3: RUNNING QUERIES")
    print("-" * 80)
    print()
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 80}")
        print(f"QUERY {i}/{len(queries)}")
        print(f"{'=' * 80}")
        
        try:
            answer, docs = query_document(
                query=query,
                rag_instance=rag_instance,
                show_context=False,  # Set to True to see retrieved context
            )
            
            results.append({
                "query": query,
                "answer": answer,
                "docs": docs,
                "success": True,
            })
            
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            results.append({
                "query": query,
                "answer": None,
                "docs": None,
                "success": False,
                "error": str(e),
            })
        
        print()
    
    # =========================================================================
    # STEP 4: SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"Document: {filename}")
    print(f"Queries processed: {successful}/{total}")
    print()
    
    if successful == total:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"⚠️  {total - successful} test(s) failed")
    
    print()
    print("=" * 80)
    
    return successful == total


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run complete RAG-Anything pipeline test"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="attention.pdf",
        help="PDF filename to process (default: attention.pdf)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if document is cached",
    )
    parser.add_argument(
        "--query",
        type=str,
        action="append",
        help="Add a custom query (can be used multiple times)",
    )
    
    args = parser.parse_args()
    
    # Use custom queries if provided, otherwise use defaults
    queries = args.query if args.query else None
    
    try:
        success = run_complete_pipeline(
            filename=args.filename,
            force_reprocess=args.force,
            queries=queries,
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
