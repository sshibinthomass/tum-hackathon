# Usage Examples

## Complete Workflow

### 1. Store Documents (First Time)

```bash
# Navigate to project root
cd /Users/qtf4195/tum-hackathon

# Store a document
python rag_anything_pipeline/store_documents.py --filename "Allplan_2020_Manual.pdf"
```

**Expected Output:**
```
================================================================================
RAG-ANYTHING DOCUMENT STORAGE
================================================================================
Document: Allplan_2020_Manual.pdf

================================================================================
INITIALIZING MODELS
================================================================================
✓ Using Ollama models (provider: ollama)

================================================================================
INITIALIZING RAG-ANYTHING
================================================================================
✓ RAG-Anything initialized successfully!
  Working directory: ./rag_storage
  Parser: mineru
  Parse method: auto

================================================================================
PROCESSING DOCUMENT
================================================================================
File: /Users/qtf4195/tum-hackathon/data/Allplan_2020_Manual.pdf
Output directory: ./output/raganything_processed
Extracting: text, tables, formulas, and images...

This may take several minutes...

[Processing output...]

================================================================================
✓ DOCUMENT PROCESSING COMPLETE!
================================================================================
  Processed content stored in: ./output/raganything_processed
  RAG-Anything storage: ./rag_storage

  You can now use query_documents.py to query the stored documents.
================================================================================
```

### 2. Query Documents

#### Single Query
```bash
python rag_anything_pipeline/query_documents.py --query "How do I control pattern height in Allplan?"
```

#### Interactive Mode
```bash
python rag_anything_pipeline/query_documents.py --interactive
```

#### With Context Display
```bash
python rag_anything_pipeline/query_documents.py --query "Your question" --show-context
```

### 3. Reprocess Document (If Needed)

```bash
python rag_anything_pipeline/store_documents.py --filename "Allplan_2020_Manual.pdf" --force
```

## Python API Usage

You can also import and use these functions in your own code:

```python
from rag_anything_pipeline.store_documents import store_document
from rag_anything_pipeline.query_documents import (
    create_rag_instance,
    initialize_models,
    create_rag_anything_config,
    query_document,
    verify_storage,
)

# Store a document
store_document("Allplan_2020_Manual.pdf")

# Query documents
chat_model, embedding_model = initialize_models()
config = create_rag_anything_config()
rag_instance = create_rag_instance(chat_model, embedding_model, config)

answer, docs = query_document(
    "How do I use patterns?",
    rag_instance,
    show_context=True
)
```

## Integration with Evaluation

```python
from rag_anything_pipeline.query_documents import (
    create_rag_instance,
    initialize_models,
    create_rag_anything_config,
)
from ai_eval.resources import eval_dataset_builder as eval

# Initialize RAG instance
chat_model, embedding_model = initialize_models()
config = create_rag_anything_config()
rag_instance = create_rag_instance(chat_model, embedding_model, config)

# Use with evaluation framework
builder = eval.EvalDatasetBuilder(rag_instance)
dataset = builder.build_evaluation_dataset(
    input_contexts=ground_truth_contexts,
    sample_queries=sample_queries,
    expected_responses=expected_responses,
)
```

## Command-Line Options

### store_documents.py

```bash
python store_documents.py --help

Options:
  --filename FILENAME    Name of PDF file in data directory
  --force                Force reprocessing even if cached
  --output-dir DIR       Output directory (default: ./output/raganything_processed)
```

### query_documents.py

```bash
python query_documents.py --help

Options:
  --query TEXT           The question to ask
  --interactive          Run in interactive mode
  --show-context         Show retrieved context documents
  --output-dir DIR       Output directory (default: ./output/raganything_processed)
  --working-dir DIR      RAG-Anything working directory (default: ./rag_storage)
```


