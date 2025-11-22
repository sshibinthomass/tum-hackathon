# RAG-Anything Workflow Guide

## Overview

The notebook has been reorganized into **5 clear steps** that separate document processing (storage) from querying (retrieval). This allows you to:
- Process documents once and cache them
- Query multiple times without reprocessing
- Easily switch between RAG implementations

## Step-by-Step Workflow

### Step 1: Setup & Configuration
**Cell**: After model initialization  
**Purpose**: Import libraries and create RAG-Anything configuration  
**Runs**: Once per session

```python
# Creates config with optimized settings
config = RAGAnythingConfig(
    parser="mineru",
    parse_method="auto",
    enable_image_processing=True,
    enable_table_processing=True,
    enable_equation_processing=True,
)
```

### Step 2: Initialize RAG-Anything
**Cell**: After `chat_model` and `embedding_model` are set up  
**Purpose**: Initialize RAG-Anything with model functions  
**Runs**: Once per session

```python
# Creates LLM and embedding function wrappers
# Initializes RAG-Anything with LightRAG instance
rag_anything = RAGAnything(
    config=config,
    llm_model_func=llm_func,
    embedding_func=embedding_func,
)
```

**Important**: This must be run before Step 3.

### Step 3: Process & Store Documents (One-Time)
**Cell**: After Step 2  
**Purpose**: Process PDF and store in RAG-Anything storage  
**Runs**: Once (cached after first run)

**Features**:
- ✅ Checks if documents are already processed
- ✅ Skips processing if cache exists
- ✅ Stores processed content in `./output/raganything_processed/`
- ✅ Stores in RAG-Anything working directory (`./rag_storage/`)

**What it does**:
1. Parses PDF (text, images, tables, formulas)
2. Stores content in LightRAG storage
3. Creates a `.processed` marker file
4. Caches results for future runs

**To reprocess**: Delete the `.processed` marker file and run again.

### Step 4: Query & Retrieve (Can Run Multiple Times)
**Cell**: After Step 3  
**Purpose**: Query stored documents and get answers  
**Runs**: Multiple times (as many as needed)

**Features**:
- ✅ Verifies documents are processed
- ✅ Retrieves relevant documents
- ✅ Generates answers using retrieved context
- ✅ Can be run with different queries

**Example**:
```python
# Change this query as needed
test_query = "Your question here..."

# Retrieve
relevant_docs = rag_anything_rag.retrieve(question=test_query)

# Get full answer
answer, docs = rag_anything_rag.answer(question=test_query)
```

### Step 5: Use with Evaluation Framework
**Cell**: After Step 4  
**Purpose**: Switch to RAG-Anything for evaluation  
**Runs**: Once before evaluation

**Features**:
- ✅ Easy switching between RAG-Anything and FAISSRAG
- ✅ Compatible with existing evaluation framework
- ✅ Works with `EvalDatasetBuilder` and `DeepEvalScorer`

```python
USE_RAG_ANYTHING = True  # Set to False for FAISSRAG
rag = rag_anything_rag if USE_RAG_ANYTHING else FAISSRAG(...)
```

## Execution Order

### First Time Setup:
1. Run cells 1-9: Basic setup, load data, initialize models
2. **Step 1**: Setup & Configuration
3. **Step 2**: Initialize RAG-Anything
4. **Step 3**: Process & Store Documents (takes several minutes)
5. **Step 4**: Query & Retrieve (test queries)
6. **Step 5**: Use with Evaluation Framework
7. Continue with evaluation cells

### Subsequent Runs (Documents Already Processed):
1. Run cells 1-9: Basic setup, load data, initialize models
2. **Step 1**: Setup & Configuration
3. **Step 2**: Initialize RAG-Anything
4. **Step 3**: Skip (documents already processed - shows cached message)
5. **Step 4**: Query & Retrieve (run as many times as needed)
6. **Step 5**: Use with Evaluation Framework
7. Continue with evaluation cells

## Caching Mechanism

### Storage Locations:
- **Processed documents**: `./output/raganything_processed/`
- **RAG-Anything storage**: `./rag_storage/` (LightRAG storage)
- **Cache marker**: `./output/raganything_processed/.processed`

### Cache Check:
Step 3 automatically checks for the `.processed` marker file:
- ✅ **Exists**: Skips processing, shows cached message
- ❌ **Missing**: Processes document and creates marker

### To Reprocess:
```bash
# Delete the marker file
rm ./output/raganything_processed/.processed

# Or delete entire output directory
rm -rf ./output/raganything_processed/
```

## Benefits of This Structure

1. **Separation of Concerns**:
   - Storage (Step 3) is separate from retrieval (Step 4)
   - Clear workflow with distinct steps

2. **Efficiency**:
   - Process once, query many times
   - No unnecessary reprocessing
   - Fast iteration on queries

3. **Flexibility**:
   - Easy to switch between RAG implementations
   - Can test different queries without reprocessing
   - Compatible with existing evaluation framework

4. **Error Handling**:
   - Clear error messages for missing steps
   - Verification checks at each step
   - Helpful guidance messages

## Troubleshooting

### Error: "rag_anything not initialized"
**Solution**: Run Step 2 before Step 3

### Error: "Documents not processed yet"
**Solution**: Run Step 3 before Step 4

### Error: "LightRAG instance is None"
**Solution**: Make sure Step 2 completed successfully and models are initialized

### Documents not being cached
**Check**:
- Marker file exists: `./output/raganything_processed/.processed`
- Storage directory exists: `./rag_storage/`
- No errors during Step 3 execution

## Testing Checklist

- [ ] Step 1: Configuration created successfully
- [ ] Step 2: RAG-Anything initialized with LightRAG instance
- [ ] Step 3: Document processed and cached (first run)
- [ ] Step 3: Cache detected and skipped (second run)
- [ ] Step 4: Retrieval works correctly
- [ ] Step 4: Answer generation works correctly
- [ ] Step 5: Evaluation framework integration works
- [ ] Can run Step 4 multiple times with different queries
- [ ] Can switch between RAG-Anything and FAISSRAG

## File Structure

```
tum-hackathon/
├── src/notebooks/
│   ├── tum_hackathon_hybrid.ipynb  # Main notebook
│   └── output/
│       └── raganything_processed/  # Processed documents
│           ├── .processed          # Cache marker
│           └── Allplan_2020_Manual/
│               └── auto/           # Parsed content
├── rag_storage/                     # RAG-Anything storage
│   ├── kv_storage/                 # Key-value storage
│   ├── vector_storage/             # Vector embeddings
│   └── graph_storage/              # Knowledge graph
└── data/
    └── Allplan_2020_Manual.pdf     # Source document
```

## Next Steps

After completing all steps:
1. Run evaluation with `EvalDatasetBuilder`
2. Compare RAG-Anything vs FAISSRAG performance
3. Analyze results with `DeepEvalScorer`
4. Iterate on queries and parameters

