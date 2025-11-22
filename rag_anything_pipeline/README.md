# RAG-Anything Pipeline - Detailed Step-by-Step Guide

This document provides a comprehensive, step-by-step explanation of how the RAG-Anything pipeline works, breaking down each component and the flow of execution.

## Table of Contents

1. [Overview](#overview)
2. [Architecture & Flow](#architecture--flow)
3. [store_documents.py - Detailed Breakdown](#store_documentspy---detailed-breakdown)
4. [query_documents.py - Detailed Breakdown](#query_documentspy---detailed-breakdown)
5. [test_pipeline.py - Detailed Breakdown](#test_pipelinepy---detailed-breakdown)
6. [How They Work Together](#how-they-work-together)
7. [Technical Deep Dive](#technical-deep-dive)

---

## Overview

The RAG-Anything pipeline consists of three main scripts that work together to:
1. **Process and store** PDF documents (one-time operation)
2. **Query** stored documents (multiple times)
3. **Test** the complete pipeline end-to-end

The pipeline uses RAG-Anything, which is built on top of LightRAG, to provide advanced multimodal document processing capabilities including text, images, tables, and mathematical formulas.

---

## Architecture & Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCUMENT PROCESSING                      │
│                    (store_documents.py)                      │
│                                                              │
│  PDF File → Parse → Extract → Embed → Store → Cache        │
│              (MinerU)  (Text/Images/  (Vector)  (LightRAG)  │
│                         Tables/Formulas)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    [Storage Created]
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      DOCUMENT QUERYING                       │
│                    (query_documents.py)                      │
│                                                              │
│  Query → Embed → Retrieve → Generate → Answer               │
│          (Vector)  (Similarity)  (LLM)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    END-TO-END TESTING                       │
│                    (test_pipeline.py)                        │
│                                                              │
│  Store → Initialize → Query → Validate → Report             │
└─────────────────────────────────────────────────────────────┘
```

---

## store_documents.py - Detailed Breakdown

### Purpose
Processes PDF documents and stores them in RAG-Anything's storage system. This is a **one-time operation** per document - results are cached for future use.

### Step-by-Step Execution Flow

#### Step 1: Command-Line Parsing (`main()` function)
**Lines: 272-325**

1. **Detects execution environment** (lines 276-288)
   - Checks if running in Jupyter/IPython
   - If in interactive mode, skips argument parsing
   - Otherwise, parses command-line arguments

2. **Parses arguments** (lines 290-311)
   - `--filename`: PDF filename (default: "attention.pdf")
   - `--force`: Force reprocessing even if cached
   - `--output-dir`: Output directory for processed documents

3. **Calls `store_document()`** (line 314)

#### Step 2: Document Storage Function (`store_document()`)
**Lines: 170-267**

**2.1. Path Setup & Validation** (lines 192-195)
```python
file_path = Path(glob.DATA_PKG_DIR) / filename
if not file_path.exists():
    raise FileNotFoundError(f"Document not found: {file_path}")
```
- Constructs full path to PDF file
- Validates file exists

**2.2. Output Directory Setup** (lines 197-201)
```python
if output_dir is None:
    output_dir = "./output/raganything_processed"
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)
```
- Creates output directory if it doesn't exist
- Default: `./output/raganything_processed`

**2.3. Cache Check** (lines 203-211)
```python
processed_marker = output_path / ".processed"
if processed_marker.exists() and not force_reprocess:
    print("✓ Document already processed and stored!")
    return False
```
- Checks for `.processed` marker file
- If exists and not forcing, skips processing
- Returns `False` to indicate skipping

**2.4. Force Reprocessing** (lines 213-215)
```python
if force_reprocess and processed_marker.exists():
    print("⚠️  Force reprocessing enabled. Removing existing cache...")
    processed_marker.unlink()
```
- Removes cache marker if force reprocessing

#### Step 3: Model Initialization (`initialize_models()`)
**Lines: 38-65**

**3.1. Provider Detection** (lines 44-63)
```python
if glob.MODEL_PROVIDER == "openai":
    chat_model = ChatOpenAI(model="gpt-4o-mini", ...)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
else:
    chat_model = ChatOllama(model="llama3.1:latest", ...)
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
```
- Checks `global_config.MODEL_PROVIDER`
- Initializes appropriate models:
  - **OpenAI**: GPT-4o-mini + text-embedding-3-small
  - **Ollama**: llama3.1:latest + nomic-embed-text

**3.2. Returns Models** (line 65)
- Returns tuple: `(chat_model, embedding_model)`

#### Step 4: RAG-Anything Configuration (`create_rag_anything_config()`)
**Lines: 70-86**

**4.1. Configuration Creation** (lines 76-85)
```python
config = RAGAnythingConfig(
    working_dir=working_dir,          # "./rag_storage"
    parser=parser,                     # "mineru"
    parse_method=parse_method,         # "auto"
    parser_output_dir="./output",
    enable_image_processing=True,
    enable_table_processing=True,
    enable_equation_processing=True,
    display_content_stats=True,
)
```
- **working_dir**: Where RAG-Anything stores its internal data
- **parser**: "mineru" (advanced PDF parser)
- **parse_method**: "auto" (intelligently chooses parsing strategy)
- **Multimodal features**: Images, tables, equations enabled

#### Step 5: RAG-Anything Initialization (`initialize_rag_anything()`)
**Lines: 91-148**

This is the most complex step, creating async wrappers for LightRAG compatibility.

**5.1. LLM Function Wrapper** (lines 100-112)
```python
async def llm_func(prompt, **kwargs):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, chat_model.invoke, prompt)
    if hasattr(response, "content"):
        return response.content
    return str(response)

llm_func.func = llm_func  # LightRAG decorator pattern
```
- **Why async?**: LightRAG (underlying RAG-Anything) expects async functions
- **Problem**: LangChain models are synchronous
- **Solution**: Wraps sync call in `run_in_executor()` to make it async
- **`.func` attribute**: Required by LightRAG's decorator pattern

**5.2. Embedding Function Wrapper** (lines 114-134)
```python
# Get embedding dimension first
sample_embedding = embedding_model.embed_documents(["test"])
embedding_dim = len(sample_embedding[0])

async def embedding_func(texts):
    if isinstance(texts, str):
        texts = [texts]
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        None, embedding_model.embed_documents, texts
    )
    return embeddings

embedding_func.embedding_dim = embedding_dim
embedding_func.func = embedding_func
```
- **Dimension detection**: Gets embedding dimension by testing
- **String handling**: Converts single strings to lists
- **Async wrapper**: Same pattern as LLM function
- **`.embedding_dim` attribute**: Required by LightRAG

**5.3. RAG-Anything Instance Creation** (lines 137-141)
```python
rag_anything = RAGAnything(
    config=config,
    llm_model_func=llm_func,
    embedding_func=embedding_func,
)
```
- Creates RAG-Anything instance with configuration and model functions
- Internally creates LightRAG instance for knowledge graph storage

#### Step 6: Document Processing (`process_document()`)
**Lines: 153-165**

**6.1. Async Processing** (lines 239-246)
```python
asyncio.run(
    process_document(
        rag_anything=rag_anything,
        file_path=str(file_path),
        output_dir=output_dir,
        parse_method="auto",
    )
)
```
- Runs async document processing
- Calls `rag_anything.process_document_complete()`

**6.2. What Happens Inside `process_document_complete()`**:
1. **PDF Parsing** (MinerU parser):
   - Extracts text with structure preservation
   - Extracts images and saves them
   - Extracts tables in structured format
   - Extracts mathematical formulas (LaTeX)
   
2. **Content Processing**:
   - Chunks text intelligently
   - Creates embeddings for text chunks
   - Processes images (extracts, embeds if configured)
   - Processes tables (structured extraction)
   - Processes equations (LaTeX extraction)

3. **Storage**:
   - Stores in LightRAG knowledge graph (`./rag_storage/`)
   - Creates entity-relationship graph
   - Stores vector embeddings
   - Saves processed content to `output_dir`

**6.3. Cache Marker** (line 249)
```python
processed_marker.touch()
```
- Creates `.processed` file to mark completion
- Used for cache detection on future runs

#### Step 7: Completion
**Lines: 251-260**
- Prints success message
- Shows storage locations
- Returns `True` if successful

---

## query_documents.py - Detailed Breakdown

### Purpose
Queries documents that have been processed and stored by `store_documents.py`. Can be run multiple times with different queries.

### Step-by-Step Execution Flow

#### Step 1: Command-Line Parsing (`main()`)
**Lines: 289-368**

**1.1. Environment Detection** (lines 293-305)
- Same as `store_documents.py` - detects Jupyter/IPython

**1.2. Argument Parsing** (lines 307-337)
- `--query`: Single question to ask
- `--interactive`: Interactive mode for multiple queries
- `--output-dir`: Where processed documents are stored
- `--working-dir`: RAG-Anything working directory
- `--show-context`: Show retrieved context documents

**1.3. Storage Verification** (line 342)
```python
if not verify_storage(args.output_dir):
    return 1
```

#### Step 2: Storage Verification (`verify_storage()`)
**Lines: 145-159**

**2.1. Check for Processed Marker** (lines 147-157)
```python
processed_marker = Path(output_dir) / ".processed"
if not processed_marker.exists():
    print("❌ ERROR: Documents not processed yet!")
    return False
```
- Checks if `.processed` marker exists
- If not, prompts user to run `store_documents.py` first
- Returns `False` if not processed

#### Step 3: Model Initialization
**Lines: 40-61**
- Same as `store_documents.py` - initializes chat and embedding models

#### Step 4: RAG-Anything Configuration
**Lines: 66-82**
- Same as `store_documents.py`, but with `display_content_stats=False`

#### Step 5: RAG-Anything Initialization
**Lines: 87-140**
- Same as `store_documents.py` - creates async wrappers and initializes RAG-Anything

#### Step 6: RAG Instance Creation (`create_rag_instance()`)
**Lines: 164-180**

**6.1. Initialize RAG-Anything** (line 169)
```python
rag_anything = initialize_rag_anything(chat_model, embedding_model, config)
```

**6.2. Create RAGAnythingRAG Wrapper** (lines 172-178)
```python
rag_anything_rag = RAGAnythingRAG(
    llm=chat_model,
    documents=None,  # RAG-Anything manages its own documents
    k=3,  # Number of documents to retrieve
    rag_anything_instance=rag_anything,
    processed_docs_cache=None,
)
```
- **RAGAnythingRAG**: Wrapper class that integrates RAG-Anything with evaluation framework
- **k=3**: Retrieves top 3 most relevant documents
- **documents=None**: RAG-Anything loads documents from storage
- **rag_anything_instance**: The initialized RAG-Anything instance

#### Step 7: Query Execution (`query_document()`)
**Lines: 185-252**

**7.1. Document Retrieval** (lines 208-231)
```python
relevant_docs = rag_instance.retrieve(question=query)
```
- **What happens**:
  1. Embeds the query using embedding model
  2. Searches RAG-Anything storage for similar content
  3. Uses LightRAG's knowledge graph for semantic search
  4. Returns top `k` most relevant documents (default: 3)
  
- **Document structure**:
  - `doc.page_content`: The text content
  - `doc.metadata`: Source information, page numbers, etc.

**7.2. Context Display** (lines 212-224)
- If `show_context=True`, displays preview of retrieved documents
- Shows first 200 characters of each document
- Shows source metadata

**7.3. Answer Generation** (lines 234-245)
```python
answer, relevant_docs = rag_instance.answer(question=query)
```
- **What happens**:
  1. Retrieves relevant documents (if not already done)
  2. Combines retrieved context
  3. Constructs prompt with question + context
  4. Calls LLM to generate answer
  5. Returns answer and relevant documents

**7.4. Return Results** (line 245)
- Returns tuple: `(answer: str, relevant_docs: List[Document])`

#### Step 8: Interactive Mode (`interactive_mode()`)
**Lines: 257-284**

**8.1. Loop Structure** (lines 265-282)
```python
while True:
    query = input("Query: ").strip()
    if query.lower() in ["quit", "exit", "q"]:
        break
    query_document(query, rag_instance, show_context=False)
```
- Continuously prompts for queries
- Processes each query
- Exits on "quit", "exit", or "q"
- Handles KeyboardInterrupt gracefully

---

## test_pipeline.py - Detailed Breakdown

### Purpose
Runs the complete pipeline end-to-end: stores a document, initializes query system, and runs multiple test queries.

### Step-by-Step Execution Flow

#### Step 1: Command-Line Parsing (`main()`)
**Lines: 176-220**

**1.1. Argument Parsing** (lines 178-197)
- `--filename`: PDF to process (default: "attention.pdf")
- `--force`: Force reprocessing
- `--query`: Custom queries (can be used multiple times)

**1.2. Query Setup** (line 202)
```python
queries = args.query if args.query else None
```
- Uses custom queries if provided
- Otherwise uses defaults in `run_complete_pipeline()`

#### Step 2: Complete Pipeline Execution (`run_complete_pipeline()`)
**Lines: 35-173**

**2.1. Default Queries** (lines 54-59)
```python
if queries is None:
    queries = [
        "What is the attention mechanism?",
        "What is the Transformer architecture?",
        "How does self-attention work?",
    ]
```

**2.2. Step 1: Store Document** (lines 64-80)
```python
was_processed = store_document(
    filename=filename,
    force_reprocess=force_reprocess,
)
```
- Calls `store_document()` from `store_documents.py`
- Handles errors gracefully
- Reports success/failure

**2.3. Step 2: Initialize Query System** (lines 87-106)
```python
chat_model, embedding_model = init_models_query()
config = create_rag_anything_config()
rag_instance = create_rag_instance(chat_model, embedding_model, config)
```
- Initializes models
- Creates configuration
- Creates RAG instance for querying

**2.4. Step 3: Run Queries** (lines 113-148)
```python
for i, query in enumerate(queries, 1):
    answer, docs = query_document(
        query=query,
        rag_instance=rag_instance,
        show_context=False,
    )
    results.append({
        "query": query,
        "answer": answer,
        "docs": docs,
        "success": True,
    })
```
- Iterates through all queries
- Calls `query_document()` for each
- Collects results in list
- Handles errors per query (doesn't stop on failure)

**2.5. Step 4: Summary** (lines 153-173)
```python
successful = sum(1 for r in results if r["success"])
total = len(results)
```
- Counts successful queries
- Prints summary report
- Returns `True` if all queries succeeded

---

## How They Work Together

### Complete Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER RUNS: python store_documents.py --filename "doc.pdf"│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. store_documents.py:                                      │
│    - Validates file exists                                   │
│    - Checks cache (.processed marker)                       │
│    - Initializes models (OpenAI/Ollama)                     │
│    - Creates RAG-Anything config                             │
│    - Initializes RAG-Anything with async wrappers          │
│    - Processes PDF (MinerU parsing)                          │
│    - Extracts: text, images, tables, formulas               │
│    - Stores in LightRAG knowledge graph                     │
│    - Creates .processed marker                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. STORAGE CREATED:                                          │
│    - ./rag_storage/ (LightRAG knowledge graph)              │
│    - ./output/raganything_processed/ (processed content)    │
│    - .processed marker file                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. USER RUNS: python query_documents.py --query "question"   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. query_documents.py:                                       │
│    - Verifies .processed marker exists                       │
│    - Initializes models (same as storage)                   │
│    - Creates RAG-Anything config                            │
│    - Initializes RAG-Anything                               │
│    - Creates RAGAnythingRAG wrapper                         │
│    - Embeds query                                            │
│    - Retrieves relevant documents (semantic search)         │
│    - Generates answer using LLM                             │
│    - Returns answer + context                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. USER RUNS: python test_pipeline.py                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. test_pipeline.py:                                         │
│    - Calls store_document() (Step 1)                         │
│    - Initializes query system (Step 2)                      │
│    - Runs multiple queries (Step 3)                          │
│    - Generates summary report (Step 4)                      │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
PDF File
   ↓
[MinerU Parser]
   ↓
┌─────────────────┐
│ Text Chunks     │
│ Images          │
│ Tables          │
│ Formulas        │
└─────────────────┘
   ↓
[Embedding Model]
   ↓
[Vector Embeddings]
   ↓
[LightRAG Storage]
   ↓
┌─────────────────┐
│ Knowledge Graph │
│ Entity-Relation │
│ Vector Store    │
└─────────────────┘
   ↓
[Query Time]
   ↓
Query → Embed → Search → Retrieve → Generate → Answer
```

---

## Technical Deep Dive

### Async/Sync Bridge

**Problem**: 
- LightRAG (used by RAG-Anything) expects async functions
- LangChain models are synchronous

**Solution**:
```python
async def llm_func(prompt, **kwargs):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, chat_model.invoke, prompt)
    return response.content
```

**How it works**:
1. `run_in_executor()` runs sync function in thread pool
2. `await` makes it async-compatible
3. LightRAG can call it as async function

### LightRAG Integration

RAG-Anything uses LightRAG under the hood, which:
- Creates a knowledge graph from documents
- Stores entities and relationships
- Uses graph + vector search for retrieval
- Provides better semantic understanding than pure vector search

### RAGAnythingRAG Wrapper

The `RAGAnythingRAG` class (from `ai_eval.resources.rag_template`) bridges:
- **RAG-Anything's internal storage** (LightRAG knowledge graph)
- **Evaluation framework interface** (retrieve, answer methods)

**Key Methods**:
- `retrieve(question)`: Gets relevant documents from RAG-Anything
- `answer(question)`: Retrieves + generates answer
- `generate(question, context)`: Generates answer from context

### Caching Strategy

**Storage Cache**:
- `.processed` marker file indicates document is processed
- If marker exists, `store_documents.py` skips processing
- Use `--force` to reprocess

**Model Initialization**:
- Models are re-initialized each run (no caching)
- This is intentional for flexibility

**RAG-Anything Storage**:
- LightRAG stores data in `./rag_storage/`
- This persists between runs
- Only needs to be created once per document

### Error Handling

**store_documents.py**:
- FileNotFoundError: Document doesn't exist
- Processing errors: Caught and displayed with traceback
- KeyboardInterrupt: Graceful exit

**query_documents.py**:
- Storage verification: Checks for `.processed` marker
- Retrieval errors: Caught and displayed
- Answer generation errors: Caught and displayed

**test_pipeline.py**:
- Per-query error handling: One failure doesn't stop others
- Summary report shows success/failure counts
- Returns exit code based on overall success

---

## File Structure

```
rag_anything_pipeline/
├── store_documents.py      # Document processing & storage
├── query_documents.py      # Document querying
├── test_pipeline.py       # End-to-end testing
└── README.md              # This file

Output directories:
├── output/
│   └── raganything_processed/  # Processed document content
│       ├── .processed          # Cache marker
│       └── [document_name]/    # Per-document processed files
└── rag_storage/                # RAG-Anything internal storage
    ├── graph_chunk_entity_relation.graphml  # Knowledge graph
    ├── kv_store_*.json         # Key-value stores
    └── vdb_*.json              # Vector databases
```

---

## Usage Examples

### Example 1: Process a Document

```bash
cd rag_anything_pipeline
python store_documents.py --filename "Allplan_2020_Manual.pdf"
```

**What happens**:
1. Validates file exists in `data/` directory
2. Checks if already processed (looks for `.processed` marker)
3. If not processed:
   - Initializes models
   - Creates RAG-Anything instance
   - Parses PDF with MinerU
   - Extracts text, images, tables, formulas
   - Stores in LightRAG knowledge graph
   - Creates cache marker
4. Prints success message with storage locations

### Example 2: Query Documents

```bash
python query_documents.py --query "How do I set pattern height?"
```

**What happens**:
1. Verifies documents are processed
2. Initializes models and RAG-Anything
3. Creates RAGAnythingRAG wrapper
4. Embeds the query
5. Searches knowledge graph for relevant content
6. Retrieves top 3 most relevant documents
7. Generates answer using LLM with retrieved context
8. Displays answer

### Example 3: Interactive Mode

```bash
python query_documents.py --interactive
```

**What happens**:
1. Same initialization as single query
2. Enters loop:
   - Prompts for query
   - Processes query
   - Displays answer
   - Repeats until "quit"
3. Exits gracefully

### Example 4: Complete Pipeline Test

```bash
python test_pipeline.py --filename "attention.pdf" --query "What is attention?" --query "How does it work?"
```

**What happens**:
1. Stores document (if not already stored)
2. Initializes query system
3. Runs each query sequentially
4. Collects results
5. Prints summary report

---

## Configuration

### Model Provider

Set in `ai_eval.config.global_config.MODEL_PROVIDER`:
- `"openai"`: Uses OpenAI models (requires API key)
- `"ollama"`: Uses Ollama models (requires Ollama running locally)

### Model Settings

**OpenAI**:
- Chat: `gpt-4o-mini` (temperature=0.1)
- Embeddings: `text-embedding-3-small`

**Ollama**:
- Chat: `llama3.1:latest` (temperature=0.1)
- Embeddings: `nomic-embed-text`

### RAG-Anything Settings

**Parser**: `mineru` (advanced PDF parser)
**Parse Method**: `auto` (intelligent method selection)
**Multimodal**: Images, tables, equations enabled
**Retrieval**: Top `k=3` documents

---

## Troubleshooting

### Error: "Documents not processed yet"
**Cause**: `.processed` marker doesn't exist
**Solution**: Run `store_documents.py` first

### Error: "Document not found"
**Cause**: File doesn't exist in `data/` directory
**Solution**: Check filename and path

### Error: "Models not initialized"
**Cause**: Ollama not running or OpenAI API key not set
**Solution**: 
- For Ollama: Start Ollama service
- For OpenAI: Set `OPENAI_API_KEY` environment variable

### Slow Processing
**Cause**: First-time processing is slow (parsing + embedding)
**Solution**: Normal behavior - subsequent queries are fast

### Want to Reprocess
**Solution**: Use `--force` flag:
```bash
python store_documents.py --filename "doc.pdf" --force
```

---

## Key Concepts

### RAG-Anything
- Advanced multimodal RAG framework
- Built on LightRAG (knowledge graph RAG)
- Supports text, images, tables, formulas
- Uses MinerU for PDF parsing

### LightRAG
- Knowledge graph-based RAG
- Creates entity-relationship graphs
- Combines graph + vector search
- Better semantic understanding

### MinerU
- Advanced PDF parser
- Better structure preservation than PyPDFLoader
- Extracts images, tables, formulas
- Maintains document layout

### RAGAnythingRAG
- Wrapper class for evaluation framework compatibility
- Bridges RAG-Anything and evaluation framework
- Implements standard RAG interface (retrieve, answer, generate)

---

## Summary

This pipeline provides a complete solution for:
1. **Processing** PDF documents with advanced parsing
2. **Storing** processed content in a knowledge graph
3. **Querying** stored documents with semantic search
4. **Testing** the complete pipeline end-to-end

The architecture separates concerns:
- **Storage** (one-time): `store_documents.py`
- **Querying** (multiple times): `query_documents.py`
- **Testing** (validation): `test_pipeline.py`

Each component is designed to be:
- **Modular**: Can be used independently
- **Cached**: Avoids reprocessing
- **Error-resilient**: Handles errors gracefully
- **Flexible**: Supports different models and configurations
