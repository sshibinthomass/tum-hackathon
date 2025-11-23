# Document Processing and Query System

A RAG (Retrieval-Augmented Generation) system for processing and querying PDF documents with multimodal support (text, images, tables, equations).

## Overview

The system consists of three main components:

- **Storage Pipeline** (`simple_rag/store_improved.py`): Processes PDF documents, extracts entities/relationships, and stores them in a knowledge graph
- **Query Pipeline** (`simple_rag/query_improved.py`): Queries the stored knowledge graph using vector search and LLM generation
- **Evaluation Notebook** (`src/notebooks/tum_hackathon_graph.ipynb`): Evaluates RAG performance using QA datasets and DeepEval metrics

## Configuration

Edit `simple_rag/config.py` to configure models and settings:

```python
# Storage settings
chat_provider = "openai"
embedding_provider = "openai"
chat_model = "gpt-4o-mini"
embedding_model = "text-embedding-3-large"
parser = "mineru"

# Query settings
query_chat_provider = "openai"
query_embedding_provider = "openai"
query_chat_model = "gpt-4o-mini"
query_embedding_model = "text-embedding-3-large"  # Must match storage model
query_rag_storage_path = "./rag_storage_1"  # Path to storage directory
```

## Installation

```bash
# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate  # macOS/Linux

# Install dependencies
uv sync

# Install package
uv pip install -e .
```

## Usage

### 1. Store Documents (`store_improved.py`)

Process PDF documents and build the knowledge graph:

```bash
# Process a single document
python simple_rag/store_improved.py

# Or modify store_improved.py to specify filename
# Default: "Allplan_2020_Manual.pdf"
```

**Storage Output:**

The storage pipeline creates two main output directories:

1. **MinerU Output** (`output_<model_config>/` or `output_1/`):

   - Raw parser output from MinerU PDF extraction
   - Contains structured document files:
     - `*_layout.pdf` - Layout-preserved PDF
     - `*_origin.pdf` - Original PDF
     - `*_span.pdf` - Span-annotated PDF
     - `*.md` - Markdown representation
     - `*_model.json` - Document structure model
     - `*_content_list.json` - Content index
     - `*_middle.json` - Intermediate processing data
     - `images/` - Extracted images (JPG format)

2. **RAG Storage** (`rag_storage_<model_config>/` or `rag_storage_1/`):
   - Processed knowledge graph storage:
     - `vdb_chunks.json` - Vector embeddings of text chunks
     - `vdb_entities.json` - Vector embeddings of extracted entities
     - `vdb_relationships.json` - Vector embeddings of relationships
     - `kv_store_*.json` - Key-value stores for documents, entities, relations
     - `graph_chunk_entity_relation.graphml` - Knowledge graph visualization
     - `allplan_manual_graph.html` - Interactive graph visualization

### 2. Query Documents (`query_improved.py`)

Query the stored knowledge graph:

```bash
# Query with a question
python simple_rag/query_improved.py "Your question here"

# Run QA test against dataset
python simple_rag/query_improved.py --test-qa --num-questions 5

# Specify custom storage path
python simple_rag/query_improved.py "Question" --storage-path ./rag_storage_1
```

**Query Parameters:**

- `mode="naive"` - Pure vector search (best for metadata queries)
- `top_k=50` - Number of chunks to retrieve
- `cosine_threshold=0.1` - Similarity threshold

### 3. Evaluate RAG Performance (`tum_hackathon_graph.ipynb`)

Run the Jupyter notebook to evaluate RAG performance:

```bash
# Start Jupyter
jupyter notebook

# Open src/notebooks/tum_hackathon_graph.ipynb
```

**Workflow:**

1. **Load QA Dataset**: Reads evaluation questions and expected answers from `data/generated_qa_data_tum.json`
2. **Initialize RAG Model**:
   - Loads knowledge graph from `rag_storage_1` directory
   - Validates embedding dimensions match (3072 for text-embedding-3-large)
   - Supports multiple LLM providers (OpenAI, Ollama with qwen2.5vl:3b, etc.)
   - Uses `RAGAnythingRAG` wrapper for querying stored knowledge graph
3. **Build Evaluation Dataset**:
   - Runs all queries through the RAG system
   - Collects retrieved contexts and generated answers
   - Creates test cases for evaluation
4. **Calculate Metrics**: Uses DeepEval framework with LLM-as-judge:
   - **Answer Relevancy**: Measures how well the answer addresses the query
   - **Faithfulness**: Checks if answer is grounded in retrieved context (no hallucinations)
   - **Contextual Recall**: Evaluates if retriever found all relevant information
   - **Contextual Precision**: Measures ranking quality of retrieved documents
5. **Generate Reports**:
   - Overall metrics summary
   - Detailed per-question results
   - Saves results to `data/deepeval_results.json`

**Configuration:**

- Embedding model must match storage: `text-embedding-3-large` (3072 dimensions)
- Chat model can be configured via `glob.MODEL_PROVIDER` (default: Ollama)
- Supports async processing with `nest_asyncio` for Jupyter compatibility

## Technical Details

### Architecture

The system uses a knowledge graph-based RAG architecture with the following components:

- **Document Parser**: MinerU for PDF extraction with layout analysis
- **Vector Database**: JSON-based vector storage with cosine similarity search
- **Knowledge Graph**: GraphML format storing entities, relationships, and document chunks
- **LLM Integration**: LangChain-based wrappers for OpenAI, Anthropic, Groq, Gemini, and Ollama
- **Multimodal Processing**: Vision models (GPT-4o) for image, table, and equation understanding

### Storage Pipeline (`store_improved.py`)

**Processing Flow:**

1. **MinerU Parsing** → `output_<model_config>/`: PDF is parsed into structured format (markdown, JSON, images)
2. **RAG Processing** → `rag_storage_<model_config>/`: Parsed content is processed into knowledge graph with embeddings

**Document Processing:**

- **Parser**: MinerU extracts structured content (text, images, tables, equations) from PDFs
  - Output stored in `output_1/` (or `output_<model_config>/`)
  - Generates layout PDFs, markdown, JSON models, and extracted images
- **Chunking**: Page-based context windows (2 pages before/after) with 3000 token limits
- **Embeddings**: OpenAI `text-embedding-3-large` (3072 dimensions) for semantic search
- **Entity Extraction**: LLM-based extraction of entities with structured format: `entity<|#|>name<|#|>type<|#|>description`
- **Relationship Extraction**: LLM-based extraction of relationships: `relation<|#|>source<|#|>target<|#|>keywords<|#|>description`
- **Vision Processing**: GPT-4o vision model for multimodal content understanding
- **Context Awareness**: Includes headers, captions, and cross-page references

**Storage Format:**

- **MinerU Output** (`output_1/`): Raw parser output with structured PDFs, markdown, JSON, and images
- **RAG Storage** (`rag_storage_1/`):
  - Vector embeddings stored in JSON format (vdb\_\*.json)
  - Key-value stores for full documents, entities, and relationships (kv_store\_\*.json)
  - GraphML visualization of knowledge graph structure
  - Interactive HTML graph visualization
  - Parse cache for efficient reprocessing

### Query Pipeline (`query_improved.py`)

**Retrieval:**

- **Vector Search**: Cosine similarity search over chunk, entity, and relationship embeddings
- **Retrieval Modes**:
  - `naive`: Pure vector search (best for metadata queries)
  - `hybrid`: Combines vector and keyword search
  - `mix`: Alternative hybrid approach
- **Parameters**: `top_k=50`, `cosine_threshold=0.1` for broad retrieval
- **Multi-source Retrieval**: Searches across chunks, entities, and relationships simultaneously

**Generation:**

- **LLM**: GPT-4o-mini with temperature=0.1 for consistent answers
- **Context Assembly**: Combines retrieved chunks with entity/relationship context
- **Prompt Engineering**: Structured prompts for answer generation with source attribution

### Evaluation Notebook (`tum_hackathon_graph.ipynb`)

**Features:**

- **Knowledge Graph Integration**: Loads pre-processed knowledge graph from `rag_storage_1` without rebuilding
- **Multi-Provider Support**:
  - OpenAI (GPT-4o-mini, GPT-4o)
  - Ollama (qwen2.5vl:3b, llama3.1, etc.)
  - Configurable via `glob.MODEL_PROVIDER`
- **DeepEval Framework**: LLM-as-judge evaluation with parallel processing
- **Comprehensive Metrics**:
  - **Answer Relevancy**: Measures how well generated answer addresses the query
  - **Faithfulness**: Detects hallucinations by checking answer grounding in context
  - **Contextual Recall**: Evaluates completeness of retrieved information
  - **Contextual Precision**: Measures ranking quality of retrieved documents
- **Robust Retrieval**: Uses direct chunk access for reliable knowledge graph queries
- **Async Processing**: Full async support with `nest_asyncio` for Jupyter compatibility
- **Embedding Validation**: Automatically validates embedding dimensions match storage
- **Report Generation**:
  - Overall metrics summary with `get_overall_metrics()`
  - Detailed per-question analysis
  - JSON export to `data/deepeval_results.json`

### Storage Structure

**Processing Pipeline:**

1. **MinerU Output** (`output_<model_config>/<document_name>/auto/`):

```
output_1/Allplan_2020_Manual/auto/
├── Allplan_2020_Manual.md              # Markdown representation
├── Allplan_2020_Manual_layout.pdf       # Layout-preserved PDF
├── Allplan_2020_Manual_origin.pdf        # Original PDF
├── Allplan_2020_Manual_span.pdf          # Span-annotated PDF
├── Allplan_2020_Manual_model.json        # Document structure model
├── Allplan_2020_Manual_content_list.json # Content index
├── Allplan_2020_Manual_middle.json       # Intermediate processing data
└── images/                               # Extracted images (JPG)
    └── [248 image files]
```

2. **RAG Storage** (`rag_storage_<model_config>/` or `rag_storage_1/`):

```
rag_storage_1/
├── vdb_chunks.json          # Text chunk embeddings (vector database)
├── vdb_entities.json        # Entity embeddings (vector database)
├── vdb_relationships.json   # Relationship embeddings (vector database)
├── kv_store_text_chunks.json      # Full text chunk metadata
├── kv_store_full_entities.json    # Complete entity records with descriptions
├── kv_store_full_relations.json   # Complete relationship records
├── kv_store_entity_chunks.json    # Entity-to-chunk mappings
├── kv_store_relation_chunks.json  # Relationship-to-chunk mappings
├── kv_store_doc_status.json       # Document processing status
├── kv_store_parse_cache.json      # Cached parse results
├── kv_store_llm_response_cache.json # Cached LLM responses
├── graph_chunk_entity_relation.graphml # Knowledge graph (GraphML)
└── allplan_manual_graph.html            # Interactive graph visualization
```

**Directory Purpose:**

- `output_1/`: MinerU parser output - raw structured extraction from PDFs
- `rag_storage_1/`: RAG storage - processed knowledge graph ready for querying

### Data Formats

**Entity Format:**

```
entity<|#|>name<|#|>type<|#|>description
```

**Relationship Format:**

```
relation<|#|>source<|#|>target<|#|>keywords<|#|>description
```

**Vector Embeddings:**

- Stored as JSON arrays of float vectors (3072 dimensions)
- Indexed by chunk/entity/relationship IDs
- Cosine similarity used for retrieval

**Knowledge Graph:**

- GraphML format with nodes (entities, chunks) and edges (relationships)
- Supports visualization in graph analysis tools

## Environment Variables

Set in `.env` file:

```bash
OPENAI_API_KEY=your_key_here
LOG_DIR=./logs
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5
```

## Technology Stack

- **LLM Framework**: LangChain for model integration and prompt management
- **Embeddings**: OpenAI `text-embedding-3-large` (3072 dimensions)
- **Chat Models**: OpenAI GPT-4o-mini, GPT-4o (vision), with support for Anthropic, Groq, Gemini, Ollama
- **PDF Parser**: MinerU for layout-aware document extraction
- **Vector Storage**: JSON-based vector database with cosine similarity
- **Knowledge Graph**: GraphML format for entity-relationship visualization
- **Evaluation**: DeepEval for LLM-as-judge metrics
- **Async Processing**: asyncio with nest_asyncio for concurrent operations
- **Caching**: In-memory caching for embeddings and LLM responses

## Requirements

- Python 3.12+
- OpenAI API key (or other LLM provider API key)
- MinerU parser installed and configured
- LangChain and related provider libraries (langchain-openai, langchain-anthropic, etc.)
- Jupyter notebook (for evaluation)
