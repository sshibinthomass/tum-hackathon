# RAG-Anything Integration & Optimization

## Overview

This document describes the integration of RAG-Anything into the TUM Hackathon project, providing optimized multimodal RAG capabilities with full compatibility with the existing evaluation framework.

## What Was Done

### 1. Dependencies Added
- **raganything** (>=1.2.8): Core RAG-Anything library for multimodal document processing
- **nest-asyncio** (>=1.6.0): Enables nested event loops in Jupyter notebooks

### 2. New RAG Implementation: `RAGAnythingRAG`

Created a wrapper class (`src/ai_eval/resources/rag_template.py`) that:
- Implements the same `RAG` interface as `FAISSRAG` and `TFIDFRAG`
- Integrates RAG-Anything's powerful document parsing (MinerU/Docling)
- Supports multimodal content: text, images, tables, formulas
- Fully compatible with existing evaluation framework
- Handles async operations properly in Jupyter notebooks

### 3. Notebook Integration

Updated `tum_hackathon_hybrid.ipynb` with:
- RAG-Anything initialization and setup
- Document processing with optimized settings
- Test queries to verify functionality
- Easy switching between FAISSRAG and RAGAnythingRAG

## Key Features

### Multimodal Support
- **Text**: Advanced parsing with better structure preservation
- **Images**: Automatic extraction and embedding
- **Tables**: Structured table extraction
- **Formulas**: LaTeX formula extraction

### Optimized Settings
- **Parser**: MinerU (better PDF parsing than PyPDFLoader)
- **Parse Method**: Auto (intelligently chooses best method)
- **Language**: Optimized for English
- **Chunking**: RAG-Anything's optimized chunking strategy

### Evaluation Framework Compatibility
The `RAGAnythingRAG` class implements the same interface:
- `retrieve(question)` → Returns `List[Document]`
- `generate(question, context)` → Returns `str`
- `answer(question)` → Returns `(str, List[Document])`

This means it works seamlessly with:
- `EvalDatasetBuilder`
- `DeepEvalScorer`
- All existing evaluation metrics

## Usage

### Basic Usage

```python
from raganything import RAGAnything
from ai_eval.resources.rag_template import RAGAnythingRAG

# Initialize RAG-Anything
rag_anything = RAGAnything(
    api_key=os.getenv("OPENAI_API_KEY"),
    parser="mineru",
    parse_method="auto"
)

# Process document (one-time)
await rag_anything.process_document_complete(
    file_path="document.pdf",
    output_dir="./output/",
    table=True,
    formula=True,
    lang="en"
)

# Create RAG instance
rag = RAGAnythingRAG(
    llm=chat_model,
    k=3,
    rag_anything_instance=rag_anything
)

# Use with evaluation framework
from ai_eval.resources import eval_dataset_builder as eval
builder = eval.EvalDatasetBuilder(rag)
dataset = builder.build_evaluation_dataset(...)
```

### Switching Between Implementations

In the notebook, you can easily switch:

```python
# Use RAG-Anything (recommended)
rag = rag_anything_rag

# Or use original FAISS
# rag = FAISSRAG(chat_model, documents, k=3, vectorstore=vectorstore)
```

## Performance Improvements

### Expected Benefits
1. **Better Document Parsing**: MinerU provides superior PDF parsing compared to PyPDFLoader
2. **Multimodal Retrieval**: Can retrieve and use images, tables, and formulas
3. **Optimized Chunking**: RAG-Anything uses intelligent chunking strategies
4. **Structure Preservation**: Better preservation of document structure

### Evaluation Metrics
The evaluation framework will automatically work with RAG-Anything, measuring:
- Answer quality
- Retrieval accuracy
- Context relevance
- All existing DeepEval metrics

## Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key"
```

### Optional Settings
- `parser`: "mineru" or "docling"
- `parse_method`: "auto", "ocr", or "txt"
- `table`: Enable table extraction (True/False)
- `formula`: Enable formula extraction (True/False)
- `lang`: Language code for optimization (e.g., "en", "de")

## Troubleshooting

### Common Issues

1. **Import Error**: Install dependencies
   ```bash
   uv sync
   # or
   pip install raganything nest-asyncio
   ```

2. **API Key Missing**: Set environment variable
   ```bash
   export OPENAI_API_KEY="your-key"
   ```

3. **Async Issues**: `nest-asyncio` should handle this automatically, but if issues persist, ensure it's imported and applied in the notebook.

4. **Processing Time**: First-time document processing may take several minutes. Results are cached in the output directory.

## Next Steps

### Further Optimizations
1. **Hybrid Retrieval**: Combine RAG-Anything with keyword search
2. **Re-ranking**: Add cross-encoder re-ranking for better results
3. **Query Expansion**: Expand queries with synonyms/related terms
4. **Knowledge Graph**: Add relationship extraction for complex queries

### Testing
Run the evaluation to compare:
- Original FAISSRAG performance
- RAGAnythingRAG performance
- Identify areas for further improvement

## Files Modified

1. `pyproject.toml`: Added dependencies
2. `src/ai_eval/resources/rag_template.py`: Added `RAGAnythingRAG` class
3. `src/notebooks/tum_hackathon_hybrid.ipynb`: Added RAG-Anything integration cells

## References

- [RAG-Anything GitHub](https://github.com/HKUDS/RAG-Anything)
- [RAG-Anything Paper](https://arxiv.org/abs/2510.12323)
- [MinerU Documentation](https://github.com/opendatalab/MinerU)

