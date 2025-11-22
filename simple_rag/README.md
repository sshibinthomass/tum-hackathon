# Simple RAG-Anything

A simplified, minimal implementation of RAG-Anything for document storage and querying.

## Files

- **store.py** - Process and store PDF documents (run once per document)
- **query.py** - Query stored documents and get answers

## Usage

### 1. Store a Document

```bash
python store.py attention.pdf
```

This processes the PDF and stores it in `./rag_storage/`.

### 2. Query Documents

```bash
python query.py "Your question here"
```

Example:
```bash
python query.py "How do I use patterns in Allplan?"
```

The script will return the answer directly.

## Requirements

- Documents must be in the `data/` directory (as configured in `global_config`)
- Run `store.py` first before querying
- Uses the same model configuration as the main project (OpenAI or Ollama)

