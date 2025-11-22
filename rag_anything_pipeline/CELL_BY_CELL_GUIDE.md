# Cell-by-Cell Execution Guide

Both Python files now support cell-by-cell execution in Jupyter/VS Code using `# %%` markers.

## How to Use

### In VS Code:
1. Open the `.py` file
2. Click "Run Cell" above each `# %%` marker
3. Or use `Shift+Enter` to run the current cell

### In Jupyter:
1. Open the `.py` file in Jupyter
2. Each `# %%` creates a new cell
3. Run cells individually

## store_documents.py - Cell Structure

### Cell 1: Imports and Setup
- All imports
- nest_asyncio setup

### Cell 2: Initialize Models
- Function to initialize chat and embedding models

### Cell 3: Create RAG-Anything Configuration
- Function to create optimized configuration

### Cell 4: Initialize RAG-Anything
- Function to initialize RAG-Anything with model functions

### Cell 5: Process Document (Async)
- Async function to process documents

### Cell 6: Main Storage Function
- Complete storage workflow with caching

### Cell 7: Command-Line Interface
- CLI argument parsing

### Cell 8: Execute Main
- Script execution entry point

### Cell 9: Example Usage (Interactive)
- Example code to run interactively (commented out)

## query_documents.py - Cell Structure

### Cell 1: Imports and Setup
- All imports
- nest_asyncio setup

### Cell 2: Initialize Models
- Function to initialize chat and embedding models

### Cell 3: Create RAG-Anything Configuration
- Function to create optimized configuration

### Cell 4: Initialize RAG-Anything Instance
- Function to initialize RAG-Anything

### Cell 5: Verify Storage
- Function to check if documents are processed

### Cell 6: Create RAG Instance for Querying
- Function to create RAG instance

### Cell 7: Query Document Function
- Function to query and get answers

### Cell 8: Interactive Query Mode
- Interactive mode for multiple queries

### Cell 9: Command-Line Interface
- CLI argument parsing

### Cell 10: Execute Main
- Script execution entry point

### Cell 11: Example Usage (Interactive)
- Example code to run interactively (commented out)

## Recommended Execution Flow

### For store_documents.py:

**Option 1: Run all cells in order**
1. Run Cell 1 (Imports)
2. Run Cell 2-5 (Functions)
3. Run Cell 9 (Example) - uncomment and modify

**Option 2: Use as script**
- Run Cell 8 (Execute Main) or run from command line

### For query_documents.py:

**Option 1: Run all cells in order**
1. Run Cell 1 (Imports)
2. Run Cell 2-7 (Functions)
3. Run Cell 11 (Example) - uncomment and modify

**Option 2: Use as script**
- Run Cell 10 (Execute Main) or run from command line

## Interactive Example

### store_documents.py:
```python
# In Cell 9, uncomment and modify:
filename = "Allplan_2020_Manual.pdf"
force_reprocess = False
store_document(filename=filename, force_reprocess=force_reprocess)
```

### query_documents.py:
```python
# In Cell 11, uncomment and modify:
# Step 1: Verify storage
output_dir = "./output/raganything_processed"
if not verify_storage(output_dir):
    print("Please run store_documents.py first!")

# Step 2: Initialize
chat_model, embedding_model = initialize_models()
config = create_rag_anything_config()
rag_instance = create_rag_instance(chat_model, embedding_model, config)

# Step 3: Query
query = "How do I control pattern height in Allplan?"
answer, docs = query_document(query, rag_instance, show_context=True)
```

## Benefits

- ✅ Run code step by step
- ✅ Test individual functions
- ✅ Debug easily
- ✅ Still works as regular Python scripts
- ✅ Clear separation of concerns


