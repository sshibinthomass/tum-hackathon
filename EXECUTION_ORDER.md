# RAG-Anything Execution Order - Quick Reference

## ⚠️ IMPORTANT: Run Cells in This Exact Order

### Required Execution Order:

1. **Cell 1-8**: Basic setup (load data, get QA data)
2. **Cell 9**: Initialize `chat_model` and `embedding_model` ⚠️ REQUIRED
3. **Cell 11**: Step 1 - Setup & Configuration (creates `config`)
4. **Cell 13**: Step 2 - Initialize RAG-Anything ⚠️ REQUIRED BEFORE STEP 3
5. **Cell 15**: Step 3 - Process & Store Documents
6. **Cell 16-17**: Step 4 - Query & Retrieve
7. **Cell 20**: Step 5 - Use with Evaluation Framework

## Common Error: "rag_anything not initialized"

**If you see this error**, it means you're trying to run Step 3 before Step 2.

### Solution:

1. Go to **Cell 13** (Step 2: Initialize RAG-Anything)
2. Run that cell first
3. Then come back to Step 3

### Quick Check:

Before running Step 3, verify:

```python
# Run this in a cell to check:
'rag_anything' in globals() and rag_anything is not None
```

If this returns `False`, you need to run Step 2 first!

## Cell Number Reference

| Step       | Cell Number | What It Does                | Prerequisites        |
| ---------- | ----------- | --------------------------- | -------------------- |
| Setup      | 9           | Initialize models           | None                 |
| Step 1     | 11          | Create config               | Models from cell 9   |
| **Step 2** | **13**      | **Initialize RAG-Anything** | **Cell 9, Cell 11**  |
| **Step 3** | **15**      | **Process documents**       | **Cell 13 (Step 2)** |
| Step 4     | 16-17       | Query & Retrieve            | Cell 15 (Step 3)     |
| Step 5     | 20          | Evaluation setup            | Cell 15 (Step 3)     |

## Troubleshooting

### Error: "rag_anything not initialized"

- **Cause**: Step 3 run before Step 2
- **Fix**: Run Cell 13 (Step 2) first

### Error: "chat_model not initialized"

- **Cause**: Step 2 run before models are set up
- **Fix**: Run Cell 9 first

### Error: "config not created"

- **Cause**: Step 2 run before Step 1
- **Fix**: Run Cell 11 (Step 1) first

## Visual Execution Flow

```
Cell 9 (Models)
    ↓
Cell 11 (Step 1: Config)
    ↓
Cell 13 (Step 2: Initialize) ← YOU MUST RUN THIS
    ↓
Cell 15 (Step 3: Process)    ← BEFORE THIS
    ↓
Cell 16-17 (Step 4: Query)
    ↓
Cell 20 (Step 5: Evaluation)
```
