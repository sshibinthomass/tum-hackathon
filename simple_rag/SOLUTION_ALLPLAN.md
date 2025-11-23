# ✅ Solution: Optimized RAG for Allplan Manual

I have optimized the RAG pipeline to process the `Allplan_2020_Manual.pdf` and verify it against the QA dataset.

## 🔧 Improvements Made

### 1. Fixed `store_improved.py`
- **Dependency Fix**: Removed unused `ChatVertexAI` import that was causing crashes.
- **Path Fix**: Added logic to ensure `mineru` (Magic-PDF) is found in the system PATH, resolving `RuntimeError: mineru command not found`.
- **Robustness**: Added file existence checks and debug logging.

### 2. Enhanced `query_improved.py`
- **QA Testing**: Added a `--test-qa` flag to automatically run questions from `generated_qa_data_tum.json`.
- **Auto-Detection**: Added logic to automatically find the latest storage directory if the configured one is missing.
- **Command Line Args**: Added support for passing questions directly via command line.

### 3. Configuration
- Updated `config.py` to point to the correct storage path for the Allplan manual.

## 🚀 How to Run

### Step 1: Process the Document (Long Running)
The manual is large (300+ pages) and we are using **full multimodal extraction** (tables, formulas, images) for best quality. This process takes about **20-30 minutes**.

```bash
python store_improved.py
```
*Note: If this process is interrupted, it must be restarted.*

### Step 2: Verify with QA Data
Once storage is complete, run the QA test to verify retrieval accuracy:

```bash
python query_improved.py --test-qa --num-questions 5
```

### Step 3: Ask Specific Questions
You can also ask specific questions:

```bash
python query_improved.py "How can I control the height of a pattern element?"
```

## 📊 Expected Results
The system uses `mode="naive"` with **robust retrieval** (direct vector search) which is optimized for the specific, procedural questions found in the Allplan manual QA dataset.
