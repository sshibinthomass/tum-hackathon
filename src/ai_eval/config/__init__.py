import os
# Fix for FAISS on Mac causing OpenMP conflict crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
