chat_provider = "openai"
embedding_provider = "openai"
chat_model = "gpt-4o-mini"  # Correct OpenAI model name
embedding_model = "text-embedding-3-large"  # 3072 dimensions
parser = "mineru"

# LLM settings for better structured output (entity/relationship extraction)
llm_temperature = 0.0  # Lower temperature for more deterministic, structured output
llm_max_tokens = 4000  # Ensure complete responses (prevents truncation)
llm_max_retries = 3  # Retry on failures for better reliability

# Query-specific settings (can be different from storage settings)
query_chat_provider = "openai"
query_embedding_provider = "openai"
query_chat_model = "gpt-4o-mini"  # Correct OpenAI model name
query_embedding_model = "text-embedding-3-large"  # Must match storage embedding model!
query_rag_storage_path = "/Users/qtf4195/tum-hackathon/simple_rag/rag_storage_gpt-4o_text-embedding-3-large_mineru_temp0.0_max4k_retry3"

# Output directory for parsed documents (from store_output.py)
# Used by store_rag.py to read parsed output files
output_dir = "./output_gpt-4o_text-embedding-3-large_mineru_temp0.0_max4k_retry3/raganything_processed"
