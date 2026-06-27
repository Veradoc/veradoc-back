import os
import pynvml
import psutil

from app.config import settings

# Default Model configurations
DEF_LLM_MODEL = "hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:latest"         # Generative model
DEF_EMBEDDING_MODEL = "hf.co/Ralriki/multilingual-e5-large-instruct-GGUF" # Embedding model
DEF_RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"          # CrossEncoder model
DEF_TOP_RERANKER_VECTORS = 20
DEF_TOP_VECTORS = 5

# Google configurations
GOOGLE_CLIENT_ID = "203872501539-5utooc4ptpso11301ruqllthq17nb3kb.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"

# LanceDB configurations
os.environ["AWS_ACCESS_KEY_ID"] = settings.minio_access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = settings.minio_secret_key
os.environ["AWS_ENDPOINT"] = settings.minio_endpoint
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["ALLOW_HTTP"] = "True"

# LLM configurations from your computer hardware architecture
def get_model_gpu_options() -> dict:
    try:        
        pynvml.nvmlInit()

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_mb = mem_info.total / 1024 / 1024

        if vram_mb >= 8000:       # 8GB+ a full GPU, large context window (num_ctx  >  system_prompt + history + retrieved_chunks(vector database) + question(user promt) + expected_response)
            return {"num_gpu": 99, "num_ctx": 32768}
        elif vram_mb >= 6000:     # 6GB  a full GPU, moderate context window
            return {"num_gpu": 99, "num_ctx": 16384}
        elif vram_mb >= 4000:     # 4GB  a full GPU, safe context window
            return {"num_gpu": 99, "num_ctx": 8192}
        else:                     # CPU fallback
            return {"num_gpu": 0, "num_ctx": 2048}
    except Exception:
        return {"num_gpu": 0, "num_ctx": 2048}

settings.gpu_options = get_model_gpu_options()
print(f"GPU Options: {settings.gpu_options}")

def get_top_k_chunks() -> int:
    # In your LanceDB retrieval logic you can do something like:
    if settings.gpu_options["num_ctx"] >= 16384:
        top_k_chunks = 8  # Excellent hardware, we add a lot of context
    elif settings.gpu_options["num_ctx"] >= 8192:
        top_k_chunks = 5  # Safe context
    else:
        top_k_chunks = 2  # CPU mode, we reduce chunks to fit in 2048 tokens

    return top_k_chunks

settings.top_k_chunks = get_top_k_chunks()
print(f"Top K Chunks: {settings.top_k_chunks}")

def get_optimal_threads() -> int:
    try:
        
        # 1. We attempt to obtain only the PHYSICAL cores (without Hyper-Threading).
        physical_cores = psutil.cpu_count(logical=False)
        
        if physical_cores and physical_cores > 0:
            # We reserve 1 core so that the operating system and the backend (FastAPI) remain responsive.
            optimal_threads = max(1, physical_cores - 1)

            return optimal_threads
            
    except Exception:
        pass
    
    # 2. Fallback in case psutil fails: we use the OS's logical threads.
    try:
        logical_cores = os.cpu_count() or 4
        
        # If we rely on logical cores, dividing by 2 is a safe bet against SMT.
        return max(1, logical_cores // 2)
    except Exception:
        return 4 # Último recurso seguro

settings.num_threads = get_optimal_threads()
print(f"Num Threads: {settings.num_threads}")

# RAG configurations
BUCKET_NAME = settings.minio_knowledge_base
DOCS_TABLE = settings.minio_knowledge_vectordb_table
METADATA_PREFIX = settings.minio_knowledge_base_metadata

# Embedding configurations
EMBEDDINGS_DIM = 1024

# RAG user prompt template
RAG_USER_PROMPT = """
DOCUMENT:
{documents}

QUESTION:
{user_question}

INSTRUCTIONS:
Answer in detail the user's QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT. Do not use sentence like "The document states" citing the document.
If the DOCUMENT doesn't contain the facts to answer the QUESTION only Respond with "Sorry! I Don't know"
"""
