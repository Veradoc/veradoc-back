import os

from app.config import settings

# LLM configurations
LLM_MODEL = "hf.co/unsloth/gemma-3-1b-it-GGUF:latest"
EMBEDDING_MODEL = "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:latest"
#EMBEDDING_MODEL = "nomic-embed-text:v1.5"

# Google configurations
GOOGLE_CLIENT_ID = "203872501539-5utooc4ptpso11301ruqllthq17nb3kb.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"

# LanceDB configurations
os.environ["AWS_ACCESS_KEY_ID"] = settings.minio_access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = settings.minio_secret_key
os.environ["AWS_ENDPOINT"] = settings.minio_endpoint
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["ALLOW_HTTP"] = "True"

# RAG configurations
DOCS_TABLE = "docs"
EMBEDDINGS_DIM = 768
METADATA_PREFIX = "metadata"
EMBEDDING_DOCUMENT_PREFIX = "search_document"
EMBEDDING_QUERY_PREFIX = "search_query"
BUCKET_NAME = "custom-corpus"

# RAG prompt template
RAG_PROMPT = """
DOCUMENT:
{documents}

QUESTION:
{user_question}

INSTRUCTIONS:
Answer in detail the user's QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT. Do not use sentence like "The document states" citing the document.
If the DOCUMENT doesn't contain the facts to answer the QUESTION only Respond with "Sorry! I Don't know"
"""
