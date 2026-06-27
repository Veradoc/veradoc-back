import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine which context file to load (default to .env if nothing is set)
env_suffix = os.getenv("APP_ENV", "")
env_file = f".env.{env_suffix}" if env_suffix else ".env"

# These will look for environment variables of the same name (case-insensitive)
class Settings(BaseSettings):    
    # minio configurations
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_knowledge_base: str = "custom-corpus"
    minio_knowledge_base_metadata: str = "metadata"
    minio_knowledge_vectordb: str = "warehouse"
    minio_knowledge_vectordb_table: str = "docs"
    # embedding model configurations
    top_k_chunks: int | None = None
    top_rerank_chunks: int | None = None
    # infrastructure configurations
    gpu_options: dict | None = None
    num_threads: int | None = None

    # ollama configurations
    ollama_host: str = "http://localhost:11434"
    
    # This tells Pydantic to read from a .env file
    model_config = SettingsConfigDict(env_file=env_file)

# Create a singleton instance
settings = Settings()