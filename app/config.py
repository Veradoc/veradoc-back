import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine which file to load (default to .env if nothing is set)
env_suffix = os.getenv("APP_ENV", "")
env_file = f".env.{env_suffix}" if env_suffix else ".env"

class Settings(BaseSettings):
    # These will look for environment variables of the same name (case-insensitive)
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_knowledge_base: str = "custom-corpus"
    minio_knowledge_metadata: str = "warehouse"
    ollama_host: str = "http://localhost:11434"
    
    # This tells Pydantic to read from a .env file
    model_config = SettingsConfigDict(env_file=env_file)

# Create a singleton instance
settings = Settings()