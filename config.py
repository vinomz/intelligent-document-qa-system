# config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    # FAST API Configs
    app_name: str = "RAG AI Assistant"
    app_description: str = "RAG service powered by LangChain, ChromaDB, and Gemini."
    app_version: str = "0.1.0"

    # Defaults
    DEFAULT_DOCS_PATH: str = "docs"
    DEFAULT_CHROMA_PATH: str = "vector_store/chroma_db"
    DEFAULT_LOGLEVEL: str = "DEBUG"

    GOOGLE_API_KEY: str

    # Vector Store Configurations
    EMBEDDING_MODEL: str = "models/text-embedding-004"

    # Streamlit Configurations
    AI_RAG_API_URL: str = "http://ai-rag-api:5050/"

    #Environment Variables
    model_config  = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY