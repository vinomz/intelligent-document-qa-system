# config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    #FAST API Configs
    app_name: str = "RAG AI Assistant"
    app_description: str = "RAG service powered by LangChain, ChromaDB, and Gemini."
    app_version: str = "0.1.0"

    #Defaults
    DEFAULT_DOCS_PATH: str = "./internal_docs"
    DEFAULT_CHROMA_PATH: str = "chroma_db"

    GOOGLE_API_KEY: str

    #Environment Variables
    model_config  = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()