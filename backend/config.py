"""
Configuration management for the AI Study Buddy application.
Loads environment variables and provides typed configuration objects.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional

# project root directory
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Vector Database Configuration
    vector_db_type: str = "chromadb"
    chromadb_path: str = str(PROJECT_ROOT / "data" / "chromadb")
    
    # Application Settings
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    upload_dir: str = str(PROJECT_ROOT / "data" / "raw")
    processed_dir: str = str(PROJECT_ROOT / "data" / "processed")
    courses_dir: str = str(PROJECT_ROOT / "courses")
    max_file_size: int = 50  # MB
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5
    temperature: float = 0.4
    
    # Evaluation
    enable_evaluation: bool = True
    enable_user_feedback: bool = True
    
    @field_validator('chromadb_path', 'upload_dir', 'processed_dir', mode='before')
    @classmethod
    def resolve_path(cls, v):
        """Convert relative paths to absolute paths based on PROJECT_ROOT."""
        if v and not Path(v).is_absolute():
            return str(PROJECT_ROOT / v.lstrip('./'))
        return v
    
    class Config:
        env_file = str(ENV_FILE)
        env_file_encoding = 'utf-8'
        case_sensitive = False
        extra = "ignore"

# Global settings instance
settings = Settings()
