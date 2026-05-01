import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Anthropic
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    # AWS S3 — optional. Leave blank to skip S3 archival.
    AWS_BUCKET      = os.getenv("AWS_BUCKET", "")
    AWS_ACCESS_KEY  = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_KEY  = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION      = os.getenv("AWS_REGION", "us-east-1")

    # Local storage
    UPLOAD_FOLDER   = os.getenv("UPLOAD_FOLDER", "./uploads")
    CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")

    # RAG settings
    CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "6"))

    # LLM
    MODEL           = os.getenv("MODEL", "claude-sonnet-4-5")
    MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "1500"))
