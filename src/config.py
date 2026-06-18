"""
Configuration module for AI Research Assistant.
Centralizes all environment variables, model settings, and paths.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DUCKDB_PATH = DATA_DIR / "vectors.duckdb"
PDF_PATH = PROJECT_ROOT / "intro-to-ml.pdf"

# --- API Keys ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Model Settings ---
LLM_MODEL_ID = "meta-llama/llama-3.3-70b-instruct:free"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"

# --- RAG Settings ---
CHUNK_SIZE = 500       # tokens per chunk
CHUNK_OVERLAP = 50     # overlap between chunks
TOP_K_RESULTS = 5      # number of chunks to retrieve

# --- Validation ---
def validate_config():
    """Validate that all required configuration is present."""
    errors = []
    if not OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY not set in .env")
    if not PDF_PATH.exists():
        errors.append(f"PDF not found at {PDF_PATH}")
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    # Warnings (non-fatal)
    if not TAVILY_API_KEY:
        print("[WARNING] TAVILY_API_KEY not set -- web search will use mock fallback")

    print("[OK] Configuration validated successfully")