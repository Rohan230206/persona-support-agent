import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Load environment variables
dotenv_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=dotenv_path)
load_dotenv()  # Fallback to current working directory load

# Application Constants
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# Ensure crucial directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Configurable Parameters
SIMILARITY_THRESHOLD = 0.45
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"
LOG_FILE = PROJECT_ROOT / "app.log"

def validate_environment() -> bool:
    """
    Validates that necessary configuration options are set.
    Returns:
        bool: True if environment is valid, False otherwise.
    """
    if not GEMINI_API_KEY:
        # We will not crash the import, but check at app startup
        return False
    return True
