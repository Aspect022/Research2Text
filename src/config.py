"""
Research2Code-GenAI — Central Configuration
============================================
Edit THIS file to change models, timeouts, paths, and pipeline settings.
All agents and modules import from here.
"""

from pathlib import Path


# ═══════════════════════════════════════════════════
#  Project Paths
# ═══════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
RAW_TEXT_DIR = DATA_DIR / "raw_texts"
CHROMA_DIR = DATA_DIR / "chroma_db"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ═══════════════════════════════════════════════════
#  Ollama / LLM Settings — Role-Based Model Routing
# ═══════════════════════════════════════════════════

# Default model for RAG Q&A, general chat
DEFAULT_OLLAMA_MODEL = "gpt-oss:120b-cloud"

# REASONER chain — used for method extraction, analysis, Q&A
# Prefers general-purpose models with strong reasoning
REASONER_MODEL_CHAIN = [
    "deepseek-v3.1:671b-cloud",
    "gpt-oss:120b-cloud",
    "glm-4.6:cloud",
    "minimax-m2:cloud",
    "qwen3-coder:480b-cloud",
]

# CODER chain — used for code generation, self-healing, code review
# Prefers coding-specialized models for better code output
CODER_MODEL_CHAIN = [
    "qwen3-coder:480b-cloud",
    "deepseek-v3.1:671b-cloud",
    "gpt-oss:120b-cloud",
    "glm-4.6:cloud",
]

# Backward-compatible alias (used by older imports)
MODEL_CHAIN = CODER_MODEL_CHAIN

# Reference knowledge file for code generation prompts
CODEGEN_RULES_PATH = PROJECT_ROOT / "data" / "codegen_rules.md"

# LLM temperature settings
EXTRACTION_TEMPERATURE = 0.1   # Low = factual/precise (method extraction)
CODE_GEN_TEMPERATURE = 0.2     # Slightly higher for code creativity
SELF_HEAL_TEMPERATURE = 0.1    # Low = focused fixing


# ═══════════════════════════════════════════════════
#  RAG Settings
# ═══════════════════════════════════════════════════

COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
MAX_CONTEXT_CHARS = 4000


# ═══════════════════════════════════════════════════
#  Chunking Settings
# ═══════════════════════════════════════════════════

CHUNK_SIZE_WORDS = 700
CHUNK_OVERLAP_WORDS = 100
INDEX_BATCH_SIZE = 64


# ═══════════════════════════════════════════════════
#  Pipeline / Agent Settings
# ═══════════════════════════════════════════════════

# Validator Agent
MAX_HEAL_ATTEMPTS = 3          # How many times to retry failing code
EXECUTION_TIMEOUT = 90         # Seconds before killing sandbox execution
DOCKER_BUILD_TIMEOUT = 120     # Seconds for Docker image build
DOCKER_MEMORY_LIMIT = "512m"
DOCKER_CPU_LIMIT = "1"

# Ingest Agent
OLMOCR_TIMEOUT = 300           # Seconds for olmOCR CLI pipeline
MAX_TEXT_LENGTH_FOR_LLM = 6000 # Max chars sent to LLM for extraction

# Confidence scoring
LOW_CONFIDENCE_THRESHOLD = 0.5 # Fields below this are flagged


# ═══════════════════════════════════════════════════
#  Streamlit UI
# ═══════════════════════════════════════════════════

APP_TITLE = "Research2Code-GenAI"
APP_ICON = "🧬"
APP_SUBTITLE = "Autonomous Scientific Paper → Executable Code with Conformal Prediction"
