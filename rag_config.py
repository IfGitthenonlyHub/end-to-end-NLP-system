from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

KNOWLEDGE_DIR = ROOT_DIR / "data" / "knowledgebase"
INDEX_DIR = ROOT_DIR / "data" / "index"
SYSTEM_OUTPUT_DIR = ROOT_DIR / "system_outputs"
EXPERIMENT_DIR = ROOT_DIR / "experiments"

EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"
RERANKER_MODEL = "AITeamVN/Vietnamese_Reranker"
OLLAMA_MODEL = "qwen2.5:3b-instruct-q6_K"
OLLAMA_URL = "http://localhost:11434/api/generate"

DEFAULT_CHUNK_SIZE = 650
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_RETRIEVE_TOP_K = 20
DEFAULT_RERANK_TOP_K = 3
DEFAULT_FEWSHOT_K = 3

INDEX_FILE = INDEX_DIR / "faiss.index"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"
CONFIG_FILE = INDEX_DIR / "index_config.json"
