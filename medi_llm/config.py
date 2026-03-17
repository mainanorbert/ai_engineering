from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RagSettings:
    """Store the default paths and retrieval settings for the RAG pipeline."""

    knowledge_base_path: Path = BASE_DIR / "knowledge_base"
    vector_db_path: Path = BASE_DIR / "vector_db"
    collection_name: str = "medi_llm_knowledge_base"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 900
    chunk_overlap: int = 180
    default_top_k: int = 5
    # Reranking — fetch (default_top_k × candidate_multiplier) candidates from
    # the vector DB, score every candidate against the query with a cross-encoder,
    # then keep only the best default_top_k results.
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_candidate_multiplier: int = 3
    enable_reranking: bool = True


DEFAULT_SETTINGS = RagSettings()
