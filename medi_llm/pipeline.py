"""
High-level pipeline orchestration for medi_llm.

  build_knowledge_base      — load docs → chunk → embed → persist to ChromaDB.
  search_knowledge_base     — rerank-enabled retrieval (default, higher accuracy).
  search_knowledge_base_raw — plain ANN retrieval without reranking (fast baseline).
"""

from dataclasses import dataclass
from pathlib import Path

try:
    from .config import DEFAULT_SETTINGS
    from .document_loader import load_knowledge_base_documents
    from .retriever import retrieve_and_rerank, retrieve_chunks
    from .schemas import DocumentChunk, KnowledgeDocument, RetrievalResult
    from .text_chunker import chunk_documents
    from .vector_store import index_chunks
except ImportError:
    from config import DEFAULT_SETTINGS
    from document_loader import load_knowledge_base_documents
    from retriever import retrieve_and_rerank, retrieve_chunks
    from schemas import DocumentChunk, KnowledgeDocument, RetrievalResult
    from text_chunker import chunk_documents
    from vector_store import index_chunks


@dataclass(slots=True)
class BuildArtifacts:
    """Store the output of a full knowledge-base indexing run."""

    documents: list[KnowledgeDocument]
    chunks: list[DocumentChunk]
    indexed_chunk_count: int


def build_knowledge_base(
    knowledge_base_path: Path | None = None,
    vector_db_path: Path | None = None,
    collection_name: str = DEFAULT_SETTINGS.collection_name,
    embedding_model_name: str = DEFAULT_SETTINGS.embedding_model_name,
    chunk_size: int = DEFAULT_SETTINGS.chunk_size,
    chunk_overlap: int = DEFAULT_SETTINGS.chunk_overlap,
    reset_existing: bool = True,
) -> BuildArtifacts:
    """Load, chunk, embed, and index the medical knowledge base into ChromaDB."""
    documents = load_knowledge_base_documents(knowledge_base_path=knowledge_base_path)
    chunks = chunk_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    indexed_chunk_count = index_chunks(
        chunks=chunks,
        collection_name=collection_name,
        vector_db_path=vector_db_path,
        embedding_model_name=embedding_model_name,
        reset_existing=reset_existing,
    )
    return BuildArtifacts(
        documents=documents,
        chunks=chunks,
        indexed_chunk_count=indexed_chunk_count,
    )


def search_knowledge_base(
    query: str,
    top_k: int = DEFAULT_SETTINGS.default_top_k,
    candidate_multiplier: int = DEFAULT_SETTINGS.retrieval_candidate_multiplier,
    vector_db_path: Path | None = None,
    collection_name: str = DEFAULT_SETTINGS.collection_name,
    embedding_model_name: str = DEFAULT_SETTINGS.embedding_model_name,
    reranker_model_name: str = DEFAULT_SETTINGS.reranker_model_name,
    enable_reranking: bool = DEFAULT_SETTINGS.enable_reranking,
) -> list[RetrievalResult]:
    """Search the medical knowledge base with optional cross-encoder reranking.

    When enable_reranking is True (the default) the function fetches
    top_k × candidate_multiplier candidates from ChromaDB and reranks them
    using a cross-encoder before returning the best top_k results.
    """
    if enable_reranking:
        return retrieve_and_rerank(
            query=query,
            top_k=top_k,
            candidate_multiplier=candidate_multiplier,
            collection_name=collection_name,
            vector_db_path=vector_db_path,
            embedding_model_name=embedding_model_name,
            reranker_model_name=reranker_model_name,
        )

    return retrieve_chunks(
        query=query,
        top_k=top_k,
        collection_name=collection_name,
        vector_db_path=vector_db_path,
        embedding_model_name=embedding_model_name,
    )


def search_knowledge_base_raw(
    query: str,
    top_k: int = DEFAULT_SETTINGS.default_top_k,
    vector_db_path: Path | None = None,
    collection_name: str = DEFAULT_SETTINGS.collection_name,
    embedding_model_name: str = DEFAULT_SETTINGS.embedding_model_name,
) -> list[RetrievalResult]:
    """Search the knowledge base using ANN distance only, with no reranking."""
    return retrieve_chunks(
        query=query,
        top_k=top_k,
        collection_name=collection_name,
        vector_db_path=vector_db_path,
        embedding_model_name=embedding_model_name,
    )
