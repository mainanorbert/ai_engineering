from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class KnowledgeDocument:
    """Represent one markdown source document in the medical knowledge base."""

    document_id: str
    source: str
    category: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentChunk:
    """Represent one chunk derived from a knowledge-base document."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    """Represent one retrieved chunk returned from the vector database."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # L2 distance from the vector DB query (lower = closer)
    distance: float | None = None
    # Cross-encoder relevance score assigned during reranking (higher = more relevant)
    rerank_score: float | None = None
