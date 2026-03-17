from functools import lru_cache

from sentence_transformers import SentenceTransformer

try:
    from .config import DEFAULT_SETTINGS
except ImportError:
    from config import DEFAULT_SETTINGS


class EmbeddingClient:
    """Wrap a sentence-transformer model for document and query embeddings."""

    def __init__(self, model_name: str) -> None:
        """Initialize the embedding client with a model name."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple document texts into dense vectors."""
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed one query string into a dense vector."""
        vector = self.model.encode(query, normalize_embeddings=True)
        return vector.tolist()


@lru_cache(maxsize=2)
def get_embedding_client(model_name: str = DEFAULT_SETTINGS.embedding_model_name) -> EmbeddingClient:
    """Return a cached embedding client for the requested model."""
    return EmbeddingClient(model_name=model_name)
