from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from chromadb.config import Settings

try:
    from .config import DEFAULT_SETTINGS
    from .embeddings import get_embedding_client
    from .schemas import DocumentChunk
except ImportError:
    from config import DEFAULT_SETTINGS
    from embeddings import get_embedding_client
    from schemas import DocumentChunk


def ensure_vector_db_directory(vector_db_path: Path) -> None:
    """Create the vector database directory if it does not already exist."""
    vector_db_path.mkdir(parents=True, exist_ok=True)


def normalize_metadata_value(value: Any) -> str | int | float | bool:
    """Convert metadata values into Chroma-compatible primitive types."""
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    return str(value)


def build_chroma_metadata(chunk: DocumentChunk) -> dict[str, str | int | float | bool]:
    """Convert chunk metadata into a Chroma-safe metadata dictionary."""
    metadata: dict[str, str | int | float | bool] = {}
    for key, value in chunk.metadata.items():
        metadata[key] = normalize_metadata_value(value)
    metadata["chunk_id"] = chunk.chunk_id
    metadata["document_id"] = chunk.document_id
    return metadata


def get_persistent_client(vector_db_path: Path | None = None) -> PersistentClient:
    """Return a persistent Chroma client for the configured vector database path."""
    target_path = vector_db_path or DEFAULT_SETTINGS.vector_db_path
    ensure_vector_db_directory(target_path)
    return PersistentClient(
        path=str(target_path),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(
    collection_name: str = DEFAULT_SETTINGS.collection_name,
    vector_db_path: Path | None = None,
):
    """Return the named Chroma collection from the persistent client."""
    client = get_persistent_client(vector_db_path=vector_db_path)
    return client.get_or_create_collection(name=collection_name)


def reset_collection(
    collection_name: str = DEFAULT_SETTINGS.collection_name,
    vector_db_path: Path | None = None,
) -> None:
    """Delete the collection if it exists so the index can be rebuilt cleanly."""
    client = get_persistent_client(vector_db_path=vector_db_path)
    existing_names = {collection.name for collection in client.list_collections()}
    if collection_name in existing_names:
        client.delete_collection(name=collection_name)


def index_chunks(
    chunks: list[DocumentChunk],
    collection_name: str = DEFAULT_SETTINGS.collection_name,
    vector_db_path: Path | None = None,
    embedding_model_name: str = DEFAULT_SETTINGS.embedding_model_name,
    reset_existing: bool = True,
) -> int:
    """Embed document chunks and store them in the persistent Chroma collection."""
    if not chunks:
        raise ValueError("Cannot index an empty chunk list.")

    if reset_existing:
        reset_collection(collection_name=collection_name, vector_db_path=vector_db_path)

    collection = get_collection(collection_name=collection_name, vector_db_path=vector_db_path)
    embedder = get_embedding_client(model_name=embedding_model_name)
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_documents(texts)
    metadatas = [build_chroma_metadata(chunk) for chunk in chunks]
    ids = [chunk.chunk_id for chunk in chunks]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return collection.count()


def get_collection_count(
    collection_name: str = DEFAULT_SETTINGS.collection_name,
    vector_db_path: Path | None = None,
) -> int:
    """Return the number of indexed chunks stored in the collection."""
    collection = get_collection(collection_name=collection_name, vector_db_path=vector_db_path)
    return collection.count()
