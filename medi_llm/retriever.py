"""
Retrieval layer for the medi_llm RAG pipeline.

Two public functions:
  retrieve_chunks        — plain ANN lookup, returns nearest neighbours sorted by
                           vector-DB distance (no reranking).
  retrieve_and_rerank    — over-fetches (top_k × candidate_multiplier) from the
                           vector DB, then scores every candidate with a cross-
                           encoder, and trims down to top_k.  This matches the
                           reranking pattern from week5-RAG but runs the ranking
                           model locally so no LLM API call is required here.
"""

from pathlib import Path
from typing import List

try:
    from .config import DEFAULT_SETTINGS
    from .embeddings import get_embedding_client
    from .reranker import get_reranker
    from .schemas import RetrievalResult
    from .vector_store import get_collection
except ImportError:
    from config import DEFAULT_SETTINGS
    from embeddings import get_embedding_client
    from reranker import get_reranker
    from schemas import RetrievalResult
    from vector_store import get_collection


def retrieve_chunks(
    query: str,
    top_k: int = DEFAULT_SETTINGS.default_top_k,
    collection_name: str = DEFAULT_SETTINGS.collection_name,
    vector_db_path: Path | None = None,
    embedding_model_name: str = DEFAULT_SETTINGS.embedding_model_name,
) -> List[RetrievalResult]:
    """Retrieve the nearest knowledge-base chunks using ANN distance only.

    Results are ordered by ascending L2 distance (most similar first).
    No reranking is applied — use retrieve_and_rerank for higher-accuracy retrieval.
    """
    embedder = get_embedding_client(model_name=embedding_model_name)
    query_vector = embedder.embed_query(query)
    collection = get_collection(collection_name=collection_name, vector_db_path=vector_db_path)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    return [
        RetrievalResult(
            chunk_id=chunk_id,
            document_id=str(metadata.get("document_id", "")),
            text=document_text,
            metadata=metadata,
            distance=distance,
        )
        for chunk_id, document_text, metadata, distance in zip(ids, documents, metadatas, distances)
    ]


def retrieve_and_rerank(
    query: str,
    top_k: int = DEFAULT_SETTINGS.default_top_k,
    candidate_multiplier: int = DEFAULT_SETTINGS.retrieval_candidate_multiplier,
    collection_name: str = DEFAULT_SETTINGS.collection_name,
    vector_db_path: Path | None = None,
    embedding_model_name: str = DEFAULT_SETTINGS.embedding_model_name,
    reranker_model_name: str = DEFAULT_SETTINGS.reranker_model_name,
) -> List[RetrievalResult]:
    """Retrieve candidate chunks then rerank by cross-encoder relevance score.

    The ANN stage fetches top_k × candidate_multiplier candidates so the
    reranker has enough signal to pick the truly best passages.  The final
    list contains at most top_k results, ordered by descending rerank_score.
    """
    candidate_count = top_k * candidate_multiplier
    candidates = retrieve_chunks(
        query=query,
        top_k=candidate_count,
        collection_name=collection_name,
        vector_db_path=vector_db_path,
        embedding_model_name=embedding_model_name,
    )

    if not candidates:
        return []

    reranker = get_reranker(model_name=reranker_model_name)
    return reranker.rerank(query=query, candidates=candidates, top_k=top_k)


def format_retrieval_results(results: List[RetrievalResult], show_scores: bool = True) -> str:
    """Format retrieval results into a readable multi-document preview.

    When show_scores is True, the distance and rerank_score are printed for each
    result so it is easy to see how retrieval and reranking compare.
    """
    if not results:
        return "(no results returned)"

    blocks: List[str] = []

    for index, result in enumerate(results, start=1):
        source = result.metadata.get("source", "unknown")
        section = result.metadata.get("section_heading", "unknown")
        lines = [
            f"Result {index}",
            f"Source   : {source}",
            f"Section  : {section}",
        ]
        if show_scores:
            dist_text = f"{result.distance:.4f}" if result.distance is not None else "n/a"
            score_text = f"{result.rerank_score:.4f}" if result.rerank_score is not None else "n/a"
            lines.append(f"Distance : {dist_text}  |  Rerank score : {score_text}")
        lines.append("")
        lines.append(result.text)
        blocks.append("\n".join(lines))

    separator = "\n\n" + ("-" * 80) + "\n\n"
    return "\n\n" + separator.join(blocks)
