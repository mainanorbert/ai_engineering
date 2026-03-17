"""
LLM answer-generation layer for the medi_llm RAG agent.

Grounding rule: the LLM is only allowed to answer from the retrieved context.
If the retrieved chunks do not contain relevant information the function returns
the standard "no information" message without hallucinating.
"""

from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

from anthropic import Anthropic
from dotenv import load_dotenv

try:
    from .config import DEFAULT_SETTINGS
    from .pipeline import search_knowledge_base
    from .schemas import RetrievalResult
except ImportError:
    from config import DEFAULT_SETTINGS
    from pipeline import search_knowledge_base
    from schemas import RetrievalResult

load_dotenv(override=True)

NO_INFO_MESSAGE = "No such information is available in the medical knowledge base."
RELEVANCE_SCORE_THRESHOLD = 0.0


SYSTEM_PROMPT = """You are a knowledgeable medical information assistant specialising in respiratory
viruses and the illnesses they cause (common cold, influenza, COVID-19, and RSV).

You answer questions STRICTLY and ONLY from the provided knowledge-base context below.

Rules you must follow:
1. Do not use any knowledge outside the provided context.
2. If the context does not contain enough information to answer the question,
   reply with exactly: "No such information is available in the medical knowledge base."
3. Do not diagnose, prescribe, or give personalised medical advice.
4. Keep answers factual, concise, and directly relevant to the question.
5. When you cite a fact, you may mention the source section.

Context from the knowledge base:
{context}
"""


# ----------------------------- Data Model -----------------------------

@dataclass(slots=True)
class AnswerResult:
    question: str
    answer: str
    sources: List[RetrievalResult]


# ----------------------------- Helpers -----------------------------

def _build_context(results: List[RetrievalResult]) -> str:
    if not results:
        return ""

    blocks = []
    for i, r in enumerate(results, start=1):
        source = r.metadata.get("source", "unknown")
        section = r.metadata.get("section_heading", "")
        blocks.append(f"[{i}] Source: {source} | Section: {section}\n{r.text}")

    return "\n\n".join(blocks)


def _has_relevant_context(results: List[RetrievalResult]) -> bool:
    if not results:
        return False

    for r in results:
        if r.rerank_score is None:
            return True
        if r.rerank_score > RELEVANCE_SCORE_THRESHOLD:
            return True

    return False


# ----------------------------- Non-streaming -----------------------------

def answer_question(
    question: str,
    top_k: int = DEFAULT_SETTINGS.default_top_k,
    llm_model: str = "claude-haiku-4-5",
    max_tokens: int = 1024,
) -> AnswerResult:

    results = search_knowledge_base(query=question, top_k=top_k)

    if not _has_relevant_context(results):
        return AnswerResult(question, NO_INFO_MESSAGE, [])

    context = _build_context(results)
    client = Anthropic()

    response = client.messages.create(
        model=llm_model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT.format(context=context),
        messages=[{"role": "user", "content": question}],
    )

    answer = "".join(
        block.text for block in response.content if hasattr(block, "text")
    ).strip()

    if not answer:
        answer = NO_INFO_MESSAGE

    return AnswerResult(question, answer, results)


# ----------------------------- Streaming (no sources) -----------------------------

def answer_question_stream(
    question: str,
    top_k: int = DEFAULT_SETTINGS.default_top_k,
    llm_model: str = "claude-haiku-4-5",
    max_tokens: int = 1024,
):

    results = search_knowledge_base(query=question, top_k=top_k)

    if not _has_relevant_context(results):
        yield NO_INFO_MESSAGE
        return

    context = _build_context(results)
    client = Anthropic()

    partial = ""

    with client.messages.stream(
        model=llm_model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT.format(context=context),
        messages=[{"role": "user", "content": question}],
    ) as stream:

        for event in stream:
            if getattr(event, "type", None) == "content_block_delta":
                delta = getattr(event, "delta", None)

                if delta and getattr(delta, "type", None) == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        partial += text
                        yield partial


# ----------------------------- Streaming (WITH sources) -----------------------------

def stream_answer_with_sources(
    question: str,
    top_k: int = DEFAULT_SETTINGS.default_top_k,
    llm_model: str = "claude-haiku-4-5",
    max_tokens: int = 1024,
) -> Generator[Tuple[str, Optional[List[RetrievalResult]]], None, None]:

    results = search_knowledge_base(query=question, top_k=top_k)

    # No relevant context → return immediately
    if not _has_relevant_context(results):
        yield NO_INFO_MESSAGE, []
        return

    context = _build_context(results)
    client = Anthropic()

    partial = ""

    with client.messages.stream(
        model=llm_model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT.format(context=context),
        messages=[{"role": "user", "content": question}],
    ) as stream:

        for event in stream:
            if getattr(event, "type", None) == "content_block_delta":
                delta = getattr(event, "delta", None)

                # Only process actual text tokens
                if delta and getattr(delta, "type", None) == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        partial += text
                        yield partial, None

    # Final fallback
    if not partial.strip():
        partial = NO_INFO_MESSAGE

    # FINAL YIELD (CRITICAL)
    yield partial, results