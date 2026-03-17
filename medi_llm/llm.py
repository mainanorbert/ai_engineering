"""
LLM answer-generation layer for the medi_llm RAG agent.

Behavior:
    1. Prefer knowledge-base grounded answers when relevant context is available.
    2. If KB context is missing/insufficient, fall back to general knowledge.
    3. Clearly label fallback answers as general knowledge (not from KB).
"""

from dataclasses import dataclass
from typing import Generator, List

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
GENERAL_KNOWLEDGE_PREFIX = "General medical knowledge (not from knowledge base):"
RELEVANCE_SCORE_THRESHOLD = 0.0


KB_SYSTEM_PROMPT = """You are a knowledgeable medical information assistant specialising in respiratory
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

GENERAL_SYSTEM_PROMPT = """You are a knowledgeable medical information assistant specialising in respiratory
viruses and the illnesses they cause (common cold, influenza, COVID-19, and RSV).

You are answering from your general medical knowledge, not from a provided knowledge base.

Rules you must follow:
1. Be factual, concise, and directly relevant.
2. Do not diagnose, prescribe, or give personalised medical advice.
3. If uncertain, say so briefly.
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


def _extract_answer_text(response) -> str:
    return "".join(
        block.text for block in response.content if hasattr(block, "text")
    ).strip()


def _answer_from_general_knowledge(
    client: Anthropic,
    question: str,
    llm_model: str,
    max_tokens: int,
) -> str:
    response = client.messages.create(
        model=llm_model,
        max_tokens=max_tokens,
        system=GENERAL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    text = _extract_answer_text(response)
    if not text:
        text = "I do not have enough confidence to answer from general knowledge right now."
    return f"{GENERAL_KNOWLEDGE_PREFIX}\n\n{text}"


# ----------------------------- Non-streaming -----------------------------

def answer_question(
    question: str,
    top_k: int = DEFAULT_SETTINGS.default_top_k,
    llm_model: str = "claude-haiku-4-5",
    max_tokens: int = 1024,
) -> AnswerResult:

    results = search_knowledge_base(query=question, top_k=top_k)
    client = Anthropic()

    if not _has_relevant_context(results):
        fallback_answer = _answer_from_general_knowledge(
            client=client,
            question=question,
            llm_model=llm_model,
            max_tokens=max_tokens,
        )
        return AnswerResult(question, fallback_answer, [])

    context = _build_context(results)

    response = client.messages.create(
        model=llm_model,
        max_tokens=max_tokens,
        system=KB_SYSTEM_PROMPT.format(context=context),
        messages=[{"role": "user", "content": question}],
    )

    answer = _extract_answer_text(response)

    if not answer or answer == NO_INFO_MESSAGE:
        answer = _answer_from_general_knowledge(
            client=client,
            question=question,
            llm_model=llm_model,
            max_tokens=max_tokens,
        )

    return AnswerResult(question, answer, results)


# ----------------------------- Streaming (no sources) -----------------------------

def answer_question_stream(
    question: str,
    top_k: int = DEFAULT_SETTINGS.default_top_k,
    llm_model: str = "claude-haiku-4-5",
    max_tokens: int = 1024,
):
    """Stream the LLM answer token by token."""
    results = search_knowledge_base(query=question, top_k=top_k)
    client = Anthropic()

    if not _has_relevant_context(results):
        fallback_answer = _answer_from_general_knowledge(
            client=client,
            question=question,
            llm_model=llm_model,
            max_tokens=max_tokens,
        )
        partial = ""
        for token in fallback_answer.split():
            partial = f"{partial} {token}".strip()
            yield partial
        return

    context = _build_context(results)
    response = client.messages.create(
        model=llm_model,
        max_tokens=max_tokens,
        system=KB_SYSTEM_PROMPT.format(context=context),
        messages=[{"role": "user", "content": question}],
    )
    kb_answer = _extract_answer_text(response)

    if not kb_answer or kb_answer == NO_INFO_MESSAGE:
        kb_answer = _answer_from_general_knowledge(
            client=client,
            question=question,
            llm_model=llm_model,
            max_tokens=max_tokens,
        )

    partial = ""
    for token in kb_answer.split():
        partial = f"{partial} {token}".strip()
        yield partial