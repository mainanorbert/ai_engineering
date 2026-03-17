from dataclasses import dataclass
import re

try:
    from .config import DEFAULT_SETTINGS
    from .schemas import DocumentChunk, KnowledgeDocument
except ImportError:
    from config import DEFAULT_SETTINGS
    from schemas import DocumentChunk, KnowledgeDocument


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass(slots=True)
class MarkdownSection:
    """Represent a markdown section plus its heading hierarchy."""

    heading_path: list[str]
    body: str


def normalize_text_block(text: str) -> str:
    """Normalize blank lines and surrounding whitespace in a text block."""
    cleaned_text = text.replace("\r\n", "\n").strip()
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text


def split_markdown_sections(markdown_text: str) -> list[MarkdownSection]:
    """Split markdown into sections while preserving heading hierarchy."""
    lines = markdown_text.splitlines()
    sections: list[MarkdownSection] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []

    for line in lines:
        heading_match = HEADING_PATTERN.match(line.strip())
        if heading_match:
            if current_lines:
                sections.append(
                    MarkdownSection(
                        heading_path=heading_stack.copy() or ["Overview"],
                        body=normalize_text_block("\n".join(current_lines)),
                    )
                )
                current_lines = []

            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(heading_text)
            continue

        current_lines.append(line)

    if current_lines:
        sections.append(
            MarkdownSection(
                heading_path=heading_stack.copy() or ["Overview"],
                body=normalize_text_block("\n".join(current_lines)),
            )
        )

    return [section for section in sections if section.body]


def choose_split_boundary(text: str, start_index: int, target_end: int) -> int:
    """Choose a natural split boundary near the target end position."""
    if target_end >= len(text):
        return len(text)

    candidate_boundaries = [
        text.rfind("\n\n", start_index, target_end),
        text.rfind(". ", start_index, target_end),
        text.rfind("; ", start_index, target_end),
        text.rfind(", ", start_index, target_end),
        text.rfind(" ", start_index, target_end),
    ]

    best_boundary = max(candidate_boundaries)
    if best_boundary <= start_index:
        return target_end

    if text.startswith("\n\n", best_boundary):
        return best_boundary

    return best_boundary + 1


def split_text_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping character windows using natural boundaries."""
    normalized_text = normalize_text_block(text)
    if len(normalized_text) <= chunk_size:
        return [normalized_text]

    chunks: list[str] = []
    start_index = 0

    while start_index < len(normalized_text):
        target_end = min(len(normalized_text), start_index + chunk_size)
        end_index = choose_split_boundary(normalized_text, start_index, target_end)
        chunk_text = normalized_text[start_index:end_index].strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end_index >= len(normalized_text):
            break

        next_start_index = max(0, end_index - chunk_overlap)
        if next_start_index <= start_index:
            next_start_index = end_index
        start_index = next_start_index

    return chunks


def build_chunk_text(heading_path: list[str], body_chunk: str) -> str:
    """Build the final chunk text with heading context repeated for retrieval."""
    heading_context = " > ".join(heading_path)
    return f"{heading_context}\n\n{body_chunk}".strip()


def chunk_document(
    document: KnowledgeDocument,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Chunk one knowledge-base document into heading-aware retrieval units."""
    effective_chunk_size = chunk_size or DEFAULT_SETTINGS.chunk_size
    effective_chunk_overlap = chunk_overlap or DEFAULT_SETTINGS.chunk_overlap
    document_chunks: list[DocumentChunk] = []
    sections = split_markdown_sections(document.text)

    for section_index, section in enumerate(sections):
        text_windows = split_text_with_overlap(
            text=section.body,
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap,
        )

        for window_index, window_text in enumerate(text_windows):
            chunk_id = f"{document.document_id}::section_{section_index}::chunk_{window_index}"
            chunk_metadata = {
                "document_id": document.document_id,
                "source": document.source,
                "title": document.title,
                "category": document.category,
                "section_heading": " > ".join(section.heading_path),
                "section_index": section_index,
                "chunk_index": window_index,
            }
            chunk_metadata.update(document.metadata)
            document_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    text=build_chunk_text(section.heading_path, window_text),
                    metadata=chunk_metadata,
                )
            )

    return document_chunks


def chunk_documents(
    documents: list[KnowledgeDocument],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Chunk every loaded document into retrieval-ready chunks."""
    all_chunks: list[DocumentChunk] = []
    for document in documents:
        all_chunks.extend(
            chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return all_chunks
