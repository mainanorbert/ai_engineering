from pathlib import Path
import re
from typing import Any

import yaml

try:
    from .config import DEFAULT_SETTINGS
    from .schemas import KnowledgeDocument
except ImportError:
    from config import DEFAULT_SETTINGS
    from schemas import KnowledgeDocument


FRONT_MATTER_DELIMITER = "---"
TITLE_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def split_front_matter(raw_text: str) -> tuple[dict[str, Any], str]:
    """Split optional YAML front matter from the markdown body."""
    if not raw_text.startswith(FRONT_MATTER_DELIMITER):
        return {}, raw_text

    parts = raw_text.split(FRONT_MATTER_DELIMITER, maxsplit=2)
    if len(parts) < 3:
        return {}, raw_text

    _, raw_metadata, body = parts
    metadata = yaml.safe_load(raw_metadata) or {}
    return metadata, body.strip()


def infer_title(markdown_text: str, file_path: Path) -> str:
    """Infer a document title from the first markdown H1 heading or file name."""
    match = TITLE_PATTERN.search(markdown_text)
    if match:
        return match.group(1).strip()
    return file_path.stem.replace("_", " ").title()


def infer_category(knowledge_base_path: Path, file_path: Path) -> str:
    """Infer a high-level category from the file path relative to the knowledge base."""
    relative_path = file_path.relative_to(knowledge_base_path)
    if len(relative_path.parts) > 1:
        return relative_path.parts[0]
    return "root"


def normalize_document_metadata(
    metadata: dict[str, Any],
    knowledge_base_path: Path,
    file_path: Path,
) -> dict[str, Any]:
    """Normalize metadata values and add path-derived metadata fields."""
    normalized_metadata: dict[str, Any] = {}

    for key, value in metadata.items():
        if isinstance(value, list):
            normalized_metadata[key] = ", ".join(str(item) for item in value)
        else:
            normalized_metadata[key] = value

    normalized_metadata["relative_path"] = file_path.relative_to(knowledge_base_path).as_posix()
    normalized_metadata["file_name"] = file_path.name
    normalized_metadata["category"] = infer_category(knowledge_base_path, file_path)
    return normalized_metadata


def load_markdown_document(file_path: Path, knowledge_base_path: Path) -> KnowledgeDocument:
    """Load one markdown document and convert it into a structured knowledge document."""
    raw_text = file_path.read_text(encoding="utf-8")
    front_matter, body = split_front_matter(raw_text)
    metadata = normalize_document_metadata(front_matter, knowledge_base_path, file_path)
    relative_source = file_path.relative_to(knowledge_base_path).as_posix()

    return KnowledgeDocument(
        document_id=relative_source.replace("/", "::"),
        source=relative_source,
        category=metadata["category"],
        title=infer_title(body, file_path),
        text=body.strip(),
        metadata=metadata,
    )


def load_knowledge_base_documents(
    knowledge_base_path: Path | None = None,
) -> list[KnowledgeDocument]:
    """Load all markdown documents from the medical knowledge base directory."""
    base_path = knowledge_base_path or DEFAULT_SETTINGS.knowledge_base_path
    markdown_files = sorted(base_path.rglob("*.md"))
    return [load_markdown_document(file_path, base_path) for file_path in markdown_files]
