from __future__ import annotations

from typing import List

import gradio as gr
from dotenv import load_dotenv

try:
    from .config import DEFAULT_SETTINGS
    from .pipeline import build_knowledge_base
    from .vector_store import get_collection_count
    from .llm import stream_answer_with_sources
    from .schemas import RetrievalResult
except ImportError:
    from config import DEFAULT_SETTINGS
    from pipeline import build_knowledge_base
    from vector_store import get_collection_count
    from llm import stream_answer_with_sources
    from schemas import RetrievalResult

load_dotenv(override=True)


# --- Knowledge-base lazy initialization ----------------------------------------

_kb_ready = False


def ensure_kb_ready() -> str:
    """Build the ChromaDB index if needed, then return a status string."""
    global _kb_ready
    try:
        if _kb_ready:
            return f"OK: Knowledge base ready ({get_collection_count()} chunks indexed)"

        count = get_collection_count()
        if count > 0:
            _kb_ready = True
            return f"OK: Knowledge base ready ({count} chunks indexed)"

        artifacts = build_knowledge_base(
            knowledge_base_path=DEFAULT_SETTINGS.knowledge_base_path,
            vector_db_path=DEFAULT_SETTINGS.vector_db_path,
            collection_name=DEFAULT_SETTINGS.collection_name,
            embedding_model_name=DEFAULT_SETTINGS.embedding_model_name,
            chunk_size=DEFAULT_SETTINGS.chunk_size,
            chunk_overlap=DEFAULT_SETTINGS.chunk_overlap,
            reset_existing=False,
        )
        _kb_ready = True
        return f"OK: Knowledge base built ({artifacts.indexed_chunk_count} chunks indexed)"
    except Exception as exc:
        _kb_ready = False
        return f"Warning: Knowledge base initialization failed: {exc}"


# --- Gradio helper: source citations panel ------------------------------------

_SCORE_COLORS = [
    (5.0, "#16a34a"),   # green  - highly relevant
    (2.0, "#2563eb"),   # blue   - relevant
    (0.0, "#d97706"),   # amber  - marginally relevant
    (-999, "#94a3b8"),  # grey   - low / unknown
]


def _score_color(score: float | None) -> str:
    """Return a hex color based on the cross-encoder rerank score."""
    if score is None:
        return "#94a3b8"
    for threshold, color in _SCORE_COLORS:
        if score >= threshold:
            return color
    return "#94a3b8"


def build_sources_html(sources: List[RetrievalResult]) -> str:
    """Render retrieved chunks as an HTML citation panel."""
    if not sources:
        return (
            "<p style='color:#94a3b8;font-size:0.85em;font-family:system-ui;"
            "padding:12px 4px'>No sources retrieved.</p>"
        )

    cards = []
    for i, result in enumerate(sources, start=1):
        source = result.metadata.get("source", "unknown")
        section = result.metadata.get("section_heading", "")
        score = result.rerank_score
        color = _score_color(score)
        score_label = f"{score:.2f}" if score is not None else "n/a"
        preview = result.text[:220].replace("<", "&lt;").replace(">", "&gt;")
        if len(result.text) > 220:
            preview += "..."

        cards.append(
            f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-left:3px solid {color};
            border-radius:8px;padding:10px 14px;margin-bottom:8px;font-family:system-ui">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
    <span style="font-size:0.73em;color:#64748b;font-weight:600">[{i}] {source}</span>
    <span style="background:{color};color:#fff;padding:1px 7px;border-radius:10px;
                 font-size:0.7em;font-weight:700">score {score_label}</span>
  </div>
  <div style="font-size:0.75em;color:#475569;margin-bottom:5px;font-style:italic">{section}</div>
  <div style="font-size:0.78em;color:#334155;line-height:1.55">{preview}</div>
</div>"""
        )

    header = (
        f"<div style='font-family:system-ui;font-weight:700;font-size:0.88em;"
        f"color:#1e293b;margin-bottom:8px'>Sources: {len(sources)} source chunk"
        f"{'s' if len(sources) != 1 else ''}</div>"
    )
    return header + "".join(cards)


# -----------------------------Gradio streaming chat handler----------------------------------------------

def respond(message: str, history: list):
    """Generate a grounded answer and update chat + sources in one response."""
    message = message.strip()
    if not message:
        return history, "", message

    try:
        ensure_kb_ready()
    except Exception as exc:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Warning: Knowledge base initialization failed: {exc}"},
        ]
        return history, "", ""

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]

    sources: List[RetrievalResult] = []
    final_text = ""
    try:
        for partial_text, partial_sources in stream_answer_with_sources(message):
            final_text = partial_text
            if partial_sources is not None:
                sources = partial_sources or []
    except Exception as exc:
        history[-1]["content"] = f"Warning: Error generating answer: {exc}"
        return history, "", ""

    history[-1]["content"] = final_text
    return history, build_sources_html(sources), ""


# --- Gradio UI ----------------------------------------------------------------

_HEADER_HTML = """
<div style="font-family:system-ui,sans-serif;padding:20px 8px 12px;
            background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
            border-radius:12px;margin-bottom:4px">
  <div style="display:flex;align-items:center;gap:12px">
    <span style="font-size:2.2em">Medi LLM</span>
    <div>
      <h1 style="margin:0;font-size:1.5em;color:#f1f5f9;font-weight:800;
                 letter-spacing:-0.02em">Medi LLM</h1>
      <p style="margin:2px 0 0;color:#94a3b8;font-size:0.85em">
        Respiratory virus Q&amp;A - grounded in CDC &amp; WHO knowledge base
      </p>
    </div>
  </div>
  <p style="margin:12px 0 0;color:#cbd5e1;font-size:0.78em;line-height:1.6">
    Ask about <b style="color:#7dd3fc">common cold</b>,
    <b style="color:#86efac">influenza</b>,
    <b style="color:#fca5a5">COVID-19</b>, or
    <b style="color:#fdba74">RSV</b> -
    symptoms, treatments, prevention, and how they differ.
    Answers are drawn exclusively from the indexed knowledge base.
  </p>
</div>
"""

_EXAMPLE_QUESTIONS = [
    "What are the main symptoms of influenza?",
    "How do COVID-19 and the common cold differ?",
    "What antiviral medicines are used for influenza?",
    "Who is most at risk from RSV?",
    "Which respiratory infections have vaccines available?",
    "How does RSV cause bronchiolitis in infants?",
]

_NO_SOURCES_PLACEHOLDER = (
    "<p style='color:#94a3b8;font-size:0.85em;font-family:system-ui;padding:8px 4px'>"
    "Ask a question to see the knowledge-base chunks used to generate the answer.</p>"
)


def create_demo() -> gr.Blocks:
    """Create the Gradio Blocks app."""
    with gr.Blocks(
        title="Medi LLM - Respiratory Virus Assistant",
    ) as demo:
        gr.HTML(_HEADER_HTML)

        status_bar = gr.Textbox(
            value="Loading: click 'Initialize Knowledge Base' to load the index...",
            label="",
            interactive=False,
            show_label=False,
            container=False,
            elem_id="status_bar",
        )
        init_btn = gr.Button("Initialize Knowledge Base", variant="secondary", size="sm")

        gr.HTML("<hr style='border:none;border-top:1px solid #e2e8f0;margin:8px 0'>")

        with gr.Row(equal_height=True):
            with gr.Column(scale=3, min_width=420):
                chatbot = gr.Chatbot(
                    value=[],
                    render_markdown=True,
                    height=500,
                    show_label=False,
                    avatar_images=(None, None),
                    placeholder=(
                        "<div style='text-align:center;padding:40px 20px;"
                        "color:#94a3b8;font-family:system-ui'>"
                        "<div style='font-size:2.5em;margin-bottom:8px'>Medi LLM</div>"
                        "<p style='font-size:0.9em'>Ask a question about respiratory viruses.</p>"
                        "</div>"
                    ),
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="e.g. What are the symptoms of influenza?",
                        show_label=False,
                        lines=1,
                        scale=9,
                        container=False,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Send ->", variant="primary", scale=2, min_width=90)

                # gr.Examples(
                #     examples=_EXAMPLE_QUESTIONS,
                #     inputs=msg_input,
                #     label="Example questions",
                #     examples_per_page=3,
                # )

                clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm")

            with gr.Column(scale=2, min_width=280):
                gr.Markdown("### Retrieved Sources")
                gr.Markdown(
                    "<small style='color:#64748b'>Knowledge-base chunks used to generate "
                    "the answer, ordered by cross-encoder rerank score.</small>",
                )
                sources_panel = gr.HTML(value=_NO_SOURCES_PLACEHOLDER)

        submit_inputs = [msg_input, chatbot]
        submit_outputs = [chatbot, sources_panel, msg_input]

        send_btn.click(fn=respond, inputs=submit_inputs, outputs=submit_outputs, queue=False)
        msg_input.submit(fn=respond, inputs=submit_inputs, outputs=submit_outputs, queue=False)

        clear_btn.click(
            fn=lambda: ([], _NO_SOURCES_PLACEHOLDER),
            outputs=[chatbot, sources_panel],
            queue=False,
        )

        init_btn.click(fn=ensure_kb_ready, outputs=status_bar, queue=False)
        demo.load(fn=ensure_kb_ready, outputs=status_bar, queue=False)

    return demo

def launch() -> None:
    """Launch the Gradio app locally."""
    demo = create_demo()
    demo.launch(share=False)


if __name__ == "__main__":
    launch()
