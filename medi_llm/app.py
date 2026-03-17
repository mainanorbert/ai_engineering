from __future__ import annotations

import gradio as gr
from dotenv import load_dotenv

try:
    from .config import DEFAULT_SETTINGS
    from .pipeline import build_knowledge_base
    from .vector_store import get_collection_count
    from .llm import answer_question_stream
except ImportError:
    from config import DEFAULT_SETTINGS
    from pipeline import build_knowledge_base
    from vector_store import get_collection_count
    from llm import answer_question_stream

load_dotenv(override=True)


# --- Knowledge-base lazy initialization ----------------------------------------

_kb_ready = False


def ensure_kb_ready() -> str:
    """Build the ChromaDB index if needed, then return a status string."""
    import shutil
    global _kb_ready
    try:
        if _kb_ready:
            return f"OK: Knowledge base ready ({get_collection_count()} chunks indexed)"

        try:
            count = get_collection_count()
        except Exception:
            # Existing DB is unreadable (schema mismatch / corrupted) — wipe and rebuild
            shutil.rmtree(DEFAULT_SETTINGS.vector_db_path, ignore_errors=True)
            count = 0

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
            reset_existing=True,
        )
        _kb_ready = True
        return f"OK: Knowledge base built ({artifacts.indexed_chunk_count} chunks indexed)"
    except Exception as exc:
        _kb_ready = False
        return f"Warning: Knowledge base initialization failed: {exc}"





# -----------------------------Gradio streaming chat handler----------------------------------------------

def respond(message: str, history: list):
    """Generate an answer and stream it to the chat."""
    message = message.strip()
    if not message:
        yield history, message
        return

    try:
        ensure_kb_ready()
    except Exception as exc:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Warning: Knowledge base initialization failed: {exc}"},
        ]
        yield history, ""
        return

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]

    try:
        for partial_text in answer_question_stream(message):
            history[-1]["content"] = partial_text
            yield history, ""
    except Exception as exc:
        history[-1]["content"] = f"Warning: Error generating answer: {exc}"
        yield history, ""


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

        clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm")

        submit_inputs = [msg_input, chatbot]
        submit_outputs = [chatbot, msg_input]

        send_btn.click(fn=respond, inputs=submit_inputs, outputs=submit_outputs)
        msg_input.submit(fn=respond, inputs=submit_inputs, outputs=submit_outputs)

        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, msg_input],
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
