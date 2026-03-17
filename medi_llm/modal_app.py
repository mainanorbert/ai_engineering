from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "medi-llm"
SECRET_NAME = "anthropic"
KEEP_WARM = 1

ROOT_DIR = Path(__file__).resolve().parents[1]
MEDI_LLM_DIR = ROOT_DIR / "medi_llm"
REQUIREMENTS_FILE = ROOT_DIR / "medi_llm" / "requirements.txt"

image = (
    modal.Image.debian_slim(python_version="3.11", force_build=True)
    .pip_install_from_requirements(str(REQUIREMENTS_FILE))
    .env({"PYTHONPATH": "/app"})
    .add_local_dir(str(MEDI_LLM_DIR), remote_path="/app/medi_llm")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    min_containers=KEEP_WARM,
    timeout=600,
)
@modal.asgi_app()
def gradio_app():
    import gradio as gr
    from fastapi import FastAPI
    from medi_llm.app import create_demo

    demo = create_demo()
    app = FastAPI()
    return gr.mount_gradio_app(app, demo, path="/")
