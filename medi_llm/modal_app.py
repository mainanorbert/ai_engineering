from __future__ import annotations
from pathlib import Path

import modal

APP_NAME = "medi-llm"
SECRET_NAME = "anthropic"

ROOT_DIR = Path(__file__).resolve().parents[1]
MEDI_LLM_DIR = ROOT_DIR / "medi_llm"
REQUIREMENTS_FILE = ROOT_DIR / "medi_llm" / "requirements.txt"

# Build a Modal image with your dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(REQUIREMENTS_FILE))
    .env({"PYTHONPATH": "/app"})
    .add_local_dir(str(MEDI_LLM_DIR), remote_path="/app/medi_llm")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    # === IMPORTANT: limit to 1 container for sticky sessions ===
    min_containers=1,
    max_containers=1,
    timeout=600,
)
@modal.concurrent(max_inputs=100)  # allows multiple concurrent users in one container
@modal.asgi_app()
def gradio_app():
    # Import inside the image context so that Modal bundles dependencies correctly
    with image.imports():
        import gradio as gr
        from fastapi import FastAPI
        from medi_llm.app import create_demo

    demo = create_demo()
    fastapi_app = FastAPI()

    # Mount Gradio app into FastAPI
    return gr.mount_gradio_app(fastapi_app, demo, path="/")