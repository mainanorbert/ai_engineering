# Insurellm RAG Agent (`ragllm.ipynb`)

This project is a tool-calling RAG agent built with LangChain + Claude and a Gradio UI.

It answers Insurellm questions from a prebuilt vector database and transparently falls back to web search when the knowledge base does not contain relevant context. It also supports audio questions via speech-to-text.

## Features

- RAG retrieval from local Chroma DB (`preprocessed_db`, collection `docs`)
- Chunk reranking using Claude structured output
- Tool-calling agent loop with `handle_tool_call(...)`
- Transparent web fallback message when KB context is missing
- Audio input support (microphone) using `transcriber.py`
- Gradio chat interface

## Architecture

The agent uses three tools:

1. `rag_search(query)`  
   Retrieves and reranks context from the vector store.
2. `web_search(query)`  
   Uses DuckDuckGo for general web information.
3. `transcribe_audio(path_to_file)`  
   Transcribes audio to text before sending to the agent.

At runtime, Claude decides which tool(s) to call. If web search is used, the final response is prefixed with a transparency statement so users know the answer is from the internet, not the Insurellm KB.

## Requirements

- Python environment with project dependencies installed
- A valid Anthropic API key in environment (via `.env`)
- Prebuilt vector store already available:
  - DB path: `preprocessed_db`
  - Collection name: `docs`
- For web fallback:
  - `ddgs` package installed (required by DuckDuckGo tool)
- For audio transcription:
  - `ffmpeg` installed and available on PATH
  - `week5-RAG/transcriber.py` available

## Setup

From project root:

```bash
cd week5-RAG
```

Install optional runtime dependencies if missing:

```bash
pip install -U ddgs
```

For ffmpeg (choose one):

```bash
conda install -c conda-forge ffmpeg -y
```

or

```bash
sudo apt install -y ffmpeg
```

## Run

1. Open `week5-RAG/ragllm.ipynb`.
2. Run cells top-to-bottom.
3. Launch the app from the final cell:

```python
demo_app.launch(share=True)
```

## Usage

- Type a text question and click **Send**.
- Or record a question using the microphone input and click **Send**.
- The app transcribes audio first, then sends the transcribed text to the agent.
- Use **Refresh Vector Store Status** to verify the vector DB is accessible.

## Expected Behavior

- If relevant KB context exists, the answer is KB-grounded.
- If KB context is not relevant, the agent performs web search and clearly states:
  - `"There is no such information in our Insurellm knowledge base. Here is general information from the internet:"`

## Troubleshooting

- `Collection 'docs' was not found`  
  Build/populate `preprocessed_db` first.

- `Could not import ddgs`  
  Install with `pip install -U ddgs`.

- `ffmpeg was not found`  
  Install ffmpeg and restart the notebook kernel.

- Microphone gives no transcription  
  Ensure browser mic permission is enabled and re-record audio.
