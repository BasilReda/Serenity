# MindEase AI

MindEase AI is an empathetic mental health chat assistant built with FastAPI, LangChain, and a retrieval-augmented workflow. The app detects user language and emotion, routes the request through classification and retrieval steps, and streams a compassionate answer back to the browser in real time.

## What this app does

- Provides a mental health support assistant for conversational queries
- Detects user language and emotional tone from input text
- Uses retrieval-augmented generation (RAG) when relevant context is available
- Streams responses token-by-token to the frontend for a live chat experience
- Supports a simple web interface served by FastAPI

## Main features

- Language detection and optional translation pipeline
- Emotion classification to make responses more empathetic
- Intent routing for normal vs. RAG-enabled answers
- SSE streaming endpoint for incremental reply display
- Modular architecture with agents, nodes, and pipeline graph logic

## Project structure

```
NLP_Project/
├── README.md                      # Project documentation and usage guide
└── src/
    ├── main.py                    # FastAPI application entry point
    ├── pyproject.toml             # Python project metadata and dependencies
    ├── agents/                    # Pipeline graph, states, and agent nodes
    │   ├── __init__.py
    │   ├── state.py
    │   ├── workflow.py
    │   └── nodes/                 # Individual workflow node implementations
    │       ├── __init__.py
    │       ├── generation.py
    │       ├── other_nodes.py
    │       ├── retrieval.py
    │       └── routing.py
    ├── classifiers/              # ML classification modules
    │   ├── __init__.py
    │   ├── emotion_classifier.py
    │   ├── intent_classifier.py
    │   └── language_identification.py
    ├── helpers/                  # App configuration and utilities
    │   ├── __init__.py
    │   └── config.py
    ├── notebooks/                # Jupyter notebooks for model exploration
    │   ├── emotion_classifier.ipynb
    │   ├── intent_classifier.ipynb
    │   ├── language_identification.ipynb
    │   ├── rag_answer_searching.ipynb
    │   ├── rag_question_searching.ipynb
    │   └── trained_models/
    │       └── emotion_classifier_model/
    │           ├── config.json
    │           ├── model.safetensors
    │           ├── tokenizer_config.json
    │           └── tokenizer.json
    ├── routes/                   # FastAPI router endpoints definition
    │   ├── __init__.py
    │   ├── chat.py
    │   └── schemas/
    │       ├── __init__.py
    │       └── chat_request.py
    ├── static/                   # Frontend static assets
    │   ├── css/
    │   │   └── main.css
    │   └── js/
    │       └── chat.js
    ├── store/                    # Storage, embedding, and LLM wrappers
    │   ├── __init__.py
    │   ├── embeddings.py
    │   ├── llm.py
    │   └── vector_store.py
    ├── templates/                # HTML templates for the frontend
    │   └── index.html
    └── trained_models/           # Saved model artifacts
        └── emotion_classifier_model/
            ├── config.json
            ├── model.safetensors
            ├── tokenizer_config.json
            └── tokenizer.json
```

## Requirements

- Python 3.13 or newer
- A valid `GROQ_API_KEY`
- A valid `COHERE_API_KEY`
- The correct model names for `LLM_MODEL_NAME` and `EMBEDDING_MODEL_NAME`

Dependencies are declared in `src/pyproject.toml`.

## Installation

1. Open a terminal and go to the project root:

```bash
cd d:/ITI/projects/NLP_Project/src
```

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Upgrade `pip` and install dependencies.

The package list is in `src/pyproject.toml`.

```powershell
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn langchain langchain-groq langchain-qdrant scikit-learn torch transformers
```

> If you prefer a dependency file, create `requirements.txt` from the `src/pyproject.toml` dependency list.

## Environment configuration

Create a `.env` file in the `src/` folder with the required settings:

```text
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
LLM_MODEL_NAME=your_llm_model_name
EMBEDDING_MODEL_NAME=your_embedding_model_name
APP_TITLE=MindEase AI
APP_HOST=127.0.0.1
APP_PORT=8000
```

## Running the app

From the `src/` folder run:

```powershell
python main.py
```

Then open your browser to:

```text
http://127.0.0.1:8000
```

Alternatively, run with Uvicorn directly:

```powershell
uvicorn main:app --reload
```

## How the app works

- `src/main.py` starts the FastAPI app and mounts the static frontend
- `src/routes/chat.py` exposes the `/chat/stream` SSE endpoint for chat streaming
- `src/agents/workflow.py` defines the pipeline graph and routing logic
- `src/agents/nodes/` contains individual pipeline steps for language detection, emotion classification, retrieval, and response generation
- `src/store/llm.py` configures the LLM client and model

## Notes

- The web frontend is served from `src/static/` and `src/templates/index.html`
- The chat interface uses JavaScript to stream token updates from the backend
- Use the notebooks in `src/notebooks/` for model exploration and training workflows

## Troubleshooting

- If the app fails to start, confirm your `.env` file is in `src/` and contains valid API keys
- If dependencies fail to install, double-check that your Python version is 3.13 or later
- If the browser does not load, verify the host and port in `APP_HOST` / `APP_PORT`

