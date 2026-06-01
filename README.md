# Serenity — Mental Health Chatbot

## Project Structure

```
mental_health_chatbot/
│
├── main.py                        # FastAPI app + SSE /chat/stream endpoint
│
├── core/
│   ├── config.py                  # All settings via pydantic-settings + .env
│   └── models.py                  # API request/response schemas
│
├── ml/
│   ├── llm.py                     # LLM factories (Gemma Ollama + DPO local)
│   ├── classifiers.py             # Emotion, language, intent detection
│   └── translator.py              # MarianMT many-to-one translator
│
├── rag/
│   ├── encoders.py                # Dense (HuggingFace) + sparse (BM25) encoders
│   ├── retriever.py               # Pinecone hybrid retriever + document utils
│   └── reranker.py                # CrossEncoder reranker (lazy singleton)
│
├── agents/
│   ├── state.py                   # ChatbotState TypedDict + all Pydantic schemas
│   ├── graph.py                   # Graph assembly + compiled app singleton
│   └── nodes/
│       ├── preprocessing.py       # Language detection, translation, emotion
│       ├── guardrails.py          # Input + output guardrail nodes
│       ├── routing.py             # Intent detection, complexity classifier, routers
│       ├── retrieval.py           # HyDE, RAG, ReRanker, Grader, QueryRewrite
│       └── generation.py          # Reset node, mental_health_chatbot, general_handler
│
├── templates/
│   └── index.html                 # Jinja2 chat UI template
│
├── static/
│   ├── css/main.css               # All styles (biophilic forest theme)
│   └── js/chat.js                 # SSE client, pipeline visualization, session
│
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. Clone and enter the project
cd mental_health_chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your PINECONE_API_KEY

# 5. Start Ollama (in a separate terminal)
ollama serve
ollama pull gemma2:2b

# 6. Run the app
python main.py
# → http://localhost:8000
```

## Architecture Decisions

| Decision | Reason |
|---|---|
| One `config.py` | All paths and constants in one place — no scattered hardcoded strings |
| Lazy singletons for all models | Models load once at startup, not on every request |
| `agents/nodes/` split into 5 files | Each file has one responsibility — easy to test or swap |
| `agents/graph.py` owns `hybrid_retriever` | Single source of truth for the shared RAG resource |
| SSE over WebSockets | Simpler, stateless, works through proxies without upgrade headers |
| CSS + JS as separate static files | Cacheable by browser, no inline bloat in HTML |

## Pipeline Flow

```
START
 └─ reset_per_turn_state      clears all routing fields, preserves history
     └─ language_detector
         ├─ [english]  ──────── emotion_detector
         └─ [not english] ─── translate_to_english → emotion_detector
                                    └─ input_guardrail
                                        ├─ [blocked] ──── output_guardrail → END
                                        └─ [safe]
                                             └─ intent_detector
                                                 ├─ [general] ──── general_handler
                                                 └─ [mental_health]
                                                      └─ query_classifier
                                                          ├─ [simple] ─── mental_health_chatbot
                                                          └─ [complex]
                                                               └─ HyDE → RAG → ReRanker → grade_document
                                                                   ├─ [score>=0.7] ─── mental_health_chatbot
                                                                   └─ [score<0.7]  ─── query_rewrite → RAG (retry)
                                                                                                      └─ (max 2 retries, then generate)
                                        mental_health_chatbot ──┐
                                        general_handler ────────┤
                                                                 └─ output_guardrail → END
```