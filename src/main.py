from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
from routes import chat_router

from helpers.config import get_settings
from agents.workflow import RagGraph

# ── Resolve paths relative to this file ────────────────────────────────────────
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


app_settings = get_settings()


# ── App lifecycle ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    print(f"🚀 [API] {app_settings.APP_TITLE} is spinning up and ready.")
    app.state.rag_graph = RagGraph()
    app.state.chatbot_app = app.state.rag_graph.get_agent()
    yield
    # Shutdown code
    print(f"🛑 [API] {app_settings.APP_TITLE} is shutting down.")
    app.state.rag_graph.close()


# ── FastAPI app setup ─────────────────────────────────────────────────────────
app = FastAPI(title=app_settings.APP_TITLE, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files and templates ───────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Routes ─────────────────────────────────────────────────────────────────────

app.include_router(chat_router, prefix="/chat", tags=["Chat Engine"])


@app.get("/")
async def serve_index():
    return FileResponse(str(TEMPLATES_DIR / "index.html"))


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    uvicorn.run(
        "main:app", host=app_settings.APP_HOST, port=app_settings.APP_PORT, reload=False
    )
