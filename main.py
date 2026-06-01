from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import json

from core.config import settings
from core.models import ChatRequest
from agents.graph import chatbot_app

# ── Resolve paths relative to this file so they work regardless of CWD ────────
BASE_DIR      = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR    = BASE_DIR / "static"


# ── App lifecycle ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[API] {settings.APP_TITLE} ready.")
    yield
    print("[API] Shutting down.")


app = FastAPI(title=settings.APP_TITLE, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    # FileResponse bypasses Jinja2 entirely — index.html has no template syntax
    return FileResponse(str(TEMPLATES_DIR / "index.html"))


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    """Server-Sent Events — streams pipeline status + final response."""

    async def event_generator():
        config = {"configurable": {"thread_id": body.user_id}}
        inputs = {"user_input": body.message}

        try:
            async for output in chatbot_app.astream(inputs, config=config):
                for node_name, state_change in output.items():
                    
                    # 1. Safely check for status updates
                    status_updates = state_change.get("status_update")
                    if status_updates:
                        payload = json.dumps({
                            "type": "status",
                            "node": node_name,
                            "data": status_updates[-1],
                        })
                        yield f"data: {payload}\n\n"
                        await asyncio.sleep(0)

                    # 2. Check for your shiny new flag! 
                    if state_change.get("is_final") is True:
                        final_text = state_change.get("final_response")
                        if final_text:
                            payload = json.dumps({
                                "type": "response",
                                "node": node_name,
                                "data": final_text,
                            })
                            yield f"data: {payload}\n\n"
                            await asyncio.sleep(0)

        except Exception as e:
            # This is what's triggering your error bubble!
            print(f"!!! PIPELINE CRASHED: {str(e)} !!!") 
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=False)