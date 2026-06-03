import asyncio
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from routes.schemas import ChatRequest

from agents.workflow import RagGraph

# إنشاء الـ Router
chat_router = APIRouter()


@chat_router.post("/stream")
async def chat_stream(body: ChatRequest, request: Request):
    """Server-Sent Events — Streams both pipeline graph steps and final text token-by-token."""

    chatbot_app = request.app.state.chatbot_app

    async def event_generator():
        config = {"configurable": {"thread_id": body.user_id}}
        inputs = {"question": body.message}

        try:
            # تشغيل الـ Dual Stream
            async for mode, content in chatbot_app.astream(
                inputs, config=config, stream_mode=["updates", "messages"]
            ):

                if mode == "messages":
                    message, meta_data = content
                    node_name = meta_data.get("langgraph_node")

                    if node_name in ["generate_rag_answer", "generate_normal_answer"]:
                        # 🔥 التريكة القاتلة للتكرار هنا 🔥
                        # الـ LangGraph لما بيخلص الـ Invoke جوه النود، بيبعت الـ AIMessage الكلي كـ chunk أخير
                        # بنعرفه إزاي؟ الـ Chunk الحقيقي اللحظي بيكون نوعه 'AIMessageChunk' مش 'AIMessage'
                        # أو بيكون جواه التوكنز الحقيقية فقط. عشان نضمن عدم التكرار:
                        if type(message).__name__ == "AIMessage":
                            continue  # طنش الـ Object النهائي المكتمل لأنه مسبب التكرار

                        token = message.content
                        if token:
                            payload = json.dumps(
                                {"type": "token", "node": node_name, "data": token}
                            )
                            yield f"data: {payload}\n\n"

                elif mode == "updates":
                    for node_name, state_change in content.items():
                        # تجاهل نودز التوليد في الـ updates تماماً لأن الـ messages قايمة بالواجب
                        if node_name in [
                            "generate_rag_answer",
                            "generate_normal_answer",
                        ]:
                            continue

                        payload = json.dumps(
                            {
                                "type": "status",
                                "node": node_name,
                                "emotion": state_change.get("emotion"),
                                "language": state_change.get("language"),
                            }
                        )
                        yield f"data: {payload}\n\n"
                        await asyncio.sleep(0.01)

        except Exception as e:
            print(f"🚨 !!! PIPELINE CRASHED: {str(e)} !!!")
            yield f"data: {json.dumps({'type': 'error', 'data': 'An error occurred during streaming.'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
