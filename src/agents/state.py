from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


class RagState(TypedDict):
    history: Annotated[list[AnyMessage], add_messages]
    question: str = None
    translated_question: str = None
    retrieved_docs: list[Document] = None
    answer: str = None
    language: str = None
    emotion: str = None
    score: float = None
    intent: str = None
    hyde_context: str = None
