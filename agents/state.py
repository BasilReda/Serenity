from typing import Literal, TypedDict, Optional, List, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator

class ChatbotState(TypedDict):
    user_input:            str
    detect_language:       str
    detected_emotion:      str
    intent:                str
    translated_query:      str
    missing_info_feedback: Optional[str]
    history_messages:      Annotated[List[BaseMessage], add_messages]
    status_update:         Annotated[list[str], operator.add]
    search_query:          str
    retrieved_docs:        List[str]
    reranked_docs:         List[str]
    relevance_score:       float
    checking_attempts:     int
    final_response:        str
    hypothetical_answers:  List[str]
    query_complexity:      str
    input_safe:            bool
    output_safe:           bool
    is_final:              bool

class InputGuardrailResult(BaseModel):
    safe:   bool
    threat: Literal["clean","injection","jailbreak","hate","harm_method","violence","probe"]

class OutputGuardrailResult(BaseModel):
    safe: bool
    fix:  str

class ComplexityClassification(BaseModel):
    reasoning: str = Field(description="Why simple or complex.")
    level:     Literal["simple", "complex"]

class HyDEDocument(BaseModel):
    response_1: str = Field(description="First theoretical clinical response.")
    response_2: str = Field(description="Second theoretical clinical response.")

class DocumentGrader(BaseModel):
    score:                 float = Field(description="Relevance score 0-1.")
    missing_info_feedback: str   = Field(description="What is missing.")

class QueryRewriter(BaseModel):
    search_query: str = Field(description="Optimized clinical search query.")
    