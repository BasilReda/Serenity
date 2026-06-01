from agents.state import ChatbotState, ComplexityClassification
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from ml.classifiers import intent_user
from ml.llm import get_global_llm
import json

def intent_detection_node(state: ChatbotState) -> dict:
    print("--> [NODE] Executing: intent_detection_node")
    text = state.get("translated_query") or state["user_input"]
    print(f"   [DEBUG INTENT] Classifying: '{text}'")
    info = intent_user(text)
    return {"intent": info["intent"],
            "status_update": [f"Detected Intent: {info['intent']}, Confidence: {info['confidence']}"]}

def intent_router(state: ChatbotState) -> str:
    print("--> [EDGE] Executing: intent_router")
    intent = str(state.get("intent", "")).strip().lower()
    print(f"   [ROUTER] Intent: '{intent}'")
    return "mental_health" if intent == "asking_mental_health_question" else "general"

def query_classifier_agent(state: ChatbotState) -> dict:
    print("--> [NODE] Executing: query_classifier_agent")
    query  = state.get("translated_query", state["user_input"])
    parser = PydanticOutputParser(pydantic_object=ComplexityClassification)
    sys_prompt = """You are the ROUTING AGENT for a mental health NLP pipeline.
Route to "simple" if user is venting/seeking comfort.
Route to "complex" if user asks a factual/clinical question.
{format_instructions}
Output ONLY raw JSON: {{"reasoning": "...", "level": "simple|complex"}}"""

    chain = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "Query: {query}\n\n{format_instructions}"),
    ]) | get_global_llm()
    raw  = chain.invoke({"query": query, "format_instructions": parser.get_format_instructions()})
    text = (raw if isinstance(raw, str) else raw.content)
    try:
        parsed = parser.parse(text)
    except OutputParserException:
        try:
            d = json.loads(text)
            parsed = ComplexityClassification(**(d.get("properties", d)))
        except Exception:
            parsed = ComplexityClassification(reasoning="Fallback", level="simple")
    return {"query_complexity": parsed.level, "status_update": [f"Query complexity: {parsed.level}"]}

def complexity_router(state: ChatbotState) -> str:
    print("--> [EDGE] Executing: complexity_router")
    return "complex_path" if state.get("query_complexity") == "complex" else "simple_path"
