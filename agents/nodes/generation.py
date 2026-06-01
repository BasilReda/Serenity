from agents.state import ChatbotState
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from ml.llm import get_dpo_model

def reset_per_turn_state(state: ChatbotState) -> dict:
    print("--> [NODE] Executing: reset_per_turn_state")
    return {
        "intent": "", "query_complexity": "", "search_query": "",
        "retrieved_docs": [], "reranked_docs": [], "relevance_score": 0.0,
        "checking_attempts": 0, "hypothetical_answers": [],
        "missing_info_feedback": None, "final_response": "",
        "translated_query": "", "detected_emotion": "", "detect_language": "",
        "input_safe": True, "output_safe": True,
    }

def mental_health_chatbot(state: ChatbotState) -> dict:
    print("--> [NODE] Executing: mental_health_chatbot")
    current_question = state["user_input"]
    target_language  = state.get("detect_language", "English")
    retrieved_docs   = state.get("reranked_docs", [])
    recent_history   = state.get("history_messages", [])[-4:]
    context_str      = "\n".join(retrieved_docs) if retrieved_docs else "No specific external context provided."

    system_prompt = """[ROLE]
You are an empathetic, evidence-based wellness and support assistant. Warm, validating, professional.
[TASK] Synthesize the provided context to answer accurately and conversationally.
[CONSTRAINTS]
1. Ground response in retrieved context. If empty, rely on empathy and active listening.
2. Keep paragraphs concise.
3. Never make a definitive clinical diagnosis. Never prescribe medications or dosages.
4. PROFESSIONAL REFERRAL RULE: If question involves symptoms/patterns/clinical concerns,
   end with one sentence recommending a mental health professional.
5. Plain conversational prose only. No bullet lists unless asked."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="recent_history"),
        ("human", "Retrieved Context:\n{context}\n\nUser Query: {question}\n\n[RESPOND ENTIRELY IN {language}]"),
    ])
    response   = (prompt | get_dpo_model()).invoke({
        "context": context_str, "recent_history": recent_history,
        "question": current_question, "language": target_language})
    final_text = response if isinstance(response, str) else response.content
    return {
        "history_messages": [HumanMessage(content=current_question), AIMessage(content=final_text)],
        "final_response":   final_text,
        "status_update":    [f"Response drafted in {target_language}."],
    }

def general_handler(state: ChatbotState) -> dict:
    print("--> [NODE] Executing: general_handler")
    current_question = state.get("translated_query", state["user_input"])
    target_language  = state.get("detect_language", "English")
    recent_history   = state.get("history_messages", [])[-4:]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a warm, friendly assistant.
Respond naturally to greetings and general questions.
If user moves toward mental health topics, gently let them know you can help.
CRITICAL: Respond in {language}."""),
        MessagesPlaceholder(variable_name="recent_history"),
        ("human", "{question}"),
    ])
    response   = (prompt | get_dpo_model()).invoke({
        "question": current_question, "language": target_language,
        "recent_history": recent_history})
    final_text = response if isinstance(response, str) else response.content
    return {
        "history_messages": [HumanMessage(content=current_question), AIMessage(content=final_text)],
        "final_response":   final_text,
        "status_update":    ["Handled as general message."],
    }
