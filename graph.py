from typing import TypedDict, List, Dict, Any

class ChatbotState(TypedDict):
    user_input: str
    detected_language: str
    detected_emotion: str
    intent: str
    
    # --- New Reflective RAG States ---
    search_query: str          # Starts as user_input, but might get rewritten
    retrieved_context: List[str]
    generation_attempts: int   # Counter to prevent infinite loops
    is_context_relevant: str   # 'yes' or 'no'
    is_hallucinating: str      # 'yes' or 'no'
    
    final_response: str