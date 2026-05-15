def classify_intent_api(text: str) -> str:
    """
    Calls the free Groq API with a zero-shot or few-shot system prompt.
    
    Expected output categories: 
    'greeting', 'goodbye', 'gratitude', 'asking_mental_health_question', 'out_of_scope'
    """
    # TODO: Build LLM prompt instructing it to output ONLY the category string.
    # Call Groq client.chat.completions.create(...)
    pass

def intent_node(state: dict) -> dict:
    """
    LangGraph Node: Classifies intent and updates state.
    """
    user_input = state.get("user_input", "")
    
    intent = classify_intent_api(user_input)
    
    return {"intent": intent.strip().lower()}

def route_intent(state: dict) -> str:
    """
    LangGraph Conditional Edge Function: Decides the next step based on intent.
    DO NOT return a state dict here; return the exact string name of the next node.
    """
    intent = state.get("intent", "")
    
    if intent == "asking_mental_health_question":
        return "rag_node"
    else:
        return "direct_response_node"