# modules/rag.py

def grade_document_relevance_api(query: str, document: str) -> str:
    """
    Calls Groq with a strict prompt: "Is this document relevant to the query? Answer 'yes' or 'no'."
    """
    # TODO: Implement LLM grader
    pass

def grade_hallucination_api(generation: str, context: list[str]) -> str:
    """
    Calls Groq to check if the generated answer is fully supported by the context.
    Answers 'yes' (hallucinating) or 'no' (grounded).
    """
    # TODO: Implement hallucination checker
    pass

def rewrite_query_api(query: str, emotion: str) -> str:
    """
    Calls Groq to rewrite the user's query to be better optimized for vector search.
    """
    # TODO: Ask LLM to output a better search string
    pass

# --- New LangGraph Nodes & Edges for rag.py ---

def grade_context_node(state: dict) -> dict:
    """LangGraph Node: Filters out bad context chunks using the grader."""
    query = state.get("search_query", state["user_input"])
    context = state.get("retrieved_context", [])
    
    # Filter context keeping only relevant docs
    relevant_docs = [doc for doc in context if grade_document_relevance_api(query, doc) == "yes"]
    
    status = "yes" if relevant_docs else "no"
    return {"retrieved_context": relevant_docs, "is_context_relevant": status}

def rewrite_query_node(state: dict) -> dict:
    """LangGraph Node: Rewrites the search query and increments loop counter."""
    query = state.get("search_query", state["user_input"])
    attempts = state.get("generation_attempts", 0)
    
    new_query = rewrite_query_api(query, state.get("detected_emotion", ""))
    
    return {
        "search_query": new_query, 
        "generation_attempts": attempts + 1
    }

def check_hallucination_edge(state: dict) -> str:
    """
    LangGraph Conditional Edge: Decides if we can output the answer or need to regenerate.
    """
    # If we tried too many times, force exit to avoid API rate limits
    if state.get("generation_attempts", 0) >= 3:
        return "end"
        
    is_hallucinating = grade_hallucination_api(state["final_response"], state["retrieved_context"])
    
    if is_hallucinating == "yes":
        return "regenerate"  # Loop back to generation node
    else:
        return "end"         # Safe to output to user