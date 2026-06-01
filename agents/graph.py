from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agents.state import ChatbotState
from agents.nodes.preprocessing import (language_detection_node, is_language_english,
    translate_to_english_node, emotion_detection_node)
from agents.nodes.guardrails import (input_guardrail_node, input_guardrail_router, output_guardrail_node)
from agents.nodes.routing import (intent_detection_node, intent_router, query_classifier_agent, complexity_router)
from agents.nodes.retrieval import (HyDE, RAG, ReRanker, grade_document, check_relevance, query_rewrite)
from agents.nodes.generation import reset_per_turn_state, mental_health_chatbot, general_handler
from rag.encoders import get_encoders
from rag.retriever import setup_hybrid_retriever

print("[GRAPH] Initialising RAG resources...")
_dense_enc, _sparse_enc = get_encoders()
hybrid_retriever = setup_hybrid_retriever(_dense_enc, _sparse_enc)
print("[GRAPH] RAG resources ready.")

def build_graph():
    graph = StateGraph(ChatbotState)
    graph.add_node("reset_per_turn_state",   reset_per_turn_state)
    graph.add_node("language_detector",      language_detection_node)
    graph.add_node("translate_to_english",   translate_to_english_node)
    graph.add_node("emotion_detector",       emotion_detection_node)
    graph.add_node("input_guardrail",        input_guardrail_node)
    graph.add_node("intent_detector",        intent_detection_node)
    graph.add_node("query_classifier_agent", query_classifier_agent)
    graph.add_node("HyDE",                   HyDE)
    graph.add_node("RAG",                    RAG)
    graph.add_node("ReRanker",               ReRanker)
    graph.add_node("grade_document",         grade_document)
    graph.add_node("query_rewrite",          query_rewrite)
    graph.add_node("mental_health_chatbot",  mental_health_chatbot)
    graph.add_node("general_handler",        general_handler)
    graph.add_node("output_guardrail",       output_guardrail_node)

    graph.add_edge(START, "reset_per_turn_state")
    graph.add_edge("reset_per_turn_state", "language_detector")
    graph.add_conditional_edges("language_detector", is_language_english,
        {"english": "emotion_detector", "not english": "translate_to_english"})
    graph.add_edge("translate_to_english", "emotion_detector")
    graph.add_edge("emotion_detector",     "input_guardrail")
    graph.add_conditional_edges("input_guardrail", input_guardrail_router,
        {"safe": "intent_detector", "blocked": "output_guardrail"})
    graph.add_conditional_edges("intent_detector", intent_router,
        {"mental_health": "query_classifier_agent", "general": "general_handler"})
    graph.add_conditional_edges("query_classifier_agent", complexity_router,
        {"complex_path": "HyDE", "simple_path": "mental_health_chatbot"})
    graph.add_edge("HyDE",     "RAG")
    graph.add_edge("RAG",      "ReRanker")
    graph.add_edge("ReRanker", "grade_document")
    graph.add_conditional_edges("grade_document", check_relevance,
        {"generate": "mental_health_chatbot", "rewrite": "query_rewrite"})
    graph.add_edge("query_rewrite",         "RAG")
    graph.add_edge("mental_health_chatbot", "output_guardrail")
    graph.add_edge("general_handler",       "output_guardrail")
    graph.add_edge("output_guardrail",      END)
    return graph.compile(checkpointer=MemorySaver())

chatbot_app = build_graph()
chatbot_app

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧠 Starting PsyNet Graph Tester")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    # The MemorySaver needs a thread_id to remember the conversation history
    config = {"configurable": {"thread_id": "cli_test_session"}}

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Shutting down tester...")
            break
            
        if not user_input.strip():
            continue

        # Setup the initial state payload. 
        # NOTE: Make sure "user_input" matches the exact key name in your ChatbotState!
        inputs = {"user_input": user_input} 

        print("\n[Graph Execution Path]")
        
        try:
            # .stream() allows us to watch the state move from node to node
            for output in chatbot_app.stream(inputs, config=config):
                for node_name, state_change in output.items():
                    print(f" 🟢 -> {node_name}")
                    
                    # Optional: Print specific state changes here if you want to inspect them
                    # if "detected_emotion" in state_change:
                    #     print(f"      Emotion: {state_change['detected_emotion']}")

            # After the stream finishes, fetch the final state to get the bot's response
            final_state = chatbot_app.get_state(config).values
            
            # NOTE: Change "final_response" to whatever key your output_guardrail saves the text into
            bot_reply = final_state.get("final_response", "[No 'final_response' found in state]")
            print(f"\nBot: {bot_reply}")

        except Exception as e:
            print(f"\n❌ [ERROR] Graph execution failed: {str(e)}")
            # Raise the error so you can see the full traceback and fix the bug
            raise e