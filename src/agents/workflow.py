from langgraph.graph import StateGraph, END, START
from agents.state import RagState
from agents.nodes import RoutingNodes, RetrievalNodes, OtherNodes, GenerationNodes
from store import QdrantStore
from langgraph.checkpoint.memory import InMemorySaver


class RagGraph:
    def __init__(self):
        self.checkpoint_saver = InMemorySaver()
        self.routing_nodes = RoutingNodes()
        self.retrieval_nodes = RetrievalNodes()
        self.other_nodes = OtherNodes()
        self.generation_nodes = GenerationNodes()
        self.graph = StateGraph(RagState)

        self.graph.add_node("detect_language", self.other_nodes.detect_language)
        self.graph.add_node("emotion_classifier", self.other_nodes.emotion_classifier)
        self.graph.add_node(
            "translate_into_english", self.other_nodes.translate_into_english
        )
        self.graph.add_node("intent_classifier", self.other_nodes.intent_classifier)
        self.graph.add_node(
            "retrieve_relevant_docs", self.retrieval_nodes.retrieve_relevant_docs
        )
        self.graph.add_node(
            "generate_rag_answer", self.generation_nodes.generate_rag_answer
        )
        self.graph.add_node(
            "generate_normal_answer", self.generation_nodes.generate_normal_answer
        )
        self.graph.add_node(
            "reshape_user_query_to_situation",
            self.retrieval_nodes.reshape_user_query_to_situation,
        )

        self.graph.add_edge(START, "detect_language")
        self.graph.add_conditional_edges(
            "detect_language",
            self.routing_nodes.decide_to_translate,
            ["translate_into_english", "emotion_classifier"],
        )
        self.graph.add_edge("translate_into_english", "emotion_classifier")
        self.graph.add_edge("emotion_classifier", "intent_classifier")
        self.graph.add_conditional_edges(
            "intent_classifier",
            self.routing_nodes.decide_intent_path,
            ["reshape_user_query_to_situation", "generate_normal_answer"],
        )
        self.graph.add_edge("reshape_user_query_to_situation", "retrieve_relevant_docs")
        self.graph.add_edge("generate_normal_answer", END)
        self.graph.add_edge("retrieve_relevant_docs", "generate_rag_answer")
        self.graph.add_edge("generate_rag_answer", END)

        self.agent = self.graph.compile(checkpointer=self.checkpoint_saver)

    def get_agent(self):
        return self.agent

    def close(self):
        self.retrieval_nodes.close()


if __name__ == "__main__":
    rag_graph = RagGraph()
    agent = rag_graph.get_agent()

    # Testing the agent with a sample state

    for output in agent.stream(
        {
            "question": "I can't sleep and feel anxious all the time. What should I do?",
        },
        stream_mode=["updates", "messages"],
    ):

        typee, content = output

        if typee == "updates":
            for node_name, state in content.items():
                print(f"Node: {node_name}")
        else:
            message, meta_data = content
            if meta_data.get("langgraph_node") in {
                "generate_rag_answer",
                "generate_normal_answer",
            }:
                print(message.content, end="")

    rag_graph.close()
