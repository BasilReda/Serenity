from agents.state import RagState
from store import QdrantStore
from store import LLM


class RetrievalNodes:
    def __init__(self):
        self.qdrant = QdrantStore().return_qdrant()
        self.llm = LLM().get_model()

    def close(self):
        self.qdrant.client.close()

    def retrieve_relevant_docs(self, state: RagState):
        if state.get("hyde_context"):
            search_query = state["hyde_context"]
            print(
                "🧠 [Retrieval] Searching Qdrant using the expanded narrative (Reverse-HyDE)..."
            )
        elif state.get("translated_question"):
            search_query = state["translated_question"]
            print(
                "🌐 [Retrieval] Searching Qdrant using English translated question..."
            )
        else:
            search_query = state["question"]
            print("📝 [Retrieval] Searching Qdrant using original user question...")

        retrieved_docs = self.qdrant.similarity_search(search_query, k=3)

        state["retrieved_docs"] = retrieved_docs

        return state

    def reshape_user_query_to_situation(self, state: RagState):

        question = (
            state.get("translated_question")
            if state.get("translated_question")
            else state["question"]
        )
        emotion = state["emotion"]

        story_prompt = f"""
        You are an expert in clinical psychology. Your task is to take a brief user query and expand it into a realistic, detailed, first-person mental health narrative (a patient venting or sharing their situation).
        The narrative must sound like a real person opening up about their struggles, capturing the underlying emotional distress, lifestyle impact (like sleep or motivation), and thoughts leading up to that question.

        Strict Rules:
        - Write in the FIRST PERSON ("I", "My").
        - Do NOT include any introductions, diagnoses, bullet points, or doctor advice.
        - End the narrative with the user's original question or a very similar emotional realization.
        - The detected user emotion is: {emotion}.
        - Make it in max length of 100 words.

        User's Brief Query: "{question}"
        
        Expanded First-Person Narrative:
        """

        story_response = self.llm.invoke(story_prompt).content.strip()

        state["hyde_context"] = story_response
        return state


if __name__ == "__main__":
    # Testing the retrieval nodes with a sample state
    retrieval_nodes = RetrievalNodes()
    test_state = RagState(
        question="I can't sleep and feel anxious all the time. What should I do?",
        language="English",
        emotion="anxiety",
    )
    test_state = retrieval_nodes.reshape_user_query_to_situation(test_state)
    print(test_state["hyde_context"])
    # test_state = retrieval_node.retrieve_relevant_docs(test_state)
    # print(test_state["retrieved_docs"])
    retrieval_nodes.close()
