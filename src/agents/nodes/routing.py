from agents.state import RagState


class RoutingNodes:
    def __init__(self):
        pass

    def decide_to_translate(self, state: RagState):
        language = state["language"]

        if language != "English":
            return "translate_into_english"
        else:
            return "emotion_classifier"

    def decide_intent_path(self, state: RagState):
        intent = state["intent"]

        if intent == "asking_mental_health_question":
            return "reshape_user_query_to_situation"
        else:
            return "generate_normal_answer"


if __name__ == "__main__":

    # testing the routing functions with a sample state
    state = RagState(
        history=[],
        question="What is the best way to deal with anxiety?",
        translated_question=None,
        retrieved_docs=None,
        answer=None,
        language="English",
        emotion="anxiety",
        score=0.8,
        intent="asking_mental_health_question",
    )

    routing_nodes = RoutingNodes()
    print(routing_nodes.decide_to_translate(state))
    print(routing_nodes.decide_intent_path(state))
