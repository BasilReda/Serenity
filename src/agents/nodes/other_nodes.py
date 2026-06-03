from agents.state import RagState
from classifiers.emotion_classifier import EmotionClassifier
from classifiers.intent_classifier import IntentClassifier
from classifiers.language_identification import LanguageIdentifier
from store import LLM


class OtherNodes:
    def __init__(self):

        self.language_detector = LanguageIdentifier()
        self.emotion_classifier_pipeline = EmotionClassifier()
        self.llm = LLM().get_model()
        self.intent_classifier_model = IntentClassifier()

    def detect_language(self, state: RagState):
        question = state["question"]

        language = self.language_detector.predict_language(question)

        state["language"] = language
        return state

    def emotion_classifier(self, state: RagState):
        question = (
            state.get("translated_question")
            if state.get("translated_question")
            else state["question"]
        )

        emotion, score = self.emotion_classifier_pipeline.classify_emotion(question)

        state["emotion"] = emotion
        state["score"] = score

        return state

    def translate_into_english(self, state: RagState):
        question = state["question"]

        translation_prompt = f"Translate the following text into English:\n\n{question}"
        translation_response = self.llm.invoke(translation_prompt).content.strip()

        state["translated_question"] = translation_response

        return state

    def intent_classifier(self, state: RagState):
        question = (
            state.get("translated_question")
            if state.get("translated_question")
            else state["question"]
        )

        intent = self.intent_classifier_model.classify_intent(question)
        state["intent"] = intent

        return state


if __name__ == "__main__":
    other_nodes = OtherNodes()

    test_state = RagState(
        history=[],
        question="I am feeling really anxious and stressed out.",
        translated_question=None,
        retrieved_docs=None,
        answer=None,
        language=None,
        emotion=None,
        score=None,
        intent=None,
    )
    test_state = other_nodes.detect_language(test_state)
    print(f"Detected language: {test_state['language']}")
    test_state = other_nodes.emotion_classifier(test_state)
    print(
        f"Detected emotion: {test_state['emotion']} with score {test_state['score']:.4f}"
    )
    test_state = other_nodes.translate_into_english(test_state)
    print(f"Translated question: {test_state['translated_question']}")
    test_state = other_nodes.intent_classifier(test_state)
    print(f"Detected intent: {test_state['intent']}")
