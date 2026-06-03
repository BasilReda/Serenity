import json
from agents.state import RagState
from store import LLM


class IntentClassifier:
    def __init__(self):
        self.llm = LLM().get_model()

    def classify_intent(self, question):

        intent_prompt = self.get_intent_classifier_prompt(question)
        intent_response = self.llm.invoke(intent_prompt).content.strip()
        intent = json.loads(intent_response)["intent"]

        return intent

    def get_intent_classifier_prompt(self, user_question):
        prompt = f"""
                You are an expert Intent Classifier for a Mental Health Chatbot system.
                Your job is to classify the user's input into EXACTLY ONE of the following intents:
                1. greeting (If the user is saying hi, hello, or greeting in any language)
                2. goodbye (If the user is saying goodbye, bye, or leaving)
                3. gratitude (If the user is thanking you or expressing appreciation)
                4. asking_mental_health_question (If the user is asking about anxiety, depression, stress, therapy, or any mental health topic)
                5. out_of_scope (If the user is asking about general knowledge, coding, math, weather, or anything unrelated to mental health or greetings)

                Strict Rules:
                - Output ONLY a valid JSON object with the key "intent".
                - Do NOT include any explanations, introduction, or conversational text.
                - Rely on the examples below.

                ### Examples:
                User: "Hello there, hope you are doing well"
                Output: {{"intent": "greeting"}}

                User: "Thanks for your amazing support!"
                Output: {{"intent": "gratitude"}}

                User: "How can I manage panic attacks during exams?"
                Output: {{"intent":     "asking_mental_health_question"}}

                User: "Can you help me? I feel so sad and frustrated."
                Output: {{"intent": "asking_mental_health_question"}}

                User: "What is the recipe for making chocolate cake?"
                Output: {{"intent": "out_of_scope"}}

                User: "Bye bye, see you tomorrow"
                Output: {{"intent": "goodbye"}}

                ### Current Input to Classify:
                User: "{user_question}"
                Output:"""

        return prompt.strip()


if __name__ == "__main__":
    # Testing the intent classifier with sample questions
    intent_classifier = IntentClassifier()

    test_questions = [
        "Hello there, hope you are doing well",
        "Thanks for your amazing support!",
        "How can I manage panic attacks during exams?",
        "Can you help me? I feel so sad and frustrated.",
        "What is the recipe for making chocolate cake?",
        "Bye bye, see you tomorrow",
    ]

    for question in test_questions:
        intent = intent_classifier.classify_intent(question)
        print(f"Question: {question}\nClassified Intent: {intent}\n")
