from transformers import pipeline


class EmotionClassifier:
    def __init__(self):
        self.emotion_pipeline = pipeline(
            "text-classification",
            model=r"trained_models\emotion_classifier_model",
            tokenizer=r"trained_models\emotion_classifier_model",
            return_all_scores=False,
        )

    def classify_emotion(self, text):
        result = self.emotion_pipeline(text)[0]
        label = result["label"]
        score = result["score"]

        return label, score


if __name__ == "__main__":
    emotion_classifier = EmotionClassifier()
    test_text = "I am feeling really anxious and stressed out."
    emotion, score = emotion_classifier.classify_emotion(test_text)
    print(f"Predicted emotion: {emotion} with score {score:.4f}")
    print("Expected emotion: anxiety or stress or fear")
