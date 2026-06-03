import joblib


class LanguageIdentifier:
    def __init__(self):
        self.model = joblib.load(r"trained_models\language_identification_model.pkl")

    def get_model(self):
        return self.model

    def predict_language(self, text):
        language = self.model.predict([text])[0]

        return self.map_language_code_to_name(language)

    def map_language_code_to_name(self, code):
        language_mapping = {
            "ar": "Arabic",
            "bg": "Bulgarian",
            "de": "German",
            "el": "Modern Greek",
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "hi": "Hindi",
            "it": "Italian",
            "ja": "Japanese",
            "nl": "Dutch",
            "pl": "Polish",
            "pt": "Portuguese",
            "ru": "Russian",
            "sw": "Swahili",
            "th": "Thai",
            "tr": "Turkish",
            "ur": "Urdu",
            "vi": "Vietnamese",
            "zh": "Chinese",
        }
        return language_mapping.get(code, "Unknown")


if __name__ == "__main__":
    language_identifier = LanguageIdentifier()
    test_text = "How are you"
    predicted_language = language_identifier.predict_language(test_text)
    print(f"Predicted language: {predicted_language}")
    print("Expected language: Spanish")
