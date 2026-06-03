from langchain_cohere import CohereEmbeddings
from helpers.config import get_settings


class Embeddings:
    def __init__(self):
        self.settings = get_settings()
        self.model = self.llm = CohereEmbeddings(
            cohere_api_key=self.settings.COHERE_API_KEY,
            model=self.settings.EMBEDDING_MODEL_NAME,
        )

    def get_model(self) -> CohereEmbeddings:

        return self.model
