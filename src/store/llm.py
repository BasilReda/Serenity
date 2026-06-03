from langchain_groq import ChatGroq
from helpers.config import get_settings


class LLM:
    def __init__(self):
        self.settings = get_settings()
        self.model = self.llm = ChatGroq(
            api_key=self.settings.GROQ_API_KEY,
            model=self.settings.LLM_MODEL_NAME,
            temperature=0,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
            streaming=True,
        )

    def get_model(self) -> ChatGroq:

        return self.model
