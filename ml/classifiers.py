from transformers import pipeline as hf_pipeline
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from core.config import settings
from typing import Dict, Any
from transformers import AutoTokenizer
from .emotion import EmotionPredict,EmotionModel
import torch

_lang_clf    = None

def _get_emotion_clf():

    emotion_tokenizer = AutoTokenizer.from_pretrained(settings.BERT_EMOTION_MODEL)
    emotion_model     = EmotionModel(model_name = settings.BERT_EMOTION_MODEL)
    emotion_checkpoint = torch.load(settings.EMOTION_MODEL, map_location = settings.DEVICE, weights_only=False)
    emotion_model.load_state_dict(emotion_checkpoint["model_state_dict"])

    return emotion_model, emotion_tokenizer

def _get_lang_clf():
    global _lang_clf
    if _lang_clf is None:
        print("[ML] Loading language classifier...")
        _lang_clf = hf_pipeline("text-classification",
                                 model=settings.LANGUAGE_MODEL,
                                 return_all_scores=False)
    return _lang_clf

def detect_emotion(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"emotion": "neutral", "score": 1.0}
    emotion_model, emotion_tokenizer = _get_emotion_clf()
    predictor = EmotionPredict(emotion_model, emotion_tokenizer, device=settings.DEVICE)
    result = predictor(text)
    return {"emotion": result["label"], "score": result["confidence"]}

def detect_language(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"language": "unknown", "score": 1.0}
    result = _get_lang_clf()(text)[0]
    return {"language": result["label"], "score": round(result["score"], 4)}

class _IntentSchema(BaseModel):
    intent:     str   = Field(..., description="Detected intent.")
    confidence: float = Field(..., description="Confidence 0-1.")

_intent_parser = PydanticOutputParser(pydantic_object=_IntentSchema)
_intent_prompt = ChatPromptTemplate.from_messages([
    ("system", """Classify the query into EXACTLY ONE intent:
- "greeting"
- "goodbye"
- "gratitude"
- "asking_mental_health_question"
- "out_of_scope"
{format_instructions}
OUTPUT STRICT JSON ONLY. No extra text.
Example: {{"intent": "greeting", "confidence": 0.98}}"""),
    ("human", "{text}"),
])

def intent_user(text: str) -> dict:
    from ml.llm import get_global_llm
    chain = (_intent_prompt | get_global_llm() | _intent_parser).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True)
    try:
        result = chain.invoke({
            "text": text,
            "format_instructions": _intent_parser.get_format_instructions()})
        return {"intent": result.intent, "confidence": result.confidence}
    except Exception as e:
        print(f"[INTENT] Parse failed: {e}")
        return {"intent": "out_of_scope", "confidence": 0.0}