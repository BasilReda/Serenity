from sentence_transformers import CrossEncoder
from core.config import settings

_model = None

def _get_model():
    global _model
    if _model is None:
        print("[RAG] Loading CrossEncoder reranker...")
        _model = CrossEncoder(settings.CROSS_ENCODER)
    return _model

def rerank(documents: list[str], query: str, top_k: int = None) -> list[str]:
    if not documents:
        return []
    model  = _get_model()
    pairs  = [[query, doc] for doc in documents]
    scores = model.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k or settings.RERANKER_TOP_K]]
