from transformers import MarianTokenizer, MarianMTModel
from core.config import settings
import torch

_tokenizer = None
_model     = None
_device    = "cuda" if torch.cuda.is_available() else "cpu"

def _load():
    global _tokenizer, _model
    if _tokenizer is None:
        print("[ML] Loading translator model...")
        _tokenizer = MarianTokenizer.from_pretrained(settings.TRANSLATOR_MODEL)
        _model     = MarianMTModel.from_pretrained(settings.TRANSLATOR_MODEL).to(_device)
    return _tokenizer, _model

def translate_to_english(text: str) -> str:
    tokenizer, model = _load()
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(_device)
    tokens = model.generate(**inputs)
    return tokenizer.decode(tokens[0], skip_special_tokens=True)