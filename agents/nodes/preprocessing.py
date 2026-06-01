from agents.state import ChatbotState
from ml.classifiers import detect_language, detect_emotion
from ml.translator import translate_to_english

LANGUAGE_MAP = {
    "ar":"Arabic","bg":"Bulgarian","de":"German","el":"Modern Greek",
    "en":"English","es":"Spanish","fr":"French","hi":"Hindi",
    "it":"Italian","ja":"Japanese","nl":"Dutch","pl":"Polish",
    "pt":"Portuguese","ru":"Russian","sw":"Swahili","th":"Thai",
    "tr":"Turkish","ur":"Urdu","vi":"Vietnamese","zh":"Chinese",
}

def language_detection_node(state: ChatbotState) -> dict:
    print("--> [NODE] Executing: language_detection_node")
    user_input = state["user_input"]
    info       = detect_language(user_input)
    code, conf = info["language"], info["score"]
    words      = len(user_input.strip().split())
    if words <= 2 or conf < 0.85:
        lang = "English"
        msg  = f"Defaulting to English — words: {words}, confidence: {conf:.2f}"
    else:
        lang = LANGUAGE_MAP.get(code, "English")
        msg  = f"Detected Language: {lang} ({code}), Confidence: {conf:.2f}"
    return {"detect_language": lang, "status_update": [msg]}

def is_language_english(state: ChatbotState) -> str:
    print("--> [EDGE] Executing: is_language_english")
    return "english" if state.get("detect_language","").strip().lower() == "english" else "not english"

def translate_to_english_node(state: ChatbotState) -> dict:
    print("--> [NODE] Executing: translate_to_english_node")
    if state.get("detect_language","").strip().lower() != "english":
        translated = translate_to_english(state["user_input"])
        return {"translated_query": translated, "status_update": [f"Translated: {translated}"]}
    return {"translated_query": state["user_input"]}

def emotion_detection_node(state: ChatbotState) -> dict:
    print("--> [NODE] Executing: emotion_detection_node")
    text = state.get("translated_query", state["user_input"])
    info = detect_emotion(text)
    return {
        "detected_emotion": info["emotion"],
        "status_update": [f"Detected Emotion: {info['emotion']}, Score: {info['score']}"],
    }
