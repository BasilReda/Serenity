import requests

def get_emotion_api(text: str) -> str:
    """
    Calls a Hugging Face Inference API for a Transformer/RNN emotion classifier.
    
    Args:
        text (str): The raw user input.
        
    Returns:
        str: The primary detected emotion (e.g., 'anxiety', 'sadness', 'neutral').
    """
    # TODO: Send text to HF API and parse the highest scoring emotion label
    pass

def emotion_node(state: dict) -> dict:
    """
    LangGraph Node: Reads user input, predicts emotion, and updates state.
    """
    user_input = state.get("user_input", "")
    
    emotion = get_emotion_api(user_input)
    
    return {"detected_emotion": emotion}