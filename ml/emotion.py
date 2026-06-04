"""
Emotion Detection Module for Serenity
This module provides functionality for detecting emotions in text using a pre-trained BERT-based model. It includes:
- EmotionTextPreprocessor: A class for preprocessing text data, including cleaning and normalization.
- EmotionModel: A PyTorch neural network model that utilizes a BERT architecture for emotion classification.
- EmotionPredict: A class that wraps the EmotionModel and provides a method for making predictions on input text.
"""
import torch
from transformers import AutoModel
import torch.nn as nn
import re
import contractions

class EmotionTextPreprocessor:
    def preprocess_text(self, text):

        text = re.sub(r"\s+", " ", text).strip()

        text =re.sub(r"http\S+|www\S+|https\S+", '', text)

        text = re.sub(r"@\w+", "", text)
        
        text = contractions.fix(text)

        return text
    def transform(self, X):
        return self.preprocess_text(X)
    

class EmotionModel(nn.Module):
    def __init__(self, model_name, output_dim=6, dropout=0.3):
        super(EmotionModel, self).__init__()
        self.bert    = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(self.bert.config.hidden_size, output_dim)
 
    def forward(self, input_ids, attention_mask):
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  
        cls_output = self.dropout(cls_output)
        return self.fc(cls_output) 
    
class EmotionPredict:
    def __init__(self, model, tokenizer, device="cuda"):
        self.emotion_model     = model.to(device)
        self.tokenizer         = tokenizer
        self.device            = device
        self.preprocessor      = EmotionTextPreprocessor() 
        
        self.index2label = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
            }

    def predict(self, text):
        self.emotion_model.eval()
        self.predict
        text = self.preprocessor.transform(text)
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        input_ids      = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.emotion_model(input_ids, attention_mask)

        probs     = torch.softmax(logits, dim=-1).squeeze()
        label_idx = int(probs.argmax())

        return {
            "label":      self.index2label[label_idx],
            "confidence": f"{probs[label_idx]*100:.1f}%"
        }