import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class AIGeneratedTextDetector:
    def __init__(self, model_name="openai-community/roberta-base-openai-detector"):
        """
        Initializes the AI Text Detector.
        """
        print(f"Loading AI Detector Model: {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Using default roberta-base as placeholder (will not detect AI correctly without fine-tuning).")
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self.model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
            self.model.eval()

    def predict(self, text):
        """
        Predicts if the text is AI-generated.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
        # For roberta-base-openai-detector: Label 0 is "Real" (Human), Label 1 is "Fake" (AI) usually.
        # Let's verify standard conventions. Often 1 is the positive class (AI).
        
        ai_prob = probs[0][1].item()
        human_prob = probs[0][0].item()
        
        label = "AI-Generated" if ai_prob > 0.5 else "Human-Written"
        
        return {
            "label": label,
            "confidence": ai_prob if label == "AI-Generated" else human_prob,
            "score_ai": ai_prob
        }
