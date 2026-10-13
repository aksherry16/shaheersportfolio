import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

class FakeNewsClassifier:
    def __init__(self, model_name='roberta-base'):
        """
        Initializes the Fake News Classifier.
        In a real scenario, model_name would be the path to the fine-tuned model.
        """
        print(f"Loading Fake News Model: {model_name}...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        # num_labels=2 for Fake vs Real
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.eval()
    
    def predict(self, text):
        """
        Predicts whether the text is Real or Fake.
        Returns a dictionary with label and confidence score.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
        # Assuming label 0 is "Fake" and label 1 is "Real" (or vice versa, adjustable based on training)
        # Let's assume 0: Real, 1: Fake for now, or commonly 0: Fake, 1: Real. 
        # We will standardize this: Index 1 = Fake, Index 0 = Real for this example logic, 
        # but usually datasets define this. Let's return probabilities for both.
        
        fake_prob = probs[0][1].item() # Hypothetical 'Fake' class
        real_prob = probs[0][0].item() # Hypothetical 'Real' class
        
        label = "Fake News" if fake_prob > real_prob else "Real News"
        confidence = max(fake_prob, real_prob)
        
        return {
            "label": label,
            "confidence": confidence,
            "scores": {"fake": fake_prob, "real": real_prob}
        }
