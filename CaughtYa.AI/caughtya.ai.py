import sys
import argparse
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class AIDetector:
    def __init__(self, model_name="roberta-base-openai-detector"):
        print(f"⏳ Loading model '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._ensure_nltk()
        print("✅ Model loaded successfully.\n")

    def _ensure_nltk(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)

    def detect(self, text):
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return 0.0, []

        ai_scores = []
        for sent in sentences:
            # tokenize and predict
            inputs = self.tokenizer(sent, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # probability that the text is AI-generated (class 1)
                score = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][1].item() * 100
            ai_scores.append((sent, score))

        # overall percentage
        avg_ai = sum(s for _, s in ai_scores) / len(ai_scores)
        return avg_ai, ai_scores

def print_report(avg_ai, ai_scores):
    print(f"🧠 Overall AI Probability: {avg_ai:.2f}%\n")
    print("-" * 70)
    for sent, score in ai_scores:
        label = "🤖 AI-like" if score > 60 else "🧍 Human-like"
        bar = "█" * int(score / 5)
        print(f"💬 {sent}")
        print(f" → {label} ({score:.1f}%)  {bar}\n")
    print("-" * 70)

def main():
    parser = argparse.ArgumentParser(description="Detect AI-generated text using RoBERTa.")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="Path to a text file to analyze")
    args = parser.parse_args()

    text_to_analyze = ""

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text_to_analyze = f.read()
        except FileNotFoundError:
            print(f"❌ Error: File '{args.file}' not found.")
            return
    elif args.text:
        text_to_analyze = args.text
    else:
        # Default sample text if no args provided
        print("ℹ️  No input provided. Running on sample text...\n")
        text_to_analyze = """
The rapid advancements of artificial intelligence have completely transformed the modern world. From healthcare to education, AI systems are improving efficiency, accuracy, and accessibility on a global scale. The integration of intelligent algorithms into daily life has created new opportunities for innovation and development. As technology continues to evolve, the boundaries between human creativity and machine intelligence are becoming increasingly blurred, shaping a future where collaboration between humans and machines defines progress.
"""

    if not text_to_analyze.strip():
        print("❌ Error: Input text is empty.")
        return

    detector = AIDetector()
    print("🔎 Analyzing text...\n")
    avg_score, details = detector.detect(text_to_analyze)
    print_report(avg_score, details)

if __name__ == "__main__":
    main()
