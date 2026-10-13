import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.linear_model import LogisticRegression
import joblib
from src.config import MODEL_NAME, MODEL_SAVE_PATH, MAX_LENGTH


class AIContentDetector:

    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
        self.model = RobertaModel.from_pretrained(MODEL_NAME)
        self.classifier = LogisticRegression(max_iter=200)

    # Convert text to embeddings using RoBERTa
    def extract_features(self, texts):
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings

    # Train the model
    def train(self, X_train, y_train):
        X_features = self.extract_features(X_train)
        self.classifier.fit(X_features, y_train)

    # Predict class (AI or Human)
    def predict(self, texts):
        X_features = self.extract_features(texts)
        return self.classifier.predict(X_features)

    # Save the model
    def save_model(self):
        joblib.dump(self.classifier, MODEL_SAVE_PATH)

    # Load the model
    def load_model(self):
        self.classifier = joblib.load(MODEL_SAVE_PATH)
