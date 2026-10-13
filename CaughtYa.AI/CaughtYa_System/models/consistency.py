from sentence_transformers import SentenceTransformer, util
import nltk

class InconsistencyDetector:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes SBERT model for semantic similarity.
        """
        print(f"Loading Inconsistency Detector: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self._ensure_nltk()

    def _ensure_nltk(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

    def detect_inconsistency(self, text):
        """
        Splits text into paragraphs/sentences and checks for low cosine similarity 
        which might indicate inconsistency.
        """
        # Split by paragraphs logic or sentences. 
        # For longer articles, paragraphs are better.
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return {
                "inconsistency_score": 0.0,
                "dataset_size": len(paragraphs),
                "details": "Not enough paragraphs to compare."
            }

        embeddings = self.model.encode(paragraphs, convert_to_tensor=True)
        
        # Calculate cosine similarity between adjacent paragraphs
        inconsistencies = []
        min_similarity = 1.0
        
        for i in range(len(paragraphs) - 1):
            sim = util.pytorch_cos_sim(embeddings[i], embeddings[i+1]).item()
            if sim < min_similarity:
                min_similarity = sim
            
            # Arbitrary threshold: if similarity is very low, it might be a disjointed idea or contradiction.
            # In real news, flow should be smooth (high similarity or logical connection).
            # Low similarity doesn't always mean contradiction, but it's a proxy for 'inconsistency'.
            if sim < 0.2: 
                inconsistencies.append({
                    "para_a": paragraphs[i][:50] + "...",
                    "para_b": paragraphs[i+1][:50] + "...",
                    "similarity": sim
                })

        # Inconsistency score: 1.0 means very consistent (high min_sim), 0.0 means inconsistent.
        # We can inverse it: high score = high inconsistency.
        # Let's say Score = 1 - min_similarity
        score = 1.0 - min_similarity if min_similarity < 1.0 else 0.0
        
        return {
            "inconsistency_score": score,
            "min_similarity": min_similarity,
            "details": inconsistencies
        }
