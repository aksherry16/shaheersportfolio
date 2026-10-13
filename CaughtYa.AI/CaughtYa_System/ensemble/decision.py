from ..models.fake_news import FakeNewsClassifier
from ..models.ai_detector import AIGeneratedTextDetector
from ..models.consistency import InconsistencyDetector

class EnsembleModel:
    def __init__(self):
        print("Initializing Ensemble Model System...")
        self.fake_news_model = FakeNewsClassifier()
        self.ai_detector = AIGeneratedTextDetector()
        self.consistency_model = InconsistencyDetector()
        print("Ensemble System Ready.")

    def analyze(self, text):
        """
        Runs all models and aggregates results.
        """
        # 1. Fake News Detection
        fn_result = self.fake_news_model.predict(text)
        
        # 2. AI Detection
        ai_result = self.ai_detector.predict(text)
        
        # 3. Consistency check
        consistency_result = self.consistency_model.detect_inconsistency(text)
        
        # Final Decision Logic
        # - If Fake Probability is high OR Inconsistency is high, flag as potentially unreliable.
        # - If AI Probability is high, mark as AI-generated.
        
        flags = []
        if fn_result['label'] == "Fake News" and fn_result['confidence'] > 0.7:
            flags.append("High likelihood of Fake News")
        if ai_result['label'] == "AI-Generated" and ai_result['score_ai'] > 0.8:
            flags.append("Likely AI-Generated")
        if consistency_result['inconsistency_score'] > 0.4:
            flags.append("High semantic inconsistency detected")
            
        final_verdict = "Clear"
        if flags:
            final_verdict = "Flagged: " + "; ".join(flags)
        
        return {
            "final_verdict": final_verdict,
            "fake_news_analysis": fn_result,
            "ai_detection_analysis": ai_result,
            "consistency_analysis": consistency_result
        }
