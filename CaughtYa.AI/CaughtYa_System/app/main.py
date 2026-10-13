from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add parent directory to path to allow imports from sibling packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble.decision import EnsembleModel

app = FastAPI(title="CaughtYa Fake News Detection API")

# Initialize model globally to avoid reloading on every request
# In production, use lifespan events.
try:
    ensemble = EnsembleModel()
except Exception as e:
    print(f"Failed to initialize models: {e}")
    ensemble = None

class AnalysisRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Welcome to CaughtYa API. Use /predict to analyze text."}

@app.post("/predict")
def predict(request: AnalysisRequest):
    if not ensemble:
        raise HTTPException(status_code=500, detail="Models not initialized properly.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided.")
        
    result = ensemble.analyze(request.text)
    return result
