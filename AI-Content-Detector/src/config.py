import os

# ============ PROJECT PATHS ============

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_PATH = os.path.join(BASE_DIR, "..", "data", "raw")
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, "..", "data", "processed")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "models", "content_detector_model.pkl")

# ============ MODEL SETTINGS ============

MODEL_NAME = "roberta-base"   # You can change later
TEST_SIZE = 0.2               # 20% dataset for testing
RANDOM_STATE = 42             # To reproduce results
MAX_LENGTH = 256              # Max tokens for model input
BATCH_SIZE = 16
EPOCHS = 3
