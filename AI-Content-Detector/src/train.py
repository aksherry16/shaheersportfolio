from src.dataset_loader import load_dataset, preprocess_dataset, split_data
from src.detector import AIContentDetector
from src.config import MODEL_SAVE_PATH

def train_model():
    print("\n=== STEP 1: Loading Dataset ===")
    df = load_dataset("dataset.csv")   # <-- your file must be in data/raw/dataset.csv
    print("Dataset Loaded. Total samples:", len(df))

    print("\n=== STEP 2: Preprocessing Text ===")
    df = preprocess_dataset(df)

    print("\n=== STEP 3: Splitting Train/Test Data ===")
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n=== STEP 4: Initializing Model ===")
    detector = AIContentDetector()

    print("\n=== STEP 5: Training Model (This may take time) ===")
    detector.train(X_train, y_train)

    print("\n=== STEP 6: Saving Model ===")
    detector.save_model()
    print(f"Model saved at: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
