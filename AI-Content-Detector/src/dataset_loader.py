import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_RAW_PATH, DATA_PROCESSED_PATH, TEST_SIZE, RANDOM_STATE

# -------- Load & Prepare Dataset -------- #
def load_dataset(filename):
    """
    Loads CSV dataset from data/raw/
    Columns required:
        text -> content text
        label -> 1 for AI-generated, 0 for Human-written
    """
    filepath = os.path.join(DATA_RAW_PATH, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    return df


# -------- Text Cleaning -------- #
def preprocess_text(text):
    """
    Removes unwanted characters, formatting & extra spaces.
    """
    import re
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove links
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)           # remove symbols
    text = " ".join(text.split())                       # normalize spaces
    return text


# -------- Apply Preprocessing -------- #
def preprocess_dataset(df):
    df["text"] = df["text"].apply(preprocess_text)
    return df


# -------- Split Data for Training & Testing -------- #
def split_data(df):
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test
