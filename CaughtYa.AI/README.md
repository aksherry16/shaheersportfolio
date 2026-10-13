# CaughtYa.AI 🕵️‍♂️

A simple Python tool to detect AI-generated text using the `roberta-base-openai-detector` model.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download NLTK Data** (first time only):
    The script will attempt to download necessary NLTK data automatically.

## Usage

### Run on default sample text
```bash
python caughtya.ai.py
```

### Run on specific text
```bash
python caughtya.ai.py --text "Check this text for me."
```

### Run on a file
```bash
python caughtya.ai.py --file my_article.txt
```
