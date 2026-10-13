import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import os

def train_fake_news_model():
    print("Initializing Fake News Training...")
    
    # 1. Load Dataset
    # Replace 'fake_news_dataset' with your actual dataset path or name from Hugging Face Hub
    # Expected format: {'text': ..., 'label': ...}
    try:
        dataset = load_dataset('liar') # Example dataset, replace with user's
        print("Dataset loaded.")
    except Exception as e:
        print(f"Could not load default dataset: {e}")
        print("Please configure specific dataset path in script.")
        return

    # 2. Tokenizer & Model
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 3. Preprocess
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128) # Short max_length for demo speed

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Simple split if not present (Liar has splits)
    if 'train' not in tokenized_datasets:
        tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir='./results_fake_news',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        save_strategy="epoch",
        no_cuda=not torch.cuda.is_available() # Use CPU if no GPU
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'] if 'test' in tokenized_datasets else tokenized_datasets['validation']
    )

    print("Starting Training...")
    trainer.train()
    
    # 6. Save Model
    save_path = "../models/saved_fake_news_roberta"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_fake_news_model()
