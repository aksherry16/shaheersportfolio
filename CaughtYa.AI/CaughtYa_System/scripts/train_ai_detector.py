import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

def train_ai_detector():
    print("Initializing AI Detector Training (Fine-tuning GPT-2/RoBERTa)...")
    
    # Note: For AI detection, usually we use a binary classifier (Real vs AI). 
    # The user example used GPT2LMHeadModel but that is for generation. 
    # For classification "Human vs AI", SequenceClassification is better.
    # We will stick to the user's request context but adapt for classification task.
    
    # 1. Load Data
    # Placeholder: 'openai_humaneval' is code, but we need text. 
    # Using 'imdb' as a dummy placeholder for structure to ensure script runs if swapped.
    try:
        dataset = load_dataset('imdb', split='train[:1%]') # TINY subset for demo
        print("Dummy dataset loaded (IMDB). Replace with Human vs AI dataset.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Model & Tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # GPT2 needs pad token
    
    model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id

    # 3. Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Train/Test Split
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)

    # 4. Arguments
    training_args = TrainingArguments(
        output_dir='./results_ai_detect',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="steps",
        eval_steps=10,
        logging_steps=10,
        no_cuda=not torch.cuda.is_available()
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test']
    )

    print("Starting Training...")
    trainer.train()

    save_path = "../models/saved_ai_detector_gpt2"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_ai_detector()
