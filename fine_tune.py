import json
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import os
from dotenv import load_dotenv
import wandb

# Load environment variables
load_dotenv()

# Set wandb API key from environment variable
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_AI_API_KEY")

def load_training_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['data']

def prepare_dataset(training_data):
    # Convert to the format expected by the model
    processed_data = []
    for item in training_data:
        processed_data.append({
            'text': item['instruction'],
            'summary': item['output']
        })
    return Dataset.from_list(processed_data)

def main():
    # Load model and tokenizer
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load and prepare dataset
    training_data = load_training_data('data/fine_tuning_data.json')
    dataset = prepare_dataset(training_data)

    # Tokenize dataset
    def preprocess_function(examples):
        inputs = examples['text']
        targets = examples['summary']
        
        model_inputs = tokenizer(
            inputs, 
            max_length=128, 
            truncation=True, 
            padding="max_length"
        )
        
        labels = tokenizer(
            targets, 
            max_length=128, 
            truncation=True, 
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        push_to_hub=True,
        hub_model_id="Mozes721/crypto-stock-weather-agent",
        hub_token=os.getenv("HF_ACCESS_TOKEN"),
        report_to="wandb"
    )

    # Initialize wandb
    wandb.init(
        project="crypto-stock-weather-agent",
        name="flan-t5-small-finetuning-optimized",
        config={
            "model_name": "flan-t5-small",
            "learning_rate": 2e-5,
            "batch_size": 4,
            "epochs": 3,
            "weight_decay": 0.01
        }
    )

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train the model
    trainer.train()

    # Push to hub
    trainer.push_to_hub()

if __name__ == "__main__":
    main() 