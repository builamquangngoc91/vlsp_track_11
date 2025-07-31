

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import transformers

# --- 1. Configuration ---

# Model and tokenizer names
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

# Input dataset
DATASET_PATH = "./training_data_with_context.json"

# Output directory for the fine-tuned model
OUTPUT_DIR = "./qwen2-finetuned-model"

# --- 2. Data Loading and Preparation ---

# Load the JSON file manually
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Get all unique keys from all items to ensure consistency
all_keys = set()
for item in data:
    all_keys.update(item.keys())

# Aggressively convert all values to strings to prevent any type inference errors
stringified_data = []
for item in data:
    new_item = {}
    for key in all_keys:
        value = item.get(key)
        if isinstance(value, (dict, list)):
            new_item[key] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            new_item[key] = ""  # Convert None to empty string
        else:
            new_item[key] = str(value)
    stringified_data.append(new_item)

# Convert the list of dictionaries to a dictionary of lists
data_dict = {key: [d[key] for d in stringified_data] for key in all_keys}

# Create a Dataset object from the dictionary
dataset = Dataset.from_dict(data_dict)

# --- 3. Model and Tokenizer Loading ---

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load the model without quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# --- 4. Data Preprocessing ---

def format_prompt(sample):
    """
    Formats a data sample into a prompt for the model.
    """
    # Parse the JSON strings back into Python objects
    relevant_articles = json.loads(sample['relevant_articles_content'])
    
    question = sample['question']
    
    # Format choices for multiple choice questions
    if sample['question_type'] == 'Multiple choice':
        choices_dict = json.loads(sample['choices'])
        if choices_dict:
            choices_text = "
".join([f"{key}: {value}" for key, value in choices_dict.items()])
            question = f"{question}
{choices_text}"
        
    answer = sample['answer']
    
    # Combine the text of all relevant articles into a single context string
    context = "

".join([article['article_text'] for article in relevant_articles])
    
    # Create the prompt using the Qwen-Instruct chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on Vietnamese traffic law."},
        {"role": "user", "content": f"Based on the following legal articles, please answer the question.

--- BEGIN CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}"},
        {"role": "assistant", "content": answer}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    return {"text": prompt}

# Apply the formatting to the entire dataset
processed_dataset = dataset.map(format_prompt, remove_columns=['id', 'image_id', 'question', 'question_type', 'choices', 'answer', 'relevant_articles_content'])


# --- 4.5 Tokenization ---

def tokenize_function(examples):
    """Tokenizes the text column and creates labels."""
    tokenized_output = tokenizer(
        examples["text"],
        truncation=True,
        padding=False, # The data collator will handle padding
        max_length=1024, # Adjust as needed
    )
    # For language modeling, the labels are the input_ids
    tokenized_output["labels"] = tokenized_output["input_ids"][:]
    return tokenized_output

# Tokenize the dataset
tokenized_dataset = processed_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=list(processed_dataset.features) # Remove the old 'text' column
)


# --- 5. PEFT (LoRA) Configuration ---

# Configure LoRA for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Target modules can vary by model. These are common for Qwen2.
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

# Wrap the model with the LoRA adapter
peft_model = get_peft_model(model, lora_config)

# --- 6. Training Configuration ---

# Define the training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,  # A single epoch is often enough for fine-tuning
    per_device_train_batch_size=2, # Lower this if you run out of memory
    gradient_accumulation_steps=4, # Simulates a larger batch size
    optim="adamw_torch", # Standard AdamW optimizer
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True, # Use mixed precision for speed and memory savings
    max_grad_norm=0.3,
    max_steps=-1, # Train for the full number of epochs
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# --- 7. Trainer Initialization and Training ---

# Create the Trainer instance
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    # This collator handles padding for you
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Start the training process
print("Starting model fine-tuning...")
trainer.train()

# --- 8. Save the Final Model ---

print("Training complete. Saving the fine-tuned model.")
trainer.save_model(OUTPUT_DIR)

print(f"Model saved to {OUTPUT_DIR}")
