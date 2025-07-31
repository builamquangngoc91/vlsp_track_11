import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from PIL import Image

# --- 1. Configuration ---

# Model and tokenizer names for the Vision-Language model
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"

# Input dataset
DATASET_PATH = "./training_data_with_context.json"

# Output directory for the fine-tuned model
OUTPUT_DIR = "./qwen-vl-finetuned-model"

# --- 2. Data Loading and Preparation (Robust Method) ---

print("Loading and aggressively cleaning data...")
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
            # Serialize complex types into JSON strings
            new_item[key] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            new_item[key] = ""  # Convert None to empty string
        else:
            new_item[key] = str(value) # Convert all other types to string
    stringified_data.append(new_item)

# Convert the list of dictionaries to a dictionary of lists
data_dict = {key: [d[key] for d in stringified_data] for key in all_keys}

# Create a Dataset object from the cleaned dictionary
dataset = Dataset.from_dict(data_dict)
print("Data loaded successfully.")

# --- 3. Model and Tokenizer Loading ---

print("Loading model and tokenizer...")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load the model
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True
)
print("Model and tokenizer loaded.")

# --- 4. Data Preprocessing for Vision-Language ---

def format_multimodal_prompt(sample):
    """
    Formats a data sample into a multimodal prompt for the Qwen-VL model.
    """
    # De-serialize the stringified fields
    question = sample['question']
    answer = sample['answer']
    image_path = sample['image_id']
    
    # Verify the image exists before creating the prompt
    if not os.path.exists(image_path):
        messages = [{"role": "user", "content": question}]
    else:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image_url": image_path},
                {"type": "text", "text": question}
            ]
        }]

    messages.append({"role": "assistant", "content": answer})
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": prompt}

print("Formatting prompts...")
processed_dataset = dataset.map(format_multimodal_prompt, remove_columns=list(dataset.features))

# --- 5. Tokenization ---

def tokenize_function(examples):
    """Tokenizes the text column."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )

print("Tokenizing dataset...")
tokenized_dataset = processed_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=list(processed_dataset.features)
)

# --- 6. PEFT (LoRA) Configuration ---

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
)

peft_model = get_peft_model(model, lora_config)

# --- 7. Training Configuration ---

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# --- 8. Trainer Initialization and Training ---

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Start the training process
print("Starting Vision-Language model fine-tuning...")
trainer.train()

# --- 9. Save the Final Model ---

print("Training complete. Saving the fine-tuned model.")
trainer.save_model(OUTPUT_DIR)

print(f"Model saved to {OUTPUT_DIR}")
