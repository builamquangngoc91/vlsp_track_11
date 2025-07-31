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

MODEL_NAME = "Qwen/Qwen1.5-VL-Chat"
DATASET_PATH = "./training_data_with_context.json"
OUTPUT_DIR = "./qwen2.5-vl-finetuned-model"

# --- 2. Data Loading and Preparation (Robust Method) ---

print("Loading and aggressively cleaning data...")
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

all_keys = set(k for item in data for k in item.keys())

stringified_data = []
for item in data:
    new_item = {}
    for key in all_keys:
        value = item.get(key)
        if isinstance(value, (dict, list)):
            new_item[key] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            new_item[key] = ""
        else:
            new_item[key] = str(value)
    stringified_data.append(new_item)

data_dict = {key: [d.get(key, "") for d in stringified_data] for key in all_keys}
dataset = Dataset.from_dict(data_dict)
print("Data loaded successfully.")

# --- 3. Model and Tokenizer Loading ---

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Use the correct AutoModel class for Vision-Language models
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True
)
print("Model and tokenizer loaded.")

# --- 4. Data Preprocessing for Vision-Language ---

def format_multimodal_prompt(sample):
    question = sample['question']
    answer = sample['answer']
    image_path = sample['image_id']
    
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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

peft_model = get_peft_model(model, lora_config)

# --- 7. Training Configuration ---

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
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

print("Starting Vision-Language model fine-tuning...")
trainer.train()

# --- 9. Save the Final Model ---

print("Training complete. Saving the fine-tuned model.")
trainer.save_model(OUTPUT_DIR)

print(f"Model saved to {OUTPUT_DIR}")
