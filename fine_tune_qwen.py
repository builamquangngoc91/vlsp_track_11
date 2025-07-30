

import os
import torch
from datasets import load_dataset
from transformers (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# --- 1. Configuration ---

# Model and tokenizer names
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

# Input dataset
DATASET_PATH = "./training_data_with_context.json"

# Output directory for the fine-tuned model
OUTPUT_DIR = "./qwen2-finetuned-model"

# --- 2. Data Loading and Preparation ---

# Load the dataset from the JSON file
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# --- 3. Model and Tokenizer Loading ---

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Qwen models don't have a default pad token, so we set it to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Configure quantization to save memory (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Load the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto", # Automatically maps model layers to available devices (GPU/CPU)
    trust_remote_code=True
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# --- 4. Data Preprocessing ---

def format_prompt(sample):
    """
    Formats a data sample into a prompt for the model.
    This is a crucial step where you define how the model receives the information.
    
    NOTE: For a true Vision-Language model, this function would also handle
    loading and preprocessing the image associated with `sample['image_id']`.
    """
    # Combine the text of all relevant articles into a single context string
    context = "\n\n".join([article['article_text'] for article in sample['relevant_articles_content']])
    
    # Get the question
    question = sample['question']
    
    # Format choices for multiple choice questions
    if sample['question_type'] == 'Multiple choice' and sample['choices']:
        choices_text = "\n".join([f"{key}: {value}" for key, value in sample['choices'].items()])
        question = f"{question}\n{choices_text}"
        
    # Get the answer
    answer = sample['answer']
    
    # Create the prompt using the Qwen-Instruct chat template
    # This helps the model understand the roles of user and assistant
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on Vietnamese traffic law."},
        {"role": "user", "content": f"Based on the following legal articles, please answer the question.\n\n--- BEGIN CONTEXT ---\n{context}\n--- END CONTEXT ---\n\nQuestion: {question}"},
        {"role": "assistant", "content": answer}
    ]
    
    # The tokenizer will apply the chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    return {"text": prompt}

# Apply the formatting to the entire dataset
processed_dataset = dataset.map(format_prompt, remove_columns=list(dataset.features))

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
    optim="paged_adamw_8bit",
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
    train_dataset=processed_dataset,
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
