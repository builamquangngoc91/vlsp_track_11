import os
import json
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
)
from peft import PeftModel
import re

# --- 1. Configuration ---

# Update the model name to Qwen2.5-VL
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"

# Update the path to the new fine-tuned adapter
ADAPTER_PATH = "./qwen2.5-vl-finetuned-model"

# Input test dataset
TEST_DATA_PATH = "./dataset/test/vlsp_2025_public_test_task2.json"

# Output submission file
SUBMISSION_PATH = "./submission.json"

# --- 2. Load Model and Tokenizer ---

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Use the correct AutoModel class for Vision-Language models
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True
)

# Load the LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Model loaded successfully.")

# --- 3. Load and Prepare Test Data ---

print(f"Loading test data from {TEST_DATA_PATH}...")
with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# --- 4. Inference ---

def format_inference_prompt(sample):
    """
    Formats a test sample into a multimodal prompt for the Qwen-VL model.
    """
    question = sample['question']
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
        
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt

def extract_answer(generated_text):
    """
    Extracts the most likely answer from the model's raw output.
    """
    parts = generated_text.split("<|im_start|>assistant")
    if len(parts) > 1:
        response = parts[-1].replace("<|im_end|>", "").strip()
        first_word = re.split(r'[\s\.\,]', response)[0]
        return first_word
    return ""

print("Generating predictions...")
predictions = []
for item in test_data:
    prompt = format_inference_prompt(item)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    answer = extract_answer(generated_text)
    
    predictions.append({
        "id": item["id"],
        "answer": answer
    })
    
    if len(predictions) % 10 == 0:
        print(f"  Processed {len(predictions)}/{len(test_data)} samples...")

print("Prediction generation complete.")

# --- 5. Save Submission File ---

print(f"Saving submission file to {SUBMISSION_PATH}...")
with open(SUBMISSION_PATH, 'w', encoding='utf-8') as f:
    json.dump(predictions, f, indent=4, ensure_ascii=False)

print("Submission file created successfully!")
