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

# Base Model and Tokenizer for the Vision-Language model
MODEL_NAME = "Qwen/Qwen1.5-VL-Chat"

# Path to the fine-tuned LoRA adapter
ADAPTER_PATH = "./qwen-vl-finetuned-model"

# Input test dataset
TEST_DATA_PATH = "./dataset/test/vlsp_2025_public_test_task2.json"

# Output submission file
SUBMISSION_PATH = "./submission.json"

# --- 2. Load Model and Tokenizer ---

print("Loading model and tokenizer...")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load the base model
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Load the LoRA adapter and merge it with the base model
print("Loading and merging LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()

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
    
    # For Qwen-VL, the prompt includes an <img> tag with the path
    # The test set should have the images, but we handle the case where it might not.
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
    # This is a simplified extraction logic. It looks for the content
    # after the final 'assistant' turn in the generated text.
    parts = generated_text.split("<|im_start|>assistant")
    if len(parts) > 1:
        response = parts[-1].replace("<|im_end|>", "").strip()
        # Take the first word/letter as the answer
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
