import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
import re

# --- 1. Configuration ---

# Base Model and Tokenizer
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

# Path to the fine-tuned LoRA adapter
ADAPTER_PATH = "/Users/tyler/Desktop/ngoc/vlsp_track_11/qwen2-finetuned-model"

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
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16, # Use float16 for faster inference
    trust_remote_code=True
)

# Load the LoRA adapter and merge it with the base model
print("Loading and merging LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()

print("Model loaded successfully.")

# --- 3. Load and Prepare Test Data ---

print(f"Loading test data from {TEST_DATA_PATH}...")
# Load the JSON file manually
with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# --- 4. Inference ---

def format_inference_prompt(sample):
    """
    Formats a test sample into a prompt for the model.
    """
    # The test data has the same structure, just without the 'answer'
    context_articles = sample.get('relevant_articles_content', [])
    context = "\n\n".join([article['article_text'] for article in context_articles])
    
    question = sample['question']
    
    if sample['question_type'] == 'Multiple choice':
        choices_dict = sample.get('choices', {})
        if choices_dict:
            choices_text = "\n".join([f"{key}: {value}" for key, value in choices_dict.items()])
            question = f"{question}\n{choices_text}"
            
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on Vietnamese traffic law."},
        {"role": "user", "content": f"Based on the following legal articles, please answer the question.\n\n--- BEGIN CONTEXT ---\n{context}\n--- END CONTEXT ---\n\nQuestion: {question}"}
    ]
    
    # Apply chat template for inference (add_generation_prompt=True)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def extract_answer(generated_text):
    """
    Extracts the most likely answer from the model's raw output.
    This is a simple implementation and might need refinement.
    """
    # The model's output will be the full prompt + the generated answer.
    # We look for the answer after the assistant tag.
    assistant_tag = "<|im_start|>assistant\n"
    start_index = generated_text.find(assistant_tag)
    if start_index != -1:
        response = generated_text[start_index + len(assistant_tag):].strip()
        # Take the first word/letter as the answer
        first_word = re.split(r'[\s\.\,]', response)[0]
        return first_word
    return "" # Return empty if something goes wrong

print("Generating predictions...")
predictions = []
for item in test_data:
    # Format the prompt
    prompt = format_inference_prompt(item)
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate the output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20, # Generate a short answer
            do_sample=False, # Use greedy decoding for deterministic output
        )
    
    # Decode and extract the answer
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