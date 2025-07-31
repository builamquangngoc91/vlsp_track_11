import os
import json
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
)
from peft import PeftModel
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Configuration ---

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_PATH = "./qwen2-vl-7b-finetuned-model"
TEST_TASK1_PATH = "./dataset/test/vlsp_2025_public_test_task1.json"
TEST_TASK2_PATH = "./dataset/test/vlsp_2025_public_test_task2.json"
LAW_DATA_PATH = "./article_mapping.json"
SUBMISSION_TASK1_PATH = "./submission_task1.json" # New path for task 1
SUBMISSION_TASK2_PATH = "./submission_task2.json" # New path for task 2
IMAGE_PATH_PREFIX = "dataset/test/public_test_images/"

# --- 2. Load Model and Tokenizer ---

print(f"Loading model and tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
print("Model loaded successfully.")

# --- 3. Load Law Data and Build Retriever for Task 1 ---

print("Loading law articles and building retriever...")
with open(LAW_DATA_PATH, 'r', encoding='utf-8') as f:
    law_data = json.load(f)

corpus = [article['article_text'] for article in law_data]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

def retrieve_articles(query, top_k=3):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [law_data[i] for i in top_indices]

print("Retriever built successfully.")

# --- 4. Inference Functions ---

def format_inference_prompt(question, image_path, relevant_articles):
    context = "\n\n".join([article['article_text'] for article in relevant_articles])
    
    content = [{"type": "text", "text": f"Based on the following legal articles, please answer the question.\n\n--- BEGIN CONTEXT ---\n{context}\n--- END CONTEXT ---\n\nQuestion: {question}"}]
    
    if os.path.exists(image_path):
        content.insert(0, {"type": "image", "image_url": image_path})
        
    messages = [{"role": "user", "content": content}]
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def extract_answer(generated_text):
    parts = generated_text.split("<|im_start|>assistant")
    if len(parts) > 1:
        response = parts[-1].replace("<|im_end|>", "").strip()
        first_word = re.split(r'[\s\.\,]', response)[0]
        return first_word
    return ""

# --- 5. Process Task 1 ---
print("\nProcessing Task 1...")
with open(TEST_TASK1_PATH, 'r', encoding='utf-8') as f:
    test_data1 = json.load(f)

task1_predictions = []
for i, item in enumerate(test_data1):
    image_path = f"{IMAGE_PATH_PREFIX}{item['image_id']}.jpg"
    retrieved_articles = retrieve_articles(item['question'])
    prompt = format_inference_prompt(item['question'], image_path, retrieved_articles)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    answer = extract_answer(generated_text)

    print("prompt: ", prompt)
    print("generated_text: ", generated_text)
    print("answer: ", answer)
    
    task1_predictions.append({
        "id": item["id"],
        "relevant_articles": [{"law_id": art["law_id"], "article_id": art["article_id"]} for art in retrieved_articles],
        "answer": answer
    })
    print(f"  Processed Task 1 sample {i+1}/{len(test_data1)}")

print(f"Saving Task 1 submission file to {SUBMISSION_TASK1_PATH}...")
with open(SUBMISSION_TASK1_PATH, 'w', encoding='utf-8') as f:
    json.dump(task1_predictions, f, indent=4, ensure_ascii=False)
print("Task 1 submission file created successfully!")


# --- 6. Process Task 2 ---
print("\nProcessing Task 2...")
with open(TEST_TASK2_PATH, 'r', encoding='utf-8') as f:
    test_data2 = json.load(f)

task2_predictions = []
for i, item in enumerate(test_data2):
    image_path = f"{IMAGE_PATH_PREFIX}{item['image_id']}.jpg"
    
    provided_articles = [
        next((article for article in law_data if article['law_id'] == ref['law_id'] and article['article_id'] == ref['article_id']), None)
        for ref in item['relevant_articles']
    ]
    provided_articles = [art for art in provided_articles if art]

    prompt = format_inference_prompt(item['question'], image_path, provided_articles)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    answer = extract_answer(generated_text)
    
    task2_predictions.append({
        "id": item["id"],
        "answer": answer
    })
    print(f"  Processed Task 2 sample {i+1}/{len(test_data2)}")

print(f"Saving Task 2 submission file to {SUBMISSION_TASK2_PATH}...")
with open(SUBMISSION_TASK2_PATH, 'w', encoding='utf-8') as f:
    json.dump(task2_predictions, f, indent=4, ensure_ascii=False)
print("Task 2 submission file created successfully!")
