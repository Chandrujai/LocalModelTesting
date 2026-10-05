import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./models/qwen2.5-0.5b-instruct"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)

print("Model loaded successfully!\n")

# Your question
prompt = "Explain artificial intelligence in simple terms."

messages = [
    {"role": "user", "content": prompt}
]

# Convert conversation into the format expected by Qwen
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

# Remove the original prompt from the output
response = outputs[0][inputs["input_ids"].shape[-1]:]

answer = tokenizer.decode(
    response,
    skip_special_tokens=True
)

print("Question:")
print(prompt)

print("\nAnswer:")
print(answer)