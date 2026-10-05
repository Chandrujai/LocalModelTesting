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

print("Model loaded successfully!")
print("Type your question below.")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question.strip():
        continue

    messages = [
        {"role": "user", "content": question}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True
        )

    response = outputs[0][inputs["input_ids"].shape[-1]:]

    answer = tokenizer.decode(
        response,
        skip_special_tokens=True
    )

    print(f"\nAI: {answer}\n")