
from transformers import pipeline
generator = pipeline("text-generation")
chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    full_prompt = user_input

    response = generator(full_prompt)[0]["generated_text"]
    print(f"Bot: {response.split('[/INST]')[-1].strip()}\n")
