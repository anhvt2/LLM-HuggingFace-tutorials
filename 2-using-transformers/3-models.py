
# Create a transformer
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")

# Load and save

model.save_pretrained("./") # save "config.json" "pytorch_model.bin"

# Reuse a saved model
from transformers import AutoModel

model = AutoModel.from_pretrained(".")

# Login to Huggingface

from huggingface_hub import notebook_login

notebook_login()

# model.push_to_hub("my-awesome-model") # Push to share

# Encoding text

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)

# Decode an encoded to recover the original text
tokenizer.decode(encoded_input["input_ids"])

# Encode multiple inputs
encoded_input = tokenizer("How are you?", "I'm fine, thank you!")
print(encoded_input)

# Encode multiple inputs and return as tensors
encoded_input = tokenizer("How are you?", "I'm fine, thank you!", return_tensors="pt")
print(encoded_input)

# Pad inputs to have the same length
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"], padding=True, return_tensors="pt"
)
print(encoded_input)

# Truncate input
encoded_input = tokenizer(
    "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long sentence.",
    truncation=True,
)
print(encoded_input["input_ids"])

# Combine padding and truncation to have fixed-length tokens

encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,
    truncation=True,
    max_length=5,
    return_tensors="pt",
)
print(encoded_input)

# Adding special tokens: beginning [CLS] and ending [SEP]

encoded_input = tokenizer("How are you?")
print(encoded_input["input_ids"])
tokenizer.decode(encoded_input["input_ids"])

# An example
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
