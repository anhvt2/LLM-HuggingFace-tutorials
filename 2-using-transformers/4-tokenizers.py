
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
print(tokenizer(tokenized_text))

# Tokenization

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens)

# From tokens to input IDs
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

# Decode
decoded_string = tokenizer.decode(ids)
print(decoded_string) # verify that decoded_string == sequence

