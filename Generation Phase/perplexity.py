from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Initialize the model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to calculate perplexity of a text passage
def calculate_perplexity(text):
    # Tokenize the input text and get input IDs
    inputs = tokenizer.encode(text, return_tensors='pt')

    # Get model outputs
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss

    # Convert loss to perplexity
    perplexity = torch.exp(loss)
    return perplexity.item()

# Example usage
text = "The quick brown fox jumps over the lazy dog."
perplexity = calculate_perplexity(text)
print(f"Perplexity: {perplexity}")
