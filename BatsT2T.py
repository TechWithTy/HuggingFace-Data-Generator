from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Explicitly set a padding token if it does not exist
if tokenizer.pad_token is None:
    # Typically, we can use the EOS token if available, or add a new token.
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the dataset from a JSONL file
dataset = load_dataset(
    "json", data_files="data/ProjectSkipTest-instruction-sets.jsonl", split='train')

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2")

# Tokenization function that uses the updated tokenizer with padding token set


# Example: Adjust max_length to a smaller value based on typical text length observed
def tokenize_text(text, max_length=128):  # Adjusted max_length
    return tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")


# Example: Debugging to see decoded tokens
def tokenize_function(batch):
    batch_texts = []
    for messages in batch['messages']:
        concatenated_texts = " ".join(
            [message['content'] for message in messages])
        batch_texts.append(concatenated_texts)

    # Ensure tokenized_text correctly structures its output
    tokenized_outputs = [tokenize_text(
        text, max_length=128) for text in batch_texts]  # Adjusted max_length

    # Assuming tokenize_text returns a dictionary containing 'input_ids'
    # Correcting access to input_ids
    decoded_texts = [tokenizer.decode(output['input_ids'][0])
                     for output in tokenized_outputs]

    # Printing decoded texts for verification
    print("Decoded Texts:", decoded_texts)

    # Prepare the output dictionary correctly
    return {'input_ids': [output['input_ids'] for output in tokenized_outputs],
            'attention_mask': [output['attention_mask'] for output in tokenized_outputs]}


# Apply the updated tokenization function on the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Output sample data to verify tokenization
print(dataset[0])  # Before tokenization
print(tokenized_datasets[0])  # After tokenization
