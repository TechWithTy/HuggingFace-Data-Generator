from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

#? TODO
#1 Tokenize for our data set
#2 Evaluate output data

# Load the dataset from a JSONL file
dataset = load_dataset(
    "json", data_files="ProjectSkipTest-instruction-sets.jsonl", split='train')

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(dataset[0])  # Before tokenization
print(tokenized_datasets[0])  # After tokenization
