from datasets import load_metric

# Load a test set
test_dataset = load_dataset(
    "json", data_files="your_test_file.jsonl", split='test'
)

# Tokenize the test dataset
tokenized_test_datasets = test_dataset.map(tokenize_function, batched=True)

# Function to evaluate model on the test set
def evaluate_model(model, tokenized_datasets):
    metric = load_metric("accuracy")  # Use an appropriate metric for your task
    for batch in tokenized_datasets:
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    final_score = metric.compute()
    print(f"Model accuracy: {final_score['accuracy']}")

# Evaluate the model
evaluate_model(model, tokenized_test_datasets)
