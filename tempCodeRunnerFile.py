tokenized_datasets = dataset.map(tokenize_function, batched=True)
