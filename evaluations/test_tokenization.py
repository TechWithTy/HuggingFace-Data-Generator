from transformers import AutoTokenizer


def test_tokenization():
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2")

    # Example text to tokenize
    example_text = "This is a test sentence for tokenization."

    # Tokenize the text
    tokens = tokenizer(example_text, padding="max_length",
                       truncation=True, return_tensors="pt")

    # Check if input_ids are generated
    assert "input_ids" in tokens, "The tokenized output should have input_ids."

    # Check if attention_mask is generated
    assert "attention_mask" in tokens, "The tokenized output should have an attention_mask."

    # Optional: Check the length of the tokenized output
    # This will depend on your tokenizer's max_length setting
    max_length = tokenizer.model_max_length
    assert tokens["input_ids"].shape[1] <= max_length, f"Tokenized input should not exceed max length of {
        max_length}."
