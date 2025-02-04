import os
from datasets import load_dataset
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split

def is_dataset_cached(file_path):
    return os.path.exists(file_path)

def load_dataset(file_path, dataset_cache_dir, batch_size):
    # If dataset isn't cached, download and save it
    if not is_dataset_cached(file_path):
        print("Downloading WikiText-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=dataset_cache_dir)
        text_data = "\n".join(dataset["train"]["text"])
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(text_data)
        print("Dataset downloaded and saved.")
    else:
        print("Loading dataset from cache...")
    
    # Load and tokenize the dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = "[PAD]"
    with open(file_path, "r") as f:
        text = f.read()
    
    # Split the text data into train and test sets
    train_texts, test_texts = train_test_split(text.splitlines(), test_size=0.2, random_state=42)
    
    # Tokenize the train and test datasets
    train_encodings = tokenizer(train_texts, return_tensors="pt", max_length=128, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, return_tensors="pt", max_length=128, truncation=True, padding=True)
    
    # Get number of examples
    num_train_samples = len(train_encodings["input_ids"])
    num_test_samples = len(test_encodings["input_ids"])

    # Calculate expected number of batches
    num_train_batches = (num_train_samples + batch_size - 1) // batch_size  # Round up
    num_test_batches = (num_test_samples + batch_size - 1) // batch_size

    # Print information
    print(f"Train dataset size: {num_train_samples} samples")
    print(f"Test dataset size: {num_test_samples} samples")
    print(f"Expected number of train batches (batch_size={batch_size}): {num_train_batches}")
    print(f"Expected number of test batches (batch_size={batch_size}): {num_test_batches}")

    return train_encodings, test_encodings