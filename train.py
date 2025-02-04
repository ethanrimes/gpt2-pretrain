import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, TensorDataset
from config.model_config import Config
from model.gpt2 import create_model
from data.dataset import load_dataset
from evaluate import evaluate_perplexity
import datetime
import json

# Define cache paths
MODEL_CACHE_DIR = "./cache/models/gpt2"
MODEL_SAVE_ROOT = "./cache/models/trained_gpt2"
DATASET_CACHE_DIR = "./cache/datasets"
DATASET_PATH = "./data/train.txt"

# Ensure cache directories exist
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

# Disable distributed training
os.environ["TORCH_ELASTIC_DISABLE"] = "True"

def save_model_checkpoint(model, tokenizer, save_root):
    """
    Save model checkpoint including weights, tokenizer, and configuration
    """
    # Create a unique timestamp for each model save
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = f"{save_root}_{timestamp}"
    
    # Ensure directory exists
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save model weights
    model.save_pretrained(model_save_path)
    
    # Save tokenizer and its configuration
    tokenizer.save_pretrained(model_save_path)
    
    # Save special token configuration
    config_data = {
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "special_tokens": {
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token
        }
    }
    
    with open(os.path.join(model_save_path, "token_config.json"), "w") as f:
        json.dump(config_data, f)
    
    print(f"Model checkpoint saved to {model_save_path}")
    return model_save_path

def train():
    # Check if MPS is available and set the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model and dataset using the abstracted functions
    model = create_model(MODEL_CACHE_DIR).to(device)  # Move model to the specified device
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = "[PAD]"
    model.train()

    # Load train and test datasets
    train_data, test_data = load_dataset(DATASET_PATH, DATASET_CACHE_DIR, Config.batch_size)
    train_dataset = TensorDataset(train_data["input_ids"].to(device))  # Move input IDs to the specified device
    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)

    for epoch in range(Config.num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch[0].to(device)  # Move input IDs to the specified device
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch} Loss: {loss.item()}")

    # Save the trained model with all components
    save_model_checkpoint(model, tokenizer, MODEL_SAVE_ROOT)

    # Evaluate the model on the test data
    perplexity = evaluate_perplexity(model, test_data, device, Config.batch_size)
    print(f"Perplexity: {perplexity}")

if __name__ == "__main__":
    train()