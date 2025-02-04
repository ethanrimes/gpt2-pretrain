import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json

# Define the directory where models are saved
MODEL_DIR = "./cache/models/"

def list_available_models():
    """Lists all available trained model snapshots with indexes."""
    models = [m for m in os.listdir(MODEL_DIR) if m.startswith("trained_gpt2_")]
    models.sort(reverse=True)  # Show latest models first
    if not models:
        print("No trained models found.")
        return []
    
    print("\nAvailable Model Snapshots:")
    for idx, model_name in enumerate(models):
        print(f"[{idx}] {model_name}")

    return models

def load_selected_model(models):
    """Prompts the user to select a model and loads it with all components."""
    while True:
        try:
            choice = int(input("\nEnter the number of the model to load: "))
            if 0 <= choice < len(models):
                selected_model_path = os.path.join(MODEL_DIR, models[choice])
                print(f"\nLoading model: {models[choice]} ...")
                
                # Load the saved tokenizer with its configuration
                tokenizer = GPT2Tokenizer.from_pretrained(selected_model_path)
                
                # Load the model
                model = GPT2LMHeadModel.from_pretrained(selected_model_path)
                
                # Load special token configuration if it exists
                token_config_path = os.path.join(selected_model_path, "token_config.json")
                if os.path.exists(token_config_path):
                    with open(token_config_path, "r") as f:
                        token_config = json.load(f)
                        tokenizer.pad_token = token_config["special_tokens"]["pad_token"]
                        
                return tokenizer, model
            else:
                print("Invalid selection. Please choose a valid number.")
        except ValueError:
            print("Please enter a number.")

def generate_response(model, tokenizer, prompt, max_length=50):
    """Generates a chatbot response with proper attention masking."""
    # Encode the input text
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Create attention mask
    inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).long()

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Main chatbot loop
if __name__ == "__main__":
    print("\nGPT-2 Chatbot Ready!")

    # List models and let user choose
    available_models = list_available_models()
    if not available_models:
        print("No trained models found. Please train a model first.")
        exit()

    tokenizer, model = load_selected_model(available_models)

    model.eval()  # Ensure model is in inference mode

    print("\nChatbot loaded! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        response = generate_response(model, tokenizer, user_input)
        print(f"Bot: {response}")