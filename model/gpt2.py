import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def is_model_cached(cache_path):
    # Check for model files in the snapshots directory
    snapshots_dir = os.path.join(cache_path, "snapshots")
    print(f"snapshots_dir: {snapshots_dir}")
    if os.path.exists(snapshots_dir):
        snapshot_folders = [folder for folder in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, folder))]
        print(f"snapshot_folders: {snapshot_folders}")
        if snapshot_folders:  # If there are any snapshots
            snapshot_path = os.path.join(snapshots_dir, snapshot_folders[0])
            # Check if the snapshot contains model files
            return os.path.exists(os.path.join(snapshot_path, "model.safetensors"))
    return False

def create_model(cache_path, snapshot_name=None):
    # If model isn't in cache, download and save it
    if not is_model_cached(cache_path):
        print("Downloading GPT-2 model...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        os.makedirs(cache_path, exist_ok=True)
        model.save_pretrained(cache_path)
        print("Model downloaded and cached.")
    else:
        print("Loading GPT-2 model from cache...")
        # Find the snapshot directory
        snapshots_dir = os.path.join(cache_path, "snapshots")
        if snapshot_name:
            snapshot_path = os.path.join(snapshots_dir, snapshot_name)
        else:
            snapshot_folders = [folder for folder in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, folder))]
            snapshot_path = os.path.join(snapshots_dir, snapshot_folders[0])
        print(f"snapshot_path: {snapshot_path}")
        model = GPT2LMHeadModel.from_pretrained(snapshot_path)
    
    return model