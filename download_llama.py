import os
from huggingface_hub import snapshot_download

model_name = "meta-llama/Llama-3.2-3B"
local_dir = "C:/Llama3.2/Llama-3-8B"

# Check if the model is already downloaded
if not os.path.exists(local_dir):
    print(f"Model not found locally. Downloading {model_name}...")
    
    # Download the full model from Hugging Face
    snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)

    print(f"Model downloaded successfully to {local_dir}")
else:
    print(f"Model found locally at {local_dir}")