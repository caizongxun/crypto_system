#!/usr/bin/env python3
import os
import sys
from pathlib import Path

print("Installing huggingface_hub...")
os.system('pip install huggingface_hub -q')

from huggingface_hub import HfApi, login

print("="*80)
print("UPLOAD MODELS TO HUGGING FACE")
print("="*80)
print()

# Configuration
REPO_ID = "zongowo111/cpb-models"
REPO_TYPE = "dataset"
MODELS_DIR = Path("/content/all_models/multi_crypto")
REMOTE_FOLDER = "models_v5"

if not MODELS_DIR.exists():
    print(f"ERROR: Models directory not found: {MODELS_DIR}")
    sys.exit(1)

model_files = list(MODELS_DIR.glob('*'))
if not model_files:
    print(f"ERROR: No model files found in {MODELS_DIR}")
    sys.exit(1)

print(f"Found {len(model_files)} model files")
print(f"Models directory: {MODELS_DIR}")
print(f"Target repo: {REPO_ID}")
print(f"Remote folder: {REMOTE_FOLDER}")
print()

print("Logging into Hugging Face...")
print("Visit https://huggingface.co/settings/tokens to get your token")
print()

try:
    login()
    print("Login successful!")
    print()
except Exception as e:
    print(f"ERROR during login: {e}")
    print("Make sure to use a valid Hugging Face token.")
    sys.exit(1)

api = HfApi()

try:
    print("Checking repository...")
    repo_info = api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Repository found: {REPO_ID}")
    print()
except Exception as e:
    print(f"ERROR: Repository not found or access denied: {e}")
    print(f"Please ensure the repository exists and you have write access.")
    sys.exit(1)

try:
    print(f"Uploading {len(model_files)} model files to {REMOTE_FOLDER}/...")
    print("This may take a few minutes...")
    print()
    
    api.upload_folder(
        folder_path=str(MODELS_DIR),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        path_in_repo=REMOTE_FOLDER,
        commit_message=f"Upload {len(model_files)} trained crypto models (v5)",
        ignore_patterns=["*.pyc", "__pycache__"],
        allow_patterns=["*.h5", "*.pkl"],
    )
    print()
    print("="*80)
    print("UPLOAD SUCCESSFUL!")
    print("="*80)
    print()
    print(f"Models uploaded to:")
    print(f"https://huggingface.co/datasets/{REPO_ID}/tree/main/{REMOTE_FOLDER}")
    print()
    print(f"Total files uploaded: {len(model_files)}")
    print()
except Exception as e:
    print(f"ERROR during upload: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
