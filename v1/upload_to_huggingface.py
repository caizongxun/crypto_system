#!/usr/bin/env python3
import os
import sys
from pathlib import Path

print("Installing huggingface_hub...")
os.system('pip install huggingface_hub -q')

from huggingface_hub import HfApi, create_repo

print("="*80)
print("UPLOAD MODELS TO HUGGING FACE")
print("="*80)
print()

# Configuration
HF_TOKEN = input("Enter your Hugging Face token: ").strip()
REPO_ID = "zongowo111/cpb-models"
REPO_TYPE = "dataset"
MODELS_DIR = Path("/content/all_models/multi_crypto")
REMOTE_FOLDER = "models_v5"

if not HF_TOKEN:
    print("ERROR: Hugging Face token is required!")
    sys.exit(1)

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

api = HfApi(token=HF_TOKEN)

try:
    print("Checking/creating repository...")
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
    print(f"Repository ready: {REPO_ID}")
    print()
except Exception as e:
    print(f"ERROR creating repo: {e}")
    sys.exit(1)

try:
    print(f"Uploading entire models folder to {REMOTE_FOLDER}/...")
    api.upload_folder(
        folder_path=str(MODELS_DIR),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        path_in_repo=REMOTE_FOLDER,
        commit_message=f"Upload {len(model_files)} trained crypto models (v5)",
        ignore_patterns=["*.pyc", "__pycache__"],
    )
    print()
    print("="*80)
    print("UPLOAD SUCCESSFUL!")
    print("="*80)
    print()
    print(f"Models uploaded to: https://huggingface.co/datasets/{REPO_ID}/tree/main/{REMOTE_FOLDER}")
    print(f"Total files: {len(model_files)}")
    print()
except Exception as e:
    print(f"ERROR during upload: {e}")
    sys.exit(1)
