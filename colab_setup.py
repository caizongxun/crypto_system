# ============================================================================
# Crypto System - Colab Quick Start Setup Script
# ============================================================================
# Execute this in first Colab cell to install all dependencies
# 
# Usage:
#   import requests, time
#   url = 'https://raw.githubusercontent.com/caizongxun/crypto_system/main/colab_setup.py?t=' + str(int(time.time()))
#   exec(requests.get(url).text)
#
# Then in next cell:
#   import requests, time
#   url = 'https://raw.githubusercontent.com/caizongxun/crypto_system/main/v1/train_v1.py?t=' + str(int(time.time()))
#   exec(requests.get(url).text)

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("\n" + "="*80)
print("CRYPTO SYSTEM - COLAB ENVIRONMENT SETUP")
print("="*80 + "\n")

# ============================================================================
# Step 1: Update Package Manager
# ============================================================================
logger.info("Step 1: Updating package manager...")
try:
    subprocess.check_call(["apt-get", "update", "-qq"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("✓ Package manager updated\n")
except Exception as e:
    logger.warning(f"⚠ Could not update apt: {str(e)}\n")

# ============================================================================
# Step 2: Install Python Packages (No System Dependencies)
# ============================================================================
logger.info("Step 2: Installing Python packages...")

python_packages = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "tensorflow>=2.13.0",
    "keras>=2.13.0",
    "scikit-learn>=1.3.0",
    "ccxt>=3.0.0",
    "pyarrow>=12.0.0",
    "pandas-ta>=0.3.14b0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "requests>=2.31.0",
]

failed_packages = []
for pkg in python_packages:
    try:
        logger.info(f"  Installing {pkg}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        logger.warning(f"  ⚠ Failed to install {pkg}: {str(e)}")
        failed_packages.append(pkg)

if failed_packages:
    logger.warning(f"\n⚠ {len(failed_packages)} packages failed to install:")
    for pkg in failed_packages:
        logger.warning(f"   - {pkg}")
    logger.warning("\nAttempting alternative installation...\n")
    
    # Try alternative installation
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "-q"] + failed_packages,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info("✓ Alternative installation succeeded\n")
    except Exception as e:
        logger.warning(f"✗ Alternative installation also failed: {str(e)}\n")
else:
    logger.info("✓ Python packages installed\n")

# ============================================================================
# Step 3: Verify GPU
# ============================================================================
logger.info("Step 3: Checking GPU availability...")

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        logger.info(f"✓ GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            logger.info(f"    {gpu}")
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("✓ GPU memory growth enabled\n")
    else:
        logger.warning("⚠ No GPU detected!")
        logger.warning("  Go to: Runtime → Change runtime type → Select GPU (T4 or A100)")
        logger.warning("  Then re-run this cell\n")
except Exception as e:
    logger.error(f"Error checking GPU: {str(e)}\n")

# ============================================================================
# Step 4: Mount Google Drive (Optional)
# ============================================================================
logger.info("Step 4: Mounting Google Drive (optional)...")

try:
    from google.colab import drive
    try:
        drive.mount('/content/drive', force_remount=False)
        logger.info("✓ Google Drive mounted\n")
    except:
        logger.info("✓ Google Drive already mounted\n")
except:
    logger.warning("⚠ Google Drive mount skipped (not in Colab or already mounted)\n")

# ============================================================================
# Step 5: Create Cache Directories
# ============================================================================
logger.info("Step 5: Creating cache directories...")

cache_dir = Path("/content/all_models")
cache_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"✓ Cache directory created: {cache_dir}\n")

# ============================================================================
# Step 6: Verify Critical Imports
# ============================================================================
logger.info("Step 6: Verifying critical imports...")

imports_to_check = {
    "pandas": "pd",
    "numpy": "np",
    "tensorflow": "tf",
    "sklearn": "sklearn",
    "ccxt": "ccxt",
    "matplotlib": "plt",
}

all_good = True
for module_name, alias in imports_to_check.items():
    try:
        __import__(module_name)
        logger.info(f"✓ {module_name} imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import {module_name}: {str(e)}")
        all_good = False

logger.info("")

# ============================================================================
# Summary
# ============================================================================
if all_good:
    logger.info("="*80)
    logger.info("✓ SETUP COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("")
    logger.info("1. In the next Colab cell, run the training script:")
    logger.info("")
    logger.info("   import requests, time")
    logger.info("   url = 'https://raw.githubusercontent.com/caizongxun/crypto_system/main/v1/train_v1.py?t=' + str(int(time.time()))")
    logger.info("   exec(requests.get(url).text)")
    logger.info("")
    logger.info("2. The training will:")
    logger.info("   • Fetch data from Binance US API (2-3 min)")
    logger.info("   • Process features (1-2 min)")
    logger.info("   • Train on GPU (T4: 30-40 min, A100: 10-15 min)")
    logger.info("   • Save model to /content/all_models/v1/")
    logger.info("")
else:
    logger.error("\n" + "="*80)
    logger.error("✗ SETUP INCOMPLETE - Some critical imports failed")
    logger.error("="*80)
    logger.error("\nTroubleshooting:")
    logger.error("1. Try manual installation:")
    logger.error("   !pip install --upgrade tensorflow keras pandas numpy scikit-learn ccxt")
    logger.error("")
    logger.error("2. If that fails, check Colab's internet connection")
    logger.error("3. Try again in a new notebook")
    logger.error("")
