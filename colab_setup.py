# ============================================================================
# Crypto System - Colab Quick Start Setup Script (Simplified)
# ============================================================================
# Execute this in first Colab cell to install all dependencies
# 
# Usage:
#   import requests, time
#   url = 'https://raw.githubusercontent.com/caizongxun/crypto_system/main/colab_setup.py?t=' + str(int(time.time()))
#   exec(requests.get(url).text)

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

print("\n" + "="*80)
print("CRYPTO SYSTEM - COLAB SETUP")
print("="*80 + "\n")

# ============================================================================
# Install Only Essential Packages (Skip if Already Installed)
# ============================================================================
logger.info("Installing essential packages...")

essential_packages = [
    "ccxt>=3.0.0",
    "pyarrow>=12.0.0",
    "pandas-ta>=0.3.14b0",
]

for pkg in essential_packages:
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info(f"  ✓ {pkg}")
    except Exception as e:
        logger.warning(f"  ⚠ {pkg}: {str(e)[:50]}")

print()

# ============================================================================
# Verify GPU
# ============================================================================
logger.info("Checking GPU...")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"  ✓ GPU found: {len(gpus)} device(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("  ✓ GPU memory growth enabled")
    else:
        logger.warning("  ⚠ No GPU detected!")
        logger.warning("    → Runtime → Change runtime type → T4 or A100")
except Exception as e:
    logger.error(f"  ✗ Error: {str(e)}")

print()

# ============================================================================
# Create Cache Directories
# ============================================================================
logger.info("Creating directories...")
cache_dir = Path("/content/all_models")
cache_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"  ✓ {cache_dir}")

print()

# ============================================================================
# Verify Imports
# ============================================================================
logger.info("Verifying imports...")

required_imports = [
    ("pandas", "pd"),
    ("numpy", "np"),
    ("tensorflow", "tf"),
    ("sklearn", "sklearn"),
    ("ccxt", "ccxt"),
    ("matplotlib", "plt"),
]

all_ok = True
for module_name, alias in required_imports:
    try:
        __import__(module_name)
        logger.info(f"  ✓ {module_name}")
    except ImportError:
        logger.error(f"  ✗ {module_name} (required!)")
        all_ok = False

print()

# ============================================================================
# Summary
# ============================================================================
if all_ok:
    logger.info("="*80)
    logger.info("✓ SETUP COMPLETE - Ready to train!")
    logger.info("="*80)
    logger.info("")
    logger.info("Next: Run training in the next cell:")
    logger.info("")
    logger.info("  import requests, time")
    logger.info("  url = 'https://raw.githubusercontent.com/caizongxun/crypto_system/main/v1/train_v1.py?t=' + str(int(time.time()))")
    logger.info("  exec(requests.get(url).text)")
    logger.info("")
else:
    logger.info("="*80)
    logger.info("⚠ Some imports failed - Training may not work")
    logger.info("="*80)
    logger.info("")
    logger.info("Try manual install:")
    logger.info("  !pip install --upgrade pandas numpy tensorflow scikit-learn ccxt pyarrow pandas-ta")
    logger.info("")
