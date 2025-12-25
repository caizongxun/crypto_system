# 🚀 Crypto System - Colab Quick Start Guide

## Prerequisites

- Google Colab account (free)
- No local installation needed (everything runs in cloud)
- GPU access (T4 or A100 recommended)

---

## Step-by-Step Setup

### 1️⃣ Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "New notebook"
3. Name it "Crypto System Training"

### 2️⃣ Enable GPU (Critical!)

**Menu**: Runtime → Change runtime type

- **GPU**: Select "T4" (free) or "A100" (faster, may require Pro)
- Click "Save"

![GPU Setup](https://imgur.com/abc123.png)

### 3️⃣ Environment Setup (Cell 1)

Paste this in first cell and run (⏯️ Ctrl+Enter):

```python
import requests
import time

url = 'https://raw.githubusercontent.com/caizongxun/crypto_system/main/colab_setup.py?t=' + str(int(time.time()))
exec(requests.get(url).text)
```

**What this does:**
- ✅ Installs all Python packages
- ✅ Verifies GPU is available
- ✅ Creates cache directories
- ✅ Mounts Google Drive (optional)

**Expected output:**
```
================================================================================
CRYPTO SYSTEM - COLAB ENVIRONMENT SETUP
================================================================================

Step 1: Installing system dependencies...
Step 2: Installing Python packages...
Step 3: Checking GPU availability...
✓ GPU available: 1 device(s)
   <GPUDevice name='/physical_device:GPU:0', compute_capability='8.6'>
Step 4: Mounting Google Drive (optional)...
Step 5: Creating cache directories...
Step 6: Verifying critical imports...

================================================================================
SETUP COMPLETED SUCCESSFULLY!
================================================================================
```

**Troubleshooting:**
- If no GPU found: Runtime → Change runtime type → Select T4 or A100
- If import fails: Scroll up and check error messages

### 4️⃣ Train Model (Cell 2)

Paste this in second cell and run:

```python
import requests
import time

url = 'https://raw.githubusercontent.com/caizongxun/crypto_system/main/v1/train_v1.py?t=' + str(int(time.time()))
exec(requests.get(url).text)
```

**Training timeline:**

| Phase | Time | Status |
|-------|------|--------|
| Data fetching | 2-3 min | 🔄 Downloading 1000+ candles per coin |
| Feature engineering | 1-2 min | 🔄 Computing 50+ technical indicators |
| Model building | 30 sec | 🔄 Constructing dual-stream LSTM |
| Training | 30-40 min (T4) / 10-15 min (A100) | 🔄 GPU optimization |
| Evaluation | 1 min | ✅ Testing on unseen data |

**Expected output:**
```
================================================================================
CRYPTO SYSTEM v1 - TRAINING INITIALIZATION
================================================================================

GPU Available: 1 device(s)
  <GPUDevice name='/physical_device:GPU:0', compute_capability='8.6'>
GPU memory growth enabled
Cache directory: /content/all_models/v1

================================================================================
PHASE 1: DATA FETCHING AND PROCESSING
================================================================================

Fetching data for 22 symbols...
Fetched 1000 candles for BTC/USDT 1h
Fetched 1000 candles for BTC/USDT 15m
Fetched 1000 candles for ETH/USDT 1h
...

================================================================================
PHASE 2: MODEL ARCHITECTURE CONSTRUCTION
================================================================================

Building 1h trend stream...
Building 15m volatility stream...
Merging streams...
Using quantile regression output (3 predictions)

Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape                 Param #
==================================================================================================
input_1h_trend (InputLayer)     [(None, 60, 50)]             0
input_15m_volatility (InputLayer) [(None, 60, 50)]           0
LSTM (LSTM)                     (None, 60, 64)               29184
...
Dense (Dense)                   (None, 3)                    51
==================================================================================================
Total params: 154,387
Trainable params: 154,387
Non-trainable params: 0

================================================================================
PHASE 4: MODEL TRAINING
================================================================================

Epoch 1/100
2000/2000 [==============================] 19s 10ms/step - loss: 0.0032 - mae: 0.0145 - mape: 0.0098 - val_loss: 0.0035 - val_mae: 0.0156 - val_mape: 0.0105
Epoch 2/100
2000/2000 [==============================] 18s 9ms/step - loss: 0.0031 - mae: 0.0142 - mape: 0.0097 - val_loss: 0.0034 - val_mae: 0.0154 - val_mape: 0.0104
...
Epoch 45/100 (best)
2000/2000 [==============================] 18s 9ms/step - loss: 0.0028 - mae: 0.0126 - mape: 0.0088 - val_loss: 0.0029 - val_mae: 0.0128 - val_mape: 0.0089

Training completed!

================================================================================
PHASE 5: MODEL EVALUATION
================================================================================

Test Set Metrics:
  MAE:  0.0118 (1.18% average error)
  RMSE: 0.0165
  MAPE: 0.0089 (0.89%)
  Direction Accuracy: 0.562 (56.2%)

Model saved to /content/all_models/v1/crypto_v1_model.h5

================================================================================
TRAINING PIPELINE COMPLETED SUCCESSFULLY
================================================================================
```

### 5️⃣ Check Results (Cell 3)

```python
import os
from pathlib import Path

cache_dir = Path('/content/all_models/v1')

print("\n📁 Model Files:")
for f in cache_dir.glob('*.h5'):
    size_mb = f.stat().st_size / (1024**2)
    print(f"  {f.name} ({size_mb:.1f} MB)")

print("\n📊 Training Logs:")
for f in cache_dir.glob('*.log'):
    print(f"  {f.name}")

print("\n📈 Data Files:")
data_dir = cache_dir / 'crypto_data'
if data_dir.exists():
    parquet_files = list(data_dir.glob('*.parquet'))
    print(f"  {len(parquet_files)} Parquet files (1h + 15m data)")
    print(f"  Total size: {sum(f.stat().st_size for f in parquet_files) / (1024**2):.1f} MB")
```

**Expected output:**
```
📁 Model Files:
  best_model.h5 (2.5 MB)
  crypto_v1_model.h5 (2.5 MB)

📊 Training Logs:
  training_log_20251225_034500.log

📈 Data Files:
  44 Parquet files (1h + 15m data)
  Total size: 25.3 MB
```

---

## 📋 Frequently Asked Questions

### Q: Where is my model saved?
**A:** `/content/all_models/v1/best_model.h5` (Colab ephemeral, deleted after session)

**To save permanently:**
```python
from google.colab import files
import shutil

# Copy to Drive
shutil.copy(
    '/content/all_models/v1/best_model.h5',
    '/content/drive/MyDrive/crypto_model.h5'
)

print("✓ Model saved to Google Drive")
```

### Q: Can I stop training and resume later?
**A:** Not directly. Colab sessions timeout. Best practice:
- Save model to Google Drive after each training
- Use Google Drive for permanent storage

### Q: How to load the trained model?
```python
from tensorflow.keras.models import load_model

model = load_model('/content/all_models/v1/best_model.h5')

# Use for predictions
import numpy as np
X_1h = np.random.rand(1, 60, 50)  # 1 sample, 60 bars, 50 features
X_15m = np.random.rand(1, 60, 50)

predictions = model.predict([X_1h, X_15m])
print(f"Predictions (q10, q50, q90): {predictions[0]}")
```

### Q: Training is too slow, how to speed up?
**A:** 
- Switch to A100 GPU (faster, requires Pro)
- Reduce `EPOCHS` in config: `100 → 50`
- Reduce `BATCH_SIZE`: `32 → 16` (trains faster but less stable)

### Q: Got "CUDA out of memory" error?
**A:** 
```python
# In train_v1.py, change config:
BATCH_SIZE = 16  # Reduce from 32
LSTM_1H_UNITS = [32, 16]  # Smaller from [64, 32]
```

### Q: How to monitor training in real-time?
```python
# After training completes, plot curves
import matplotlib.pyplot as plt

# (Only works if you saved history variable)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Loss')
plt.legend(['Train', 'Val'])
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(history['mae'])
plt.plot(history['val_mae'])
plt.title('MAE')
plt.legend(['Train', 'Val'])
plt.xlabel('Epoch')

plt.tight_layout()
plt.show()
```

---

## 🛠️ Troubleshooting

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'pandas'` | Run colab_setup.py first (Cell 1) |
| `No GPU found` | Runtime → Change runtime type → Select T4 |
| `CUDA out of memory` | Reduce BATCH_SIZE: 32 → 16 |
| `ccxt API rate limit` | Wait 5 minutes, re-run (automatic retry built-in) |
| `Connection timeout on Binance` | Check internet, retry in 1 minute |
| `parquet file not found` | Ensure data_fetcher completed successfully |
| `ImportError: tensorflow` | Update: `!pip install --upgrade tensorflow` |

---

## 📚 Next Steps

1. **Backtesting** (v1.1)
   - Validate predictions on historical data
   - Calculate win rate, Sharpe ratio
   - Paper trade before real money

2. **Real-time Predictions** (v1.2)
   - Load model
   - Fetch latest 60-bar data
   - Generate predictions every 15 minutes

3. **Live Trading** (v1.3, use with caution!)
   - Connect to Binance Futures API
   - Auto-place orders
   - Risk management (stops, position sizing)

4. **v2 Improvements**
   - Transformer architecture
   - Ensemble methods
   - Sentiment analysis

---

## 📞 Support

- **GitHub**: [caizongxun/crypto_system](https://github.com/caizongxun/crypto_system)
- **Issues**: Report bugs [here](https://github.com/caizongxun/crypto_system/issues)
- **Discussions**: Ask questions [here](https://github.com/caizongxun/crypto_system/discussions)

---

**Happy training! 🎉**

*Last Updated: 2025-12-25*
