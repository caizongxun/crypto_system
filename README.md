# 🚀 Crypto Price Prediction System

A multi-currency cryptocurrency price prediction system using deep learning (LSTM + CNN) with dual-timeframe architecture (1h + 15m) designed to handle extreme market volatility.

## 📋 Overview

**Version**: v1  
**Status**: Development  
**GPU Support**: NVIDIA CUDA (optimized for Google Colab)  

### Key Features

- **20+ Cryptocurrency Pairs**: BTC, ETH, SOL, XRP, ADA, DOGE, LINK, MATIC, AVAX, DOT, UNI, LTC, BCH, ETC, XLM, ATOM, ICP, NEAR, ARB, SHIB, OP, PEPE
- **Multi-Timeframe Architecture**: 
  - 1h stream (trends)
  - 15m stream (volatility detection)
- **Volatility-Aware**: Quantile regression for price distribution prediction
- **High Frequency Data**: 1000+ historical candles per pair
- **Production Ready**: Colab GPU training with checkpoint saving

## 📁 Repository Structure

```
crypto_system/
├── README.md                 # This file
├── .gitignore               # Git ignore rules
│
└── v1/                      # Version 1 (current)
    ├── config.py            # All hyperparameters and settings
    ├── data_fetcher.py      # Binance API integration
    ├── feature_engineering.py # Technical indicators & volatility features
    ├── train_v1.py          # Main training script
    └── README_V1.md         # v1 specific documentation

(Future versions: v2/, v3/, ... will follow same structure)
```

## 🛠️ Setup & Installation

### Option 1: Colab (Recommended)

#### Step 1: Open Google Colab

Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

#### Step 2: Enable GPU

```python
# In first cell
from google.colab import drive
drive.mount('/content/drive')

# Runtime → Change runtime type → GPU (T4 or A100)
```

#### Step 3: Remote Execution

Paste this single cell to execute the entire training pipeline:

```python
import requests
import time

url = 'https://raw.githubusercontent.com/caizongxun/crypto_system/main/v1/train_v1.py?t=' + str(int(time.time()))
exec(requests.get(url).text)
```

**What happens:**
1. Downloads all v1 modules from GitHub
2. Fetches 1000+ candles per coin from Binance US API
3. Processes technical indicators and volatility features
4. Builds dual-stream LSTM model
5. Trains on GPU (T4 ~30-40 mins, A100 ~10-15 mins)
6. Saves model to `/content/all_models/v1/`

### Option 2: Local (Development)

```bash
# Clone repository
git clone https://github.com/caizongxun/crypto_system.git
cd crypto_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training
cd v1
python train_v1.py
```

## 📚 Configuration

All settings are in `v1/config.py`. Key parameters:

```python
# Trading Pairs (22 cryptocurrencies)
TRADING_PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', ...]

# Timeframes
TIMEFRAMES = {'trend': '1h', 'volatility': '15m'}

# Model Architecture
LSTM_1H_UNITS = [64, 32]
LSTM_15M_UNITS = [32, 16]
CNN_FILTERS = [32, 16]

# Training
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LEARNING_RATE = 0.001

# Volatility Handling
LOSS_FUNCTION = "weighted_mse"
QUANTILE_REGRESSION = True  # Predict price distribution
```

## 🔄 Data Flow

### 1. Data Fetching (`data_fetcher.py`)

```
Binance US API (ccxt)
    ↓
1000 candles × 22 coins × 2 timeframes (1h + 15m)
    ↓
Parquet storage (/content/all_models/v1/crypto_data/)
```

**Why Parquet?** 10x faster than CSV, compressed size, preserves types.

### 2. Feature Engineering (`feature_engineering.py`)

```
Raw OHLCV
    ↓
Log Returns (normalize scale for 20+ coins)
    ↓
Volatility Features (ATR, Bollinger Bands, Rolling Volatility)
    ↓
Technical Indicators (RSI, MACD, Momentum, Volume Ratio)
    ↓
Z-Score Normalization [0, 1]
    ↓
Sequence Creation (60-bar windows)
```

**Key for volatility**: Log returns instead of prices, rolling volatility, ATR.

### 3. Model Architecture

```
                    Input Layer
                         |
                 ________|________
                |                 |
        1h Trend Branch    15m Volatility Branch
                |                 |
        LSTM(64) → Dropout   CNN(32) → LSTM(32)
        LSTM(32) → Dropout   Dropout
                |                 |
                └────Concatenate──┘
                         |
                    Dense(32)
                    Dense(16)
                         |
                    Output Layer
        (Quantile: [q10, q50, q90])
```

### 4. Training & Evaluation

```
Train/Val/Test Split (70/15/15)
    ↓
Train on GPU (T4: 30-40 mins, A100: 10-15 mins)
    ↓
Evaluate on Test Set
Metrics: MAE, RMSE, MAPE, Direction Accuracy
    ↓
Save Best Model to /content/all_models/v1/
```

## 📊 Expected Performance

| Metric | Target | Status |
|--------|--------|--------|
| MAPE (Mean Absolute % Error) | < 5% | Training |
| Direction Accuracy | > 55% | Training |
| Model Size | < 50MB | ✓ |
| Training Time (GPU T4) | 30-40 min | Expected |

## 🎯 Usage After Training

### Inference on New Data

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load trained model
model = load_model('/content/all_models/v1/best_model.h5')

# Prepare data (60-bar windows for 1h and 15m)
X_1h = np.random.rand(1, 60, n_features)  # 1 sample, 60 bars, features
X_15m = np.random.rand(1, 60, n_features)

# Predict
predictions = model.predict([X_1h, X_15m])

# If quantile regression:
q10, q50, q90 = predictions[0]
print(f"Predicted range: {q10:.2f} - {q90:.2f} (median: {q50:.2f})")
```

### Trading Logic (Future)

```python
# Find relative extrema in 10-bar window
if q10 == min_price_in_window:  # Relative low
    OPEN_LONG()
elif q90 == max_price_in_window:  # Relative high
    OPEN_SHORT()

# Risk management
if volatility_ratio > 2.0:  # High volatility detected
    REDUCE_POSITION_SIZE()
    or
    PAUSE_TRADING()
```

## 🚨 Important Notes

### Rate Limiting
- Binance API: 1000 requests per 10 minutes
- Implementation: 100ms delay between requests (automatic)
- Fetching 22 pairs × 2 timeframes: ~2-3 minutes

### GPU Memory
- Colab T4: 16GB (sufficient)
- Colab A100: 40GB (very fast)
- Mixed precision training enabled (float16)

### Data Persistence
```
Colab ephemeral storage:
  /content/all_models/v1/ (deleted after session)
  
Persistent storage (optional):
  Google Drive: /MyDrive/crypto_system/models/ (manual backup)
```

## 🔄 Version Management

**Repository Structure Principles:**

1. **Root directory**: Only v1, v2, v3... folders
2. **Each version**: Self-contained with config, data_fetcher, feature_engineering, train script
3. **Remote execution**: `https://raw.githubusercontent.com/caizongxun/crypto_system/main/vN/train_vN.py`
4. **Cache storage**: `/content/all_models/vN/` in Colab

**Adding v2 (Future):**
```bash
mkdir v2
cp v1/config.py v2/config.py  # Copy and modify
cp v1/data_fetcher.py v2/
cp v1/feature_engineering.py v2/
# Create v2/train_v2.py with new architecture
```

## 📦 Dependencies

```
tensorflow>=2.13.0
keras>=2.13.0
ccxt>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pyarrow>=12.0.0  # For parquet support
```

## 🐛 Troubleshooting

### GPU Not Found
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# If empty: Runtime → Change runtime type → T4 or A100
```

### Out of Memory
```python
# Reduce batch size in config.py
BATCH_SIZE = 16  # Default is 32
```

### Binance API Error
```python
# Check if market is open
# Retry with exponential backoff (automatic in data_fetcher.py)
```

## 📝 Logging

Training logs saved to:
```
/content/all_models/v1/training_log_YYYYMMDD_HHMMSS.log
```

Visualize training progress:
```python
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.show()
```

## 🤝 Contributing

Improvements welcome:

1. Fork repository
2. Create feature branch (`git checkout -b feature/new_feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new_feature`)
5. Open Pull Request

## 📄 License

MIT License - See LICENSE file

## 📧 Contact

- GitHub: [@caizongxun](https://github.com/caizongxun)
- Repo: [crypto_system](https://github.com/caizongxun/crypto_system)

---

**Last Updated**: 2025-12-25  
**Maintained By**: @caizongxun
