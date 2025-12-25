# ============================================================================
# Crypto System v1 - Minimal Training Script (For Testing)
# ============================================================================
# Fast version: Uses synthetic data instead of API fetching
# Good for testing the pipeline without waiting for data download
# 
# Usage in Colab (after setup.py):
#   import requests, time
#   url = 'https://raw.githubusercontent.com/caizongxun/crypto_system/main/v1/train_v1_minimal.py?t=' + str(int(time.time()))
#   exec(requests.get(url).text)

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Concatenate, Conv1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================================
# SETUP
# ============================================================================
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

print("\n" + "="*80)
print("CRYPTO SYSTEM v1 - MINIMAL TRAINING (Synthetic Data)")
print("="*80 + "\n")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logger.info(f"✓ GPU: {len(gpus)} device(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    logger.warning("⚠ No GPU found - Training will be slow")

CACHE_DIR = Path("/content/all_models/v1")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"✓ Cache: {CACHE_DIR}\n")

# ============================================================================
# PHASE 1: GENERATE SYNTHETIC DATA
# ============================================================================
logger.info("="*80)
logger.info("PHASE 1: GENERATING SYNTHETIC DATA")
logger.info("="*80 + "\n")

def generate_synthetic_ohlcv(n_samples=2000, base_price=90000):
    """
    Generate realistic synthetic OHLCV data.
    
    Args:
        n_samples (int): Number of candles
        base_price (float): Starting price
        
    Returns:
        pd.DataFrame: Synthetic OHLCV data
    """
    np.random.seed(42)
    
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(n_samples)][::-1]
    
    # Generate price movement
    returns = np.random.normal(0.0005, 0.015, n_samples)  # Small drift, some volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV
    opens = prices.copy()
    closes = prices * (1 + np.random.normal(0, 0.005, n_samples))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.003, n_samples)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.003, n_samples)))
    volumes = np.random.uniform(100000, 500000, n_samples)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    return df

# Generate data for multiple coins
logger.info("Generating synthetic OHLCV data...")
symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
all_data = {}

for symbol in symbols:
    all_data[symbol] = generate_synthetic_ohlcv()
    logger.info(f"  ✓ {symbol}: {len(all_data[symbol])} candles")

print()

# ============================================================================
# PHASE 2: FEATURE ENGINEERING
# ============================================================================
logger.info("="*80)
logger.info("PHASE 2: FEATURE ENGINEERING")
logger.info("="*80 + "\n")

def engineer_features(df):
    """
    Add technical indicators to data.
    """
    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    df['volatility'] = df['log_return'].rolling(14).std()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Volume ratio
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    
    # ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(14).mean()
    
    # Fill NaNs
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

logger.info("Adding technical indicators...")
for symbol in symbols:
    all_data[symbol] = engineer_features(all_data[symbol])
    logger.info(f"  ✓ {symbol} processed")

print()

# ============================================================================
# PHASE 3: PREPARE SEQUENCES
# ============================================================================
logger.info("="*80)
logger.info("PHASE 3: PREPARING SEQUENCES")
logger.info("="*80 + "\n")

def create_sequences(df, lookback=60):
    """
    Create time-series sequences.
    """
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                    'log_return', 'volatility', 'rsi', 'macd', 'volume_ratio', 'atr']
    
    X = []
    y = []
    
    for i in range(len(df) - lookback - 1):
        X.append(df[feature_cols].iloc[i:i+lookback].values)
        y.append(df['close'].iloc[i+lookback])
    
    return np.array(X), np.array(y)

logger.info("Creating sequences (lookback=60)...")
all_X = []
all_y = []

for symbol in symbols:
    X, y = create_sequences(all_data[symbol], lookback=60)
    all_X.append(X)
    all_y.append(y)
    logger.info(f"  ✓ {symbol}: {X.shape}")

X_combined = np.concatenate(all_X, axis=0)
y_combined = np.concatenate(all_y, axis=0)

logger.info(f"\nCombined: X {X_combined.shape}, y {y_combined.shape}")

# Split data
n_train = int(len(X_combined) * 0.7)
n_val = int(len(X_combined) * 0.15)

X_train, X_val, X_test = X_combined[:n_train], X_combined[n_train:n_train+n_val], X_combined[n_train+n_val:]
y_train, y_val, y_test = y_combined[:n_train], y_combined[n_train:n_train+n_val], y_combined[n_train+n_val:]

logger.info(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

print()

# ============================================================================
# PHASE 4: BUILD MODEL
# ============================================================================
logger.info("="*80)
logger.info("PHASE 4: BUILDING MODEL")
logger.info("="*80 + "\n")

n_features = X_train.shape[2]

# Dual-stream inputs (for compatibility with production code)
input_1h = Input(shape=(60, n_features), name='input_1h')
input_15m = Input(shape=(60, n_features), name='input_15m')

# Stream 1: LSTM
x1 = LSTM(64, activation='relu', return_sequences=True)(input_1h)
x1 = Dropout(0.2)(x1)
x1 = LSTM(32, activation='relu', return_sequences=False)(x1)
x1 = Dropout(0.2)(x1)

# Stream 2: CNN + LSTM
x2 = Conv1D(filters=32, kernel_size=3, activation='relu')(input_15m)
x2 = Dropout(0.2)(x2)
x2 = LSTM(32, activation='relu', return_sequences=False)(x2)
x2 = Dropout(0.2)(x2)

# Merge
combined = Concatenate()([x1, x2])
z = Dense(32, activation='relu')(combined)
z = Dropout(0.2)(z)
z = Dense(16, activation='relu')(z)
output = Dense(1, name='price')(z)

model = Model(inputs=[input_1h, input_15m], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

logger.info("\nModel Summary:")
logger.info(f"Total params: {model.count_params():,}")

print()

# ============================================================================
# PHASE 5: TRAIN
# ============================================================================
logger.info("="*80)
logger.info("PHASE 5: TRAINING")
logger.info("="*80 + "\n")

start_time = time.time()

history = model.fit(
    [X_train, X_train], y_train,
    validation_data=([X_val, X_val], y_val),
    epochs=20,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ],
    verbose=1
)

training_time = time.time() - start_time
logger.info(f"\nTraining completed in {training_time:.1f} seconds")

print()

# ============================================================================
# PHASE 6: EVALUATE
# ============================================================================
logger.info("="*80)
logger.info("PHASE 6: EVALUATION")
logger.info("="*80 + "\n")

y_pred = model.predict([X_test, X_test], verbose=0)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100

logger.info(f"Test Metrics:")
logger.info(f"  MAE:  ${mae:.2f}")
logger.info(f"  RMSE: ${rmse:.2f}")
logger.info(f"  MAPE: {mape:.2f}%")

print()

# ============================================================================
# PHASE 7: SAVE MODEL
# ============================================================================
logger.info("="*80)
logger.info("PHASE 7: SAVING MODEL")
logger.info("="*80 + "\n")

model_path = CACHE_DIR / "crypto_v1_minimal_model.h5"
model.save(str(model_path))
logger.info(f"✓ Model saved: {model_path}")
logger.info(f"  Size: {model_path.stat().st_size / (1024*1024):.1f} MB")

print()

# ============================================================================
# SUMMARY
# ============================================================================
logger.info("="*80)
logger.info("✓ TRAINING COMPLETE!")
logger.info("="*80)
logger.info(f"")
logger.info(f"Total time: {training_time:.1f}s")
logger.info(f"Model: {model_path.name}")
logger.info(f"")
logger.info(f"Next steps:")
logger.info(f"1. Use full training with real data:")
logger.info(f"   url = 'https://raw.githubusercontent.com/caizongxun/crypto_system/main/v1/train_v1.py?t=' + str(int(time.time()))")
logger.info(f"   exec(requests.get(url).text)")
logger.info(f"")
logger.info(f"2. Load and use this model:")
logger.info(f"   from tensorflow.keras.models import load_model")
logger.info(f"   model = load_model('{model_path}')")
logger.info(f"")
