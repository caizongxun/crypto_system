# ============================================================================
# Crypto System v1 - Direct Colab Training (Copy-Paste Ready)
# ============================================================================
# 直接貼到 Colab Cell 中執行，無需 requests 或 exec()
# 這樣避免了 exec() 導致的環境問題
#
# 步驟：
# 1. 在 Colab 中新建 Cell
# 2. 複製貼上下面的所有代碼
# 3. 執行 (Ctrl+Enter)

# ============================================================================
# PART 1: INSTALL + IMPORT (必須在最開始)
# ============================================================================
import subprocess
import sys

print("Installing packages...")
for pkg in ["pandas", "numpy", "tensorflow", "scikit-learn", "ccxt", "pyarrow", "pandas-ta", "matplotlib"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print(f"  ✓ {pkg}")

print("\nImporting...")
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("✓ All ready!\n")

# ============================================================================
# SETUP
# ============================================================================
print("="*80)
print("CRYPTO SYSTEM v1 - TRAINING")
print("="*80 + "\n")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU: {len(gpus)} device(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠ No GPU")

CACHE_DIR = Path("/content/all_models/v1")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Cache: {CACHE_DIR}\n")

# ============================================================================
# DATA GENERATION
# ============================================================================
print("Generating synthetic data...")

np.random.seed(42)
symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
all_data = {}

for symbol in symbols:
    n = 2000
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(n)][::-1]
    returns = np.random.normal(0.0005, 0.015, n)
    prices = 90000 * np.exp(np.cumsum(returns))
    
    opens = prices.copy()
    closes = prices * (1 + np.random.normal(0, 0.005, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.003, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.003, n)))
    volumes = np.random.uniform(100000, 500000, n)
    
    all_data[symbol] = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    print(f"  ✓ {symbol}")

print()

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("Computing features...")

for symbol in symbols:
    df = all_data[symbol]
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_return'].rolling(14).std()
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(14).mean()
    
    all_data[symbol] = df.fillna(method='ffill').fillna(method='bfill')
    print(f"  ✓ {symbol}")

print()

# ============================================================================
# CREATE SEQUENCES
# ============================================================================
print("Creating sequences...")

feature_cols = ['open', 'high', 'low', 'close', 'volume', 'log_return', 'volatility', 'rsi', 'macd', 'volume_ratio', 'atr']
all_X = []
all_y = []

for symbol in symbols:
    df = all_data[symbol]
    X, y = [], []
    
    for i in range(len(df) - 60 - 1):
        X.append(df[feature_cols].iloc[i:i+60].values)
        y.append(df['close'].iloc[i+60])
    
    all_X.append(np.array(X))
    all_y.append(np.array(y))
    print(f"  ✓ {symbol}: {np.array(X).shape}")

X_combined = np.concatenate(all_X, axis=0)
y_combined = np.concatenate(all_y, axis=0)

n_train = int(len(X_combined) * 0.7)
n_val = int(len(X_combined) * 0.15)

X_train, X_val, X_test = X_combined[:n_train], X_combined[n_train:n_train+n_val], X_combined[n_train+n_val:]
y_train, y_val, y_test = y_combined[:n_train], y_combined[n_train:n_train+n_val], y_combined[n_train+n_val:]

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}\n")

# ============================================================================
# BUILD MODEL
# ============================================================================
print("Building model...")

n_features = X_train.shape[2]
input_1h = Input(shape=(60, n_features), name='input_1h')
input_15m = Input(shape=(60, n_features), name='input_15m')

x1 = LSTM(64, activation='relu', return_sequences=True)(input_1h)
x1 = Dropout(0.2)(x1)
x1 = LSTM(32, activation='relu', return_sequences=False)(x1)
x1 = Dropout(0.2)(x1)

x2 = Conv1D(filters=32, kernel_size=3, activation='relu')(input_15m)
x2 = Dropout(0.2)(x2)
x2 = LSTM(32, activation='relu', return_sequences=False)(x2)
x2 = Dropout(0.2)(x2)

combined = Concatenate()([x1, x2])
z = Dense(32, activation='relu')(combined)
z = Dropout(0.2)(z)
z = Dense(16, activation='relu')(z)
output = Dense(1, name='price')(z)

model = Model(inputs=[input_1h, input_15m], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print(f"✓ Model: {model.count_params():,} params\n")

# ============================================================================
# TRAIN
# ============================================================================
print("Training...")
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
print(f"\n✓ Training completed in {training_time:.0f}s\n")

# ============================================================================
# EVALUATE
# ============================================================================
print("Evaluating...")
y_pred = model.predict([X_test, X_test], verbose=0)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100

print(f"  MAE:  ${mae:.2f}")
print(f"  RMSE: ${rmse:.2f}")
print(f"  MAPE: {mape:.2f}%\n")

# ============================================================================
# SAVE
# ============================================================================
model_path = CACHE_DIR / "crypto_v1_model.h5"
model.save(str(model_path))

print("="*80)
print("✓ SUCCESS!")
print("="*80)
print(f"Model saved: {model_path}")
print(f"Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
print()
print("Next: Train with real Binance data (45-60 min)")
print()
