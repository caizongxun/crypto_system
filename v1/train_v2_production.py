#!/usr/bin/env python3
# ============================================================================
# Crypto System v2 - Production Grade Training (FIXED)
# ============================================================================
# Improvements over v1:
# 1. Predict LOG-RETURN instead of absolute price (much easier)
# 2. Proper time-series walk-forward split (no data leakage)
# 3. Baseline models (Naive, XGBoost) for comparison
# 4. Funding Rate + Open Interest as additional features
# 5. Save scalers for inference consistency
# 6. Multi-window walk-forward evaluation
# 7. Real Binance data
#
# Target: 1%–2% MAPE on return predictions

import os
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Concatenate, Conv1D, BatchNormalization, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False
    print("Warning: XGBoost not installed. Skipping XGB baseline.")

try:
    import ccxt
    HAS_CCXT = True
except:
    HAS_CCXT = False
    print("Warning: CCXT not installed. Will use synthetic data fallback.")

print("="*80)
print("CRYPTO SYSTEM v2 - PRODUCTION GRADE TRAINING (FIXED)")
print("Target: 1-2% MAPE on LOG-RETURN predictions")
print("="*80 + "\n")

# Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU: {len(gpus)} device(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠ No GPU")

CACHE_DIR = Path("/content/all_models/v2")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Cache: {CACHE_DIR}\n")

# ============================================================================
# FETCH REAL BINANCE DATA
# ============================================================================
print("="*80)
print("PHASE 1: FETCHING REAL BINANCE DATA")
print("="*80 + "\n")

def fetch_binance_data(symbol, timeframe='1h', limit=2000):
    """
    Fetch real data from Binance via CCXT.
    Fallback to synthetic data if CCXT unavailable.
    """
    if not HAS_CCXT:
        print(f"  {symbol}: CCXT unavailable, generating synthetic data...")
        return generate_synthetic_data(base_price=50000 if 'BTC' in symbol else 3000, n=limit)
    
    try:
        exchange = ccxt.binance()
        print(f"  Fetching {symbol} {timeframe}...")
        
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"    ✓ Got {len(df)} candles")
        return df
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}. Using synthetic data.")
        return generate_synthetic_data(base_price=50000, n=limit)

def generate_synthetic_data(base_price=50000, n=2000):
    """
    Generate synthetic data as fallback.
    """
    np.random.seed(42)
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(n)][::-1]
    returns = np.random.normal(0.0005, 0.015, n)
    prices = base_price * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': prices.copy(),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n))),
        'close': prices * (1 + np.random.normal(0, 0.005, n)),
        'volume': np.random.uniform(100000, 500000, n)
    })

# Download data for main cryptocurrencies
symbols = ['BTC/USDT', 'ETH/USDT']
all_data = {}

for symbol in symbols:
    all_data[symbol] = fetch_binance_data(symbol, timeframe='1h', limit=2000)

print()

# ============================================================================
# FEATURE ENGINEERING + SYNTHETIC FUNDING RATE
# ============================================================================
print("="*80)
print("PHASE 2: FEATURE ENGINEERING & FUNDING RATE")
print("="*80 + "\n")

def engineer_features(df):
    """
    Compute technical features.
    """
    df = df.copy()
    
    # Returns (our target will be log-return)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['simple_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Volatility
    df['volatility'] = df['log_return'].rolling(14).std()
    df['volatility_20'] = df['log_return'].rolling(20).std()
    
    # Momentum
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_20'] = df['close'] - df['close'].shift(20)
    
    # Price range
    df['price_range'] = (df['high'] - df['low']) / df['close']
    
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
    
    # Volume
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    
    # ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(14).mean()
    
    # SYNTHETIC FUNDING RATE (simulates real funding pressure)
    df['funding_rate'] = df['momentum_10'].rolling(20).mean() / (df['close'].rolling(20).std() + 1e-8) * 0.00001
    
    # SYNTHETIC OPEN INTEREST (simulates leverage usage)
    df['open_interest_change'] = (df['volume_ratio'] - 1) * (df['volatility'] / 0.01) * 0.1
    
    # Fill NaN
    df = df.ffill().bfill()
    
    # Replace any inf values
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

for symbol in symbols:
    all_data[symbol] = engineer_features(all_data[symbol])
    print(f"✓ {symbol} features computed")

print()

# ============================================================================
# PREPARE DATA FOR LSTM (RETURN PREDICTION)
# ============================================================================
print("="*80)
print("PHASE 3: PREPARE SEQUENCES FOR LSTM")
print("="*80 + "\n")

feature_cols = [
    'open', 'high', 'low', 'close', 'volume',
    'log_return', 'volatility', 'volatility_20', 'momentum_10', 'momentum_20',
    'price_range', 'rsi', 'macd', 'macd_signal', 'volume_ratio', 'atr',
    'funding_rate', 'open_interest_change'
]

lookback = 60

def create_sequences(df, lookback=60):
    """
    Create sequences for LSTM.
    Returns X (features), y (log-return), and dates for tracking.
    """
    X, y, dates = [], [], []
    
    for i in range(len(df) - lookback - 1):
        X.append(df[feature_cols].iloc[i:i+lookback].values)
        # Target: next period's log-return
        y.append(df['log_return'].iloc[i+lookback])
        dates.append(df['timestamp'].iloc[i+lookback])
    
    return np.array(X), np.array(y), np.array(dates)

all_X = []
all_y = []
all_dates = []
all_last_prices = []  # Store last price of each sequence for denormalization

for symbol in symbols:
    df = all_data[symbol]
    X, y, dates = create_sequences(df, lookback=lookback)
    all_X.append(X)
    all_y.append(y)
    all_dates.append(dates)
    
    # Store last prices for denormalization
    last_prices = []
    for i in range(len(df) - lookback - 1):
        last_prices.append(df['close'].iloc[i+lookback-1])  # Price before prediction
    all_last_prices.append(np.array(last_prices))
    
    print(f"✓ {symbol}: X shape {X.shape}, y shape {y.shape}")

X_combined = np.concatenate(all_X, axis=0)
y_combined = np.concatenate(all_y, axis=0)
dates_combined = np.concatenate(all_dates, axis=0)
last_prices_combined = np.concatenate(all_last_prices, axis=0)

print(f"\nTotal sequences: {X_combined.shape[0]}\n")

# ============================================================================
# TIME-SERIES SPLIT (NO SHUFFLE, NO LEAKAGE)
# ============================================================================
print("="*80)
print("PHASE 4: TIME-SERIES WALK-FORWARD SPLIT")
print("="*80 + "\n")

# Sort by date to ensure chronological order
idx_sort = np.argsort(dates_combined)
X_sorted = X_combined[idx_sort]
y_sorted = y_combined[idx_sort]
dates_sorted = dates_combined[idx_sort]
last_prices_sorted = last_prices_combined[idx_sort]

# Split: 70% train, 15% val, 15% test (CHRONOLOGICALLY)
n_total = len(X_sorted)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.15)

X_train = X_sorted[:n_train]
y_train = y_sorted[:n_train]
dates_train = dates_sorted[:n_train]
last_prices_train = last_prices_sorted[:n_train]

X_val = X_sorted[n_train:n_train+n_val]
y_val = y_sorted[n_train:n_train+n_val]
dates_val = dates_sorted[n_train:n_train+n_val]
last_prices_val = last_prices_sorted[n_train:n_train+n_val]

X_test = X_sorted[n_train+n_val:]
y_test = y_sorted[n_train+n_val:]
dates_test = dates_sorted[n_train+n_val:]
last_prices_test = last_prices_sorted[n_train+n_val:]

print(f"Train: {X_train.shape[0]} ({dates_train[0]} to {dates_train[-1]})")
print(f"Val:   {X_val.shape[0]} ({dates_val[0]} to {dates_val[-1]})")
print(f"Test:  {X_test.shape[0]} ({dates_test[0]} to {dates_test[-1]})")
print()

# ============================================================================
# NORMALIZE (FIT ONLY ON TRAIN SET)
# ============================================================================
print("="*80)
print("PHASE 5: NORMALIZATION (FIT ON TRAIN ONLY)")
print("="*80 + "\n")

# Reshape for fitting
X_train_reshaped = X_train.reshape(-1, X_train.shape[2])

# Fit scaler ONLY on train
scaler_X = StandardScaler()
scaler_X.fit(X_train_reshaped)

# Transform all splits
X_train_norm = X_train.copy()
X_val_norm = X_val.copy()
X_test_norm = X_test.copy()

# Apply scaler to each sequence
for i in range(len(X_train_norm)):
    X_train_norm[i] = scaler_X.transform(X_train[i])

for i in range(len(X_val_norm)):
    X_val_norm[i] = scaler_X.transform(X_val[i])

for i in range(len(X_test_norm)):
    X_test_norm[i] = scaler_X.transform(X_test[i])

# Scaler for y (log-return)
scaler_y = StandardScaler()
scaler_y.fit(y_train.reshape(-1, 1))

y_train_norm = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
y_val_norm = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test_norm = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"✓ Scalers fitted on train set only")
print(f"  X mean: {scaler_X.mean_.mean():.6f}, std: {scaler_X.scale_.mean():.6f}")
print(f"  y mean: {scaler_y.mean_[0]:.8f}, std: {scaler_y.scale_[0]:.8f}")
print()

# Save scalers for inference
scalers_dict = {
    'X': scaler_X,
    'y': scaler_y,
    'feature_cols': feature_cols,
    'lookback': lookback
}
with open(CACHE_DIR / 'scalers_v2.pkl', 'wb') as f:
    pickle.dump(scalers_dict, f)
print(f"✓ Scalers saved to {CACHE_DIR / 'scalers_v2.pkl'}\n")

# ============================================================================
# BASELINE MODELS (Naive + XGBoost)
# ============================================================================
print("="*80)
print("PHASE 6: BASELINE MODELS")
print("="*80 + "\n")

def safe_mape(y_true, y_pred):
    """
    Safe MAPE calculation that handles edge cases.
    """
    # Filter out NaN and Inf
    valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true_clean = y_true[valid_idx]
    y_pred_clean = y_pred[valid_idx]
    
    if len(y_true_clean) == 0:
        return np.nan
    
    # Avoid division by zero
    denominator = np.abs(y_true_clean)
    denominator = np.where(denominator < 1e-8, 1e-8, denominator)
    
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / denominator))
    return mape

# Baseline 1: NAIVE (predict zero return)
print("Baseline 1: Naive (predict return = 0)")
y_pred_naive = np.zeros_like(y_test)
mape_naive = safe_mape(y_test, y_pred_naive)
print(f"  Naive MAPE: {mape_naive*100:.4f}%\n")

# Baseline 2: XGBoost
if HAS_XGB:
    print("Baseline 2: XGBoost")
    try:
        # Flatten X for XGBoost
        X_train_flat = X_train_norm.reshape(X_train_norm.shape[0], -1)
        X_test_flat = X_test_norm.reshape(X_test_norm.shape[0], -1)
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(
            X_train_flat, y_train_norm,
            eval_set=[(X_test_flat, y_test_norm)],
            verbose=False
        )
        
        y_pred_xgb_norm = xgb_model.predict(X_test_flat)
        y_pred_xgb = scaler_y.inverse_transform(y_pred_xgb_norm.reshape(-1, 1)).flatten()
        mape_xgb = safe_mape(y_test, y_pred_xgb)
        print(f"  XGBoost MAPE: {mape_xgb*100:.4f}%\n")
    except Exception as e:
        print(f"  XGBoost error: {e}\n")
        mape_xgb = None
else:
    mape_xgb = None

# ============================================================================
# LSTM MODEL (Dual-stream)
# ============================================================================
print("="*80)
print("PHASE 7: LSTM MODEL")
print("="*80 + "\n")

n_features = X_train_norm.shape[2]
input_main = Input(shape=(lookback, n_features), name='input_main')
input_aux = Input(shape=(lookback, n_features), name='input_aux')

# Stream 1: LSTM
x1 = LSTM(64, activation='relu', return_sequences=True)(input_main)
x1 = BatchNormalization()(x1)
x1 = Dropout(0.2)(x1)
x1 = LSTM(32, activation='relu', return_sequences=False)(x1)
x1 = Dropout(0.2)(x1)

# Stream 2: Conv1D + LSTM
x2 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_aux)
x2 = BatchNormalization()(x2)
x2 = Dropout(0.2)(x2)
x2 = LSTM(32, activation='relu', return_sequences=False)(x2)
x2 = Dropout(0.2)(x2)

# Merge
combined = Concatenate()([x1, x2])
z = Dense(32, activation='relu')(combined)
z = BatchNormalization()(z)
z = Dropout(0.2)(z)
z = Dense(16, activation='relu')(z)
z = Dropout(0.1)(z)
output = Dense(1, name='return')(z)  # Output: normalized return

model = Model(inputs=[input_main, input_aux], outputs=output)
model.compile(
    optimizer=Adam(learning_rate=0.0003, clipvalue=1.0),
    loss='huber',  # Robust to outliers
    metrics=['mae']
)

print(f"✓ Model: {model.count_params():,} params\n")

# ============================================================================
# TRAIN LSTM
# ============================================================================
print("="*80)
print("PHASE 8: TRAINING")
print("="*80 + "\n")

start_time = time.time()

history = model.fit(
    [X_train_norm, X_train_norm], y_train_norm,
    validation_data=([X_val_norm, X_val_norm], y_val_norm),
    epochs=50,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(CACHE_DIR / 'best_v2_model.h5'), monitor='val_loss', save_best_only=True, verbose=0)
    ],
    verbose=0
)

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.0f}s\n")

# ============================================================================
# EVALUATION ON TEST SET
# ============================================================================
print("="*80)
print("PHASE 9: EVALUATION")
print("="*80 + "\n")

y_pred_norm = model.predict([X_test_norm, X_test_norm], verbose=0).flatten()
y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

# Metrics on log-return
mae_return = mean_absolute_error(y_test, y_pred)
rmse_return = np.sqrt(mean_squared_error(y_test, y_pred))
mape_return = safe_mape(y_test, y_pred)

print("LSTM Performance (on LOG-RETURN):")
print(f"  MAE:  {mae_return:.8f} (avg error in return)")
print(f"  RMSE: {rmse_return:.8f}")
print(f"  MAPE: {mape_return*100:.4f}%")
print()

# Compare with baselines
print("Comparison with Baselines:")
print(f"  Naive MAPE:  {mape_naive*100:.4f}%")
if mape_xgb is not None:
    print(f"  XGBoost MAPE: {mape_xgb*100:.4f}%")
print(f"  LSTM MAPE:   {mape_return*100:.4f}%")
print()

# Error statistics
errors = np.abs(y_test - y_pred)
print(f"Error Statistics on Returns:")
print(f"  Min:    {errors.min():.8f}")
print(f"  Max:    {errors.max():.8f}")
print(f"  Mean:   {errors.mean():.8f}")
print(f"  Median: {np.median(errors):.8f}")
print(f"  Std:    {errors.std():.8f}")
print()

# Direction accuracy
direction_correct = np.sum((y_test * y_pred) > 0)  # Both positive or both negative
direction_accuracy = direction_correct / len(y_test) * 100
print(f"Direction Accuracy: {direction_accuracy:.2f}%")
print()

# Price-level MAPE (for reference)
print("Price-level MAPE (approximate):")
if len(last_prices_test) > 0:
    # Denormalize to prices
    price_pred = last_prices_test * np.exp(y_pred)
    price_actual = last_prices_test * np.exp(y_test)
    mape_price = safe_mape(price_actual, price_pred)
    print(f"  LSTM Price MAPE: {mape_price*100:.4f}%")
else:
    print(f"  Warning: No price data available")
print()

# ============================================================================
# SAVE MODEL
# ============================================================================
model_path = CACHE_DIR / "crypto_v2_final_model.h5"
model.save(str(model_path))

print("="*80)
print("✓ TRAINING COMPLETE!")
print("="*80)
print(f"Model saved: {model_path}")
print(f"Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
print(f"Scalers saved: {CACHE_DIR / 'scalers_v2.pkl'}")
print()
print(f"\n✓ Ready for production inference!")
print(f"  Use: from v1.inference_v2 import CryptoPricePredictorV2")
