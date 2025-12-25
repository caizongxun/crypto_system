#!/usr/bin/env python3
# ============================================================================
# Crypto System v2 - Multi-Crypto Multi-Timeframe Training (ENHANCED)
# ============================================================================
# Trains individual models for 20+ cryptocurrencies
# Timeframes: 1h (1-hour) and 15m (15-minute)
# Data: 7000-10000 candles per timeframe
# Multiple Binance endpoints for maximum data coverage
#
# Output: /content/all_models/multi_crypto/
#   - BTC_1h_model.h5
#   - BTC_1h_scalers.pkl
#   - BTC_15m_model.h5
#   - BTC_15m_scalers.pkl
#   - ... (for each of 20+ coins)

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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Concatenate, Conv1D, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import ccxt
    HAS_CCXT = True
except:
    HAS_CCXT = False
    print("WARNING: CCXT not installed. Installing...")
    os.system('pip install ccxt -q')
    import ccxt
    HAS_CCXT = True

print("="*80)
print("CRYPTO SYSTEM v2 - ENHANCED MULTI-CRYPTO MULTI-TIMEFRAME TRAINING")
print("20+ Cryptocurrencies × 2 Timeframes × 7000-10000 Candles")
print("="*80 + "\n")

# Setup GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU: {len(gpus)} device(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠ No GPU detected")

# Output directory
OUTPUT_DIR = Path("/content/all_models/multi_crypto")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Output directory: {OUTPUT_DIR}\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

# 24 Cryptocurrencies (select popular ones)
CRYPTOS = [
    'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'AVAX', 'MATIC',
    'LTC', 'DOT', 'UNI', 'LINK', 'XLM', 'ATOM', 'ICP', 'FIL', 'NEAR',
    'SAND', 'MANA', 'SHIB', 'OP', 'ARB', 'BLUR'
]

# Timeframes
TIMEFRAMES = ['1h', '15m']

# Training config
LOOKBACK = 60  # Number of candles to look back
EPOCHS = 40
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 8

# Data fetching
CANDLES_PER_FETCH = 1000  # CCXT limit per request
TARGET_CANDLES = 8000     # Target total candles
NUM_FETCHES = (TARGET_CANDLES + CANDLES_PER_FETCH - 1) // CANDLES_PER_FETCH

print(f"Cryptocurrencies to train: {len(CRYPTOS)}")
print(f"  {', '.join(CRYPTOS[:5])}... (+{len(CRYPTOS)-5} more)")
print(f"Timeframes: {TIMEFRAMES}")
print(f"Target candles per timeframe: {TARGET_CANDLES}")
print(f"Total models to train: {len(CRYPTOS) * len(TIMEFRAMES)}")
print(f"Lookback period: {LOOKBACK} candles\n")

# ============================================================================
# BINANCE ENDPOINTS
# ============================================================================

BINANCE_ENDPOINTS = [
    ('binance', 'Binance Global'),
    ('binanceus', 'Binance US'),
    ('binanceusdm', 'Binance USDM'),
]

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """
    Compute technical features for all cryptocurrencies.
    """
    df = df.copy()
    
    # Returns
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
    
    # Funding rate proxy
    df['funding_rate'] = df['momentum_10'].rolling(20).mean() / (df['close'].rolling(20).std() + 1e-8) * 0.00001
    
    # Open interest proxy
    df['open_interest_change'] = (df['volume_ratio'] - 1) * (df['volatility'] / 0.01) * 0.1
    
    # Fill NaN
    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

# ============================================================================
# FETCH DATA FOR SINGLE CRYPTO (MULTI-ENDPOINT)
# ============================================================================

def fetch_crypto_data_multi_endpoint(symbol, timeframe):
    """
    Fetch OHLCV data from multiple Binance endpoints.
    Tries each endpoint until successful.
    """
    all_ohlcv = []
    fetched_candles = 0
    
    for exchange_name, exchange_label in BINANCE_ENDPOINTS:
        if fetched_candles >= TARGET_CANDLES:
            break
        
        try:
            print(f"      {exchange_label}...", end=" ")
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({'enableRateLimit': True})
            
            symbol_full = f"{symbol}/USDT"
            
            # Check if symbol is supported
            if hasattr(exchange, 'symbols'):
                if symbol_full not in exchange.symbols:
                    print(f"Symbol not found")
                    continue
            
            # Fetch multiple times to accumulate candles
            ohlcv_data = []
            since = None
            
            for fetch_num in range(NUM_FETCHES):
                if fetched_candles >= TARGET_CANDLES:
                    break
                
                try:
                    ohlcv = exchange.fetch_ohlcv(
                        symbol_full,
                        timeframe,
                        since=since,
                        limit=CANDLES_PER_FETCH
                    )
                    
                    if not ohlcv or len(ohlcv) == 0:
                        break
                    
                    ohlcv_data.extend(ohlcv)
                    fetched_candles += len(ohlcv)
                    
                    # Update since for next batch
                    since = ohlcv[-1][0] + 1
                    
                    time.sleep(0.5)  # Rate limit
                    
                except Exception as e:
                    print(f"Fetch error: {str(e)[:30]}")
                    break
            
            if ohlcv_data:
                all_ohlcv.extend(ohlcv_data)
                print(f"{len(ohlcv_data)} candles")
        
        except Exception as e:
            print(f"Error: {str(e)[:40]}")
            continue
    
    if not all_ohlcv:
        print(f"      NO DATA from any endpoint")
        return None
    
    # Remove duplicates and sort
    df = pd.DataFrame(
        all_ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

# ============================================================================
# BUILD LSTM MODEL
# ============================================================================

def build_lstm_model(n_features, lookback):
    """
    Build the dual-stream LSTM model.
    """
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
    output = Dense(1, name='return')(z)
    
    model = Model(inputs=[input_main, input_aux], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.0003, clipvalue=1.0),
        loss='huber',
        metrics=['mae']
    )
    
    return model

# ============================================================================
# TRAIN SINGLE CRYPTO-TIMEFRAME PAIR
# ============================================================================

def train_crypto_model(symbol, timeframe):
    """
    Train model for a single cryptocurrency and timeframe.
    Returns: (success, mape)
    """
    print(f"\n{'='*80}")
    print(f"Training: {symbol} {timeframe}")
    print(f"{'='*80}")
    
    # 1. Fetch data from multiple endpoints
    print(f"  Phase 1: Fetching data from multiple endpoints...")
    df = fetch_crypto_data_multi_endpoint(symbol, timeframe)
    
    if df is None or len(df) < LOOKBACK + 100:
        print(f"  SKIP: Not enough data ({len(df) if df is not None else 0} candles)")
        return False, None
    
    print(f"    Total candles collected: {len(df)}")
    
    # 2. Feature engineering
    print(f"  Phase 2: Computing features...", end=" ")
    df = engineer_features(df)
    print("Done")
    
    # 3. Create sequences
    print(f"  Phase 3: Creating sequences...", end=" ")
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'log_return', 'volatility', 'volatility_20', 'momentum_10', 'momentum_20',
        'price_range', 'rsi', 'macd', 'macd_signal', 'volume_ratio', 'atr',
        'funding_rate', 'open_interest_change'
    ]
    
    X, y, base_prices = [], [], []
    for i in range(len(df) - LOOKBACK - 1):
        X.append(df[feature_cols].iloc[i:i+LOOKBACK].values)
        y.append(df['log_return'].iloc[i+LOOKBACK])
        base_prices.append(df['close'].iloc[i+LOOKBACK-1])
    
    X = np.array(X)
    y = np.array(y)
    base_prices = np.array(base_prices)
    print(f"{len(X)} sequences")
    
    # 4. Time-series split
    n_total = len(X)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    base_prices_test = base_prices[n_train+n_val:]
    
    # 5. Normalize
    print(f"  Phase 4: Normalizing...", end=" ")
    X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
    scaler_X = StandardScaler()
    scaler_X.fit(X_train_reshaped)
    
    X_train_norm = X_train.copy()
    X_val_norm = X_val.copy()
    X_test_norm = X_test.copy()
    
    for i in range(len(X_train_norm)):
        X_train_norm[i] = scaler_X.transform(X_train[i])
    for i in range(len(X_val_norm)):
        X_val_norm[i] = scaler_X.transform(X_val[i])
    for i in range(len(X_test_norm)):
        X_test_norm[i] = scaler_X.transform(X_test[i])
    
    scaler_y = StandardScaler()
    scaler_y.fit(y_train.reshape(-1, 1))
    y_train_norm = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
    y_val_norm = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_norm = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    print("Done")
    
    # 6. Train model
    print(f"  Phase 5: Training model...", end=" ")
    model = build_lstm_model(X_train_norm.shape[2], LOOKBACK)
    
    start_time = time.time()
    history = model.fit(
        [X_train_norm, X_train_norm], y_train_norm,
        validation_data=([X_val_norm, X_val_norm], y_val_norm),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=0),
        ],
        verbose=0
    )
    train_time = time.time() - start_time
    print(f"{train_time:.0f}s")
    
    # 7. Evaluate
    print(f"  Phase 6: Evaluating...", end=" ")
    y_pred_norm = model.predict([X_test_norm, X_test_norm], verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    
    price_pred = base_prices_test * np.exp(y_pred)
    price_actual = base_prices_test * np.exp(y_test)
    mape = np.mean(np.abs((price_actual - price_pred) / price_actual)) * 100
    direction_acc = np.sum((y_test * y_pred) > 0) / len(y_test) * 100
    
    print(f"MAPE: {mape:.4f}%, Dir.Acc: {direction_acc:.2f}%")
    
    # 8. Save model and scalers
    print(f"  Phase 7: Saving...", end=" ")
    
    model_filename = f"{symbol}_{timeframe}_model.h5"
    scalers_filename = f"{symbol}_{timeframe}_scalers.pkl"
    
    model_path = OUTPUT_DIR / model_filename
    scalers_path = OUTPUT_DIR / scalers_filename
    
    model.save(str(model_path))
    
    scalers_dict = {
        'X': scaler_X,
        'y': scaler_y,
        'feature_cols': feature_cols,
        'lookback': LOOKBACK
    }
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers_dict, f)
    
    print(f"Done")
    
    # Clean up
    import gc
    gc.collect()
    tf.keras.backend.clear_session()
    
    return True, mape

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

print("\nStarting enhanced multi-crypto training...\n")

results = []
total_models = len(CRYPTOS) * len(TIMEFRAMES)
current_model = 0

for symbol in CRYPTOS:
    for timeframe in TIMEFRAMES:
        current_model += 1
        print(f"[{current_model}/{total_models}] {symbol} {timeframe}")
        
        success, mape = train_crypto_model(symbol, timeframe)
        
        if success:
            results.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'mape': mape,
                'status': 'SUCCESS'
            })
        else:
            results.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'mape': None,
                'status': 'SKIPPED'
            })

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80 + "\n")

df_results = pd.DataFrame(results)

print("Results by Status:")
for status in df_results['status'].unique():
    count = len(df_results[df_results['status'] == status])
    print(f"  {status}: {count}")

print(f"\nSuccessful Models: {len(df_results[df_results['status'] == 'SUCCESS'])}")

if len(df_results[df_results['status'] == 'SUCCESS']) > 0:
    print(f"\nMAPE Statistics (for successful models):")
    mape_data = df_results[df_results['status'] == 'SUCCESS']['mape']
    print(f"  Mean MAPE: {mape_data.mean():.4f}%")
    print(f"  Median MAPE: {mape_data.median():.4f}%")
    print(f"  Best MAPE: {mape_data.min():.4f}%")
    print(f"  Worst MAPE: {mape_data.max():.4f}%")
    print(f"\nBest performing models:")
    top_5 = df_results[df_results['status'] == 'SUCCESS'].nsmallest(5, 'mape')
    for _, row in top_5.iterrows():
        print(f"    {row['symbol']:6} {row['timeframe']:3} - MAPE: {row['mape']:.4f}%")

print(f"\nAll models saved to: {OUTPUT_DIR}")
print(f"Total files created: {len(list(OUTPUT_DIR.glob('*')))}")

print("\n" + "="*80)
print("✓ ENHANCED MULTI-CRYPTO TRAINING COMPLETE!")
print("="*80)
