#!/usr/bin/env python3
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
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Conv1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

print("Installing dependencies...")
os.system('pip install yfinance -q')
import yfinance as yf

print("="*80)
print("CRYPTO SYSTEM v2 - MULTI-CRYPTO TRAINING (YFINANCE)")
print("Cryptocurrencies x 2 Timeframes (1d + 1h) x 7000+ Candles")
print("="*80)
print()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU: {len(gpus)} device(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected")

OUTPUT_DIR = Path("/content/all_models/multi_crypto")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")
print()

CRYPTOS = [
    ('BTC', 'BTC-USD'),
    ('ETH', 'ETH-USD'),
    ('BNB', 'BNB-USD'),
    ('SOL', 'SOL-USD'),
    ('XRP', 'XRP-USD'),
    ('ADA', 'ADA-USD'),
    ('DOGE', 'DOGE-USD'),
    ('AVAX', 'AVAX-USD'),
    ('LTC', 'LTC-USD'),
    ('DOT', 'DOT-USD'),
    ('UNI', 'UNI-USD'),
    ('LINK', 'LINK-USD'),
    ('XLM', 'XLM-USD'),
    ('ATOM', 'ATOM-USD'),
]

TIMEFRAMES = {'1d': '1d', '1h': '1h'}
LOOKBACK = 60
EPOCHS = 40
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 8

print(f"Cryptocurrencies: {len(CRYPTOS)}")
print(f"Timeframes: {list(TIMEFRAMES.keys())}")
print(f"Total models to train: {len(CRYPTOS) * len(TIMEFRAMES)}")
print(f"Lookback period: {LOOKBACK} candles")
print()

def engineer_features(df):
    df = df.copy()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['simple_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['volatility'] = df['log_return'].rolling(14).std()
    df['volatility_20'] = df['log_return'].rolling(20).std()
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_20'] = df['close'] - df['close'].shift(20)
    df['price_range'] = (df['high'] - df['low']) / df['close']
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
    df['funding_rate'] = df['momentum_10'].rolling(20).mean() / (df['close'].rolling(20).std() + 1e-8) * 0.00001
    df['open_interest_change'] = (df['volume_ratio'] - 1) * (df['volatility'] / 0.01) * 0.1
    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], 0)
    return df

def fetch_crypto_data(symbol, yfinance_ticker, interval):
    try:
        print(f"    Fetching {symbol} {interval}...", end=" ", flush=True)
        days_back = 3000 if interval == '1d' else 400
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        df = yf.download(yfinance_ticker, start=start_date.date(), end=end_date.date(), 
                        interval=interval, progress=False, prepost=False, threads=False)
        if df is None or len(df) == 0:
            print("NO DATA")
            return None
        
        df = df.copy()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.columns = [c.lower() for c in df.columns]
        
        df.index.name = 'timestamp'
        df = df.reset_index()
        
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        
        if missing:
            print(f"MISSING: {missing}")
            return None
        
        df = df[required].copy()
        df = df.dropna()
        df = df[df['volume'] > 0]
        
        print(f"{len(df)} candles")
        return df
    except Exception as e:
        error_msg = str(e)[:60]
        print(f"ERROR: {error_msg}")
        return None

def build_lstm_model(n_features, lookback):
    input_main = Input(shape=(lookback, n_features), name='input_main')
    input_aux = Input(shape=(lookback, n_features), name='input_aux')
    x1 = LSTM(64, activation='relu', return_sequences=True)(input_main)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.2)(x1)
    x1 = LSTM(32, activation='relu', return_sequences=False)(x1)
    x1 = Dropout(0.2)(x1)
    x2 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_aux)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.2)(x2)
    x2 = LSTM(32, activation='relu', return_sequences=False)(x2)
    x2 = Dropout(0.2)(x2)
    combined = Concatenate()([x1, x2])
    z = Dense(32, activation='relu')(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.2)(z)
    z = Dense(16, activation='relu')(z)
    z = Dropout(0.1)(z)
    output = Dense(1, name='return')(z)
    model = Model(inputs=[input_main, input_aux], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0003, clipvalue=1.0), loss='huber', metrics=['mae'])
    return model

def train_crypto_model(symbol, yfinance_ticker, interval):
    print(f"\n{'='*80}")
    print(f"Training: {symbol} {interval}")
    print(f"{'='*80}")
    print(f"  Phase 1: Fetching data...")
    df = fetch_crypto_data(symbol, yfinance_ticker, interval)
    if df is None or len(df) < LOOKBACK + 100:
        print(f"  SKIP: Not enough data ({len(df) if df is not None else 0} candles)")
        return False, None
    print(f"  Phase 2: Computing features...", end=" ", flush=True)
    df = engineer_features(df)
    print("Done")
    print(f"  Phase 3: Creating sequences...", end=" ", flush=True)
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'log_return', 'volatility', 
                   'volatility_20', 'momentum_10', 'momentum_20', 'price_range', 'rsi', 
                   'macd', 'macd_signal', 'volume_ratio', 'atr', 'funding_rate', 'open_interest_change']
    X, y, base_prices = [], [], []
    for i in range(len(df) - LOOKBACK - 1):
        X.append(df[feature_cols].iloc[i:i+LOOKBACK].values)
        y.append(df['log_return'].iloc[i+LOOKBACK])
        base_prices.append(df['close'].iloc[i+LOOKBACK-1])
    X = np.array(X)
    y = np.array(y)
    base_prices = np.array(base_prices)
    print(f"{len(X)} sequences")
    n_total = len(X)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    base_prices_test = base_prices[n_train+n_val:]
    print(f"  Phase 4: Normalizing...", end=" ", flush=True)
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
    print("Done")
    print(f"  Phase 5: Training model...", end=" ", flush=True)
    model = build_lstm_model(X_train_norm.shape[2], LOOKBACK)
    start_time = time.time()
    history = model.fit([X_train_norm, X_train_norm], y_train_norm,
                       validation_data=([X_val_norm, X_val_norm], y_val_norm),
                       epochs=EPOCHS, batch_size=BATCH_SIZE,
                       callbacks=[EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, 
                                               restore_best_weights=True, verbose=0),
                                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 
                                                  min_lr=1e-6, verbose=0)], verbose=0)
    train_time = time.time() - start_time
    print(f"{train_time:.0f}s")
    print(f"  Phase 6: Evaluating...", end=" ", flush=True)
    y_pred_norm = model.predict([X_test_norm, X_test_norm], verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    price_pred = base_prices_test * np.exp(y_pred)
    price_actual = base_prices_test * np.exp(y_test)
    mape = np.mean(np.abs((price_actual - price_pred) / price_actual)) * 100
    direction_acc = np.sum((y_test * y_pred) > 0) / len(y_test) * 100
    print(f"MAPE: {mape:.4f}%, Dir.Acc: {direction_acc:.2f}%")
    print(f"  Phase 7: Saving...", end=" ", flush=True)
    model_filename = f"{symbol}_{interval}_model.h5"
    scalers_filename = f"{symbol}_{interval}_scalers.pkl"
    model_path = OUTPUT_DIR / model_filename
    scalers_path = OUTPUT_DIR / scalers_filename
    model.save(str(model_path))
    scalers_dict = {'X': scaler_X, 'y': scaler_y, 'feature_cols': feature_cols, 'lookback': LOOKBACK}
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers_dict, f)
    print("Done")
    import gc
    gc.collect()
    tf.keras.backend.clear_session()
    return True, mape

print("Starting training...")
print()

results = []
total_models = len(CRYPTOS) * len(TIMEFRAMES)
current_model = 0

for symbol, yfinance_ticker in CRYPTOS:
    for interval_name, interval_code in TIMEFRAMES.items():
        current_model += 1
        print(f"[{current_model}/{total_models}] {symbol} {interval_name}")
        success, mape = train_crypto_model(symbol, yfinance_ticker, interval_code)
        if success:
            results.append({'symbol': symbol, 'timeframe': interval_name, 'mape': mape, 'status': 'SUCCESS'})
        else:
            results.append({'symbol': symbol, 'timeframe': interval_name, 'mape': None, 'status': 'SKIPPED'})

print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)
print()

df_results = pd.DataFrame(results)
print("Results by Status:")
for status in df_results['status'].unique():
    count = len(df_results[df_results['status'] == status])
    print(f"  {status}: {count}")

print(f"\nSuccessful Models: {len(df_results[df_results['status'] == 'SUCCESS'])}")

if len(df_results[df_results['status'] == 'SUCCESS']) > 0:
    print(f"\nMAPE Statistics:")
    mape_data = df_results[df_results['status'] == 'SUCCESS']['mape']
    print(f"  Mean MAPE: {mape_data.mean():.4f}%")
    print(f"  Median MAPE: {mape_data.median():.4f}%")
    print(f"  Best MAPE: {mape_data.min():.4f}%")
    print(f"  Worst MAPE: {mape_data.max():.4f}%")
    print(f"\nTop 5 models:")
    top_5 = df_results[df_results['status'] == 'SUCCESS'].nsmallest(5, 'mape')
    for _, row in top_5.iterrows():
        print(f"    {row['symbol']:6} {row['timeframe']:3} - MAPE: {row['mape']:.4f}%")

print(f"\nAll models saved to: {OUTPUT_DIR}")
print(f"Total files created: {len(list(OUTPUT_DIR.glob('*')))}")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
