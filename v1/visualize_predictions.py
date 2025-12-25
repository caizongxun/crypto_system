#!/usr/bin/env python3
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import yfinance as yf

print("="*80)
print("CRYPTO PRICE PREDICTION VISUALIZER")
print("Load models from Hugging Face and visualize predictions")
print("="*80)
print()

print("Installing dependencies...")
os.system('pip install huggingface_hub -q')
from huggingface_hub import hf_hub_download, list_repo_files

print("Setting up...")
from IPython.display import display
import ipywidgets as widgets

# ============================================================================
# CONFIGURATION
# ============================================================================

HF_REPO = "zongowo111/cpb-models"
HF_REPO_TYPE = "dataset"
MODELS_FOLDER = "models_v5"
LOCAL_CACHE = Path("/tmp/crypto_models")
LOCAL_CACHE.mkdir(exist_ok=True)

CRYPTOS = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'BNB': 'BNB-USD',
    'SOL': 'SOL-USD',
    'XRP': 'XRP-USD',
    'ADA': 'ADA-USD',
    'DOGE': 'DOGE-USD',
    'AVAX': 'AVAX-USD',
    'LTC': 'LTC-USD',
    'DOT': 'DOT-USD',
    'UNI': 'UNI-USD',
    'LINK': 'LINK-USD',
    'XLM': 'XLM-USD',
    'ATOM': 'ATOM-USD',
}

TIMEFRAMES = ['1d', '1h']
LOOKBACK = 60

print(f"HF Repository: {HF_REPO}")
print(f"Models folder: {MODELS_FOLDER}")
print(f"Cache directory: {LOCAL_CACHE}")
print()

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

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

# ============================================================================
# DOWNLOAD MODELS FROM HF
# ============================================================================

def download_model(symbol, timeframe):
    model_name = f"{symbol}_{timeframe}_model.h5"
    scalers_name = f"{symbol}_{timeframe}_scalers.pkl"
    
    model_path = LOCAL_CACHE / model_name
    scalers_path = LOCAL_CACHE / scalers_name
    
    try:
        if not model_path.exists():
            print(f"Downloading {model_name}...", end=" ", flush=True)
            hf_hub_download(
                repo_id=HF_REPO,
                filename=f"{MODELS_FOLDER}/{model_name}",
                repo_type=HF_REPO_TYPE,
                cache_dir=str(LOCAL_CACHE),
                force_filename=model_name
            )
            print("Done")
        
        if not scalers_path.exists():
            print(f"Downloading {scalers_name}...", end=" ", flush=True)
            hf_hub_download(
                repo_id=HF_REPO,
                filename=f"{MODELS_FOLDER}/{scalers_name}",
                repo_type=HF_REPO_TYPE,
                cache_dir=str(LOCAL_CACHE),
                force_filename=scalers_name
            )
            print("Done")
        
        model = tf.keras.models.load_model(str(model_path))
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        
        return model, scalers
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None, None

# ============================================================================
# FETCH RECENT DATA
# ============================================================================

def fetch_recent_data(symbol_ticker, interval, days=400):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = yf.download(symbol_ticker, start=start_date.date(), end=end_date.date(),
                        interval=interval, progress=False, prepost=False, threads=False)
        
        if df is None or len(df) == 0:
            return None
        
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index.name = 'timestamp'
        df = df.reset_index()
        
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required):
            return None
        
        df = df[required].copy()
        df = df.dropna()
        df = df[df['volume'] > 0]
        
        return df
    except Exception as e:
        print(f"ERROR fetching data: {e}")
        return None

# ============================================================================
# MAKE PREDICTIONS
# ============================================================================

def predict(model, scalers, df):
    if model is None or scalers is None:
        return None, None, None
    
    try:
        feature_cols = scalers['feature_cols']
        df_feat = engineer_features(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy())
        
        if len(df_feat) < LOOKBACK + 1:
            return None, None, None
        
        X = []
        for i in range(len(df_feat) - LOOKBACK):
            X.append(df_feat[feature_cols].iloc[i:i+LOOKBACK].values)
        
        X = np.array(X)
        scaler_X = scalers['X']
        scaler_y = scalers['y']
        
        X_norm = X.copy()
        for i in range(len(X_norm)):
            X_norm[i] = scaler_X.transform(X[i])
        
        y_pred_norm = model.predict([X_norm, X_norm], verbose=0).flatten()
        y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        
        base_prices = df_feat['close'].iloc[LOOKBACK:-1].values
        predicted_prices = base_prices * np.exp(y_pred[:-1])
        actual_prices = df_feat['close'].iloc[LOOKBACK+1:].values
        timestamps = df_feat['timestamp'].iloc[LOOKBACK+1:].values
        
        return timestamps[:len(predicted_prices)], predicted_prices, actual_prices
    except Exception as e:
        print(f"ERROR making predictions: {e}")
        return None, None, None

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_predictions(symbol, timeframe, timestamps, predicted, actual):
    if timestamps is None or predicted is None:
        print(f"No predictions available for {symbol} {timeframe}")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    timestamps = pd.to_datetime(timestamps)
    
    ax.plot(timestamps, actual, label='Actual Price', linewidth=2, color='#1f77b4', alpha=0.8)
    ax.plot(timestamps, predicted, label='Predicted Price', linewidth=2, color='#ff7f0e', alpha=0.8, linestyle='--')
    
    ax.fill_between(timestamps, actual, predicted, alpha=0.2, color='#d62728')
    
    ax.set_xlabel('Date/Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax.set_title(f'{symbol} Price Prediction ({timeframe}) - Actual vs Predicted', 
                 fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    if timeframe == '1d':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    stats_text = f"MAPE: {mape:.4f}% | RMSE: ${rmse:.2f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.show()

# ============================================================================
# INTERACTIVE UI
# ============================================================================

print("Creating interactive interface...")
print()

symbol_dropdown = widgets.Dropdown(
    options=list(CRYPTOS.keys()),
    value='BTC',
    description='Crypto:'
)

timeframe_dropdown = widgets.Dropdown(
    options=TIMEFRAMES,
    value='1d',
    description='Timeframe:'
)

visualize_button = widgets.Button(description='Visualize', button_style='info')
status_output = widgets.Output()

def on_visualize_clicked(b):
    with status_output:
        status_output.clear_output()
        symbol = symbol_dropdown.value
        timeframe = timeframe_dropdown.value
        ticker = CRYPTOS[symbol]
        
        print(f"Loading model for {symbol} {timeframe}...")
        model, scalers = download_model(symbol, timeframe)
        
        if model is None:
            print(f"ERROR: Could not load model")
            return
        
        print(f"Fetching recent data for {symbol}...")
        days = 3000 if timeframe == '1d' else 400
        df = fetch_recent_data(ticker, timeframe, days=days)
        
        if df is None:
            print(f"ERROR: Could not fetch data")
            return
        
        print(f"Making predictions...")
        timestamps, predicted, actual = predict(model, scalers, df)
        
        if timestamps is None:
            print(f"ERROR: Could not make predictions")
            return
        
        print(f"Plotting results...")
        plot_predictions(symbol, timeframe, timestamps, predicted, actual)

visualize_button.on_click(on_visualize_clicked)

print()
print("="*80)
print("INTERACTIVE CONTROLS")
print("="*80)

ui = widgets.VBox([
    widgets.HBox([symbol_dropdown, timeframe_dropdown]),
    visualize_button,
    status_output
])

display(ui)

print()
print("Select a cryptocurrency and timeframe, then click 'Visualize' to see predictions.")
