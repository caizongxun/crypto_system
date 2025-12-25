#!/usr/bin/env python3
# ============================================================================
# Crypto System v2 - Prediction Visualization
# ============================================================================
# Plots actual vs predicted prices with confidence bands

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler

try:
    import ccxt
    HAS_CCXT = True
except:
    HAS_CCXT = False

print("Loading trained model and scalers...")

# Paths
CACHE_DIR = Path("/content/all_models/v2")
model_path = CACHE_DIR / "crypto_v2_final_model.h5"
scaler_path = CACHE_DIR / "scalers_v2.pkl"

# Load model
model = tf.keras.models.load_model(str(model_path))

# Load scalers
with open(scaler_path, 'rb') as f:
    scalers_dict = pickle.load(f)
    scaler_X = scalers_dict['X']
    scaler_y = scalers_dict['y']
    feature_cols = scalers_dict['feature_cols']
    lookback = scalers_dict['lookback']

print(f"✓ Model loaded: {model_path}")
print(f"✓ Scalers loaded: {scaler_path}")
print()

# ============================================================================
# FETCH FRESH BINANCE DATA FOR VISUALIZATION
# ============================================================================
print("Fetching fresh BTC data for visualization...")

def fetch_binance_data(symbol, timeframe='1h', limit=300):
    if not HAS_CCXT:
        print(f"  CCXT not available, using synthetic data")
        return None
    
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"  ✓ Got {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        return df
    except Exception as e:
        print(f"  Error: {e}")
        return None

df_btc = fetch_binance_data('BTC/USDT', limit=300)

if df_btc is None:
    print("Cannot fetch data. Please check CCXT installation.")
    print("You can still use the model for inference with your own data.")
    exit(1)

# ============================================================================
# FEATURE ENGINEERING (SAME AS TRAINING)
# ============================================================================
print("\nComputing features...")

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

df_btc = engineer_features(df_btc)

# ============================================================================
# CREATE SEQUENCES AND PREDICT
# ============================================================================
print("Creating sequences and making predictions...")

X_seq = []
base_prices = []
dates = []

for i in range(len(df_btc) - lookback - 1):
    X_seq.append(df_btc[feature_cols].iloc[i:i+lookback].values)
    base_prices.append(df_btc['close'].iloc[i+lookback-1])
    dates.append(df_btc['timestamp'].iloc[i+lookback])

X_seq = np.array(X_seq)
base_prices = np.array(base_prices)
dates = np.array(dates)

# Normalize
X_norm = X_seq.copy()
for i in range(len(X_norm)):
    X_norm[i] = scaler_X.transform(X_seq[i])

# Predict
y_pred_norm = model.predict(np.array([X_norm, X_norm]), verbose=0).flatten()
y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

# Convert to prices
price_pred = base_prices * np.exp(y_pred)
price_actual = base_prices * np.exp(df_btc['log_return'].iloc[lookback+1:lookback+1+len(y_pred)].values)

print(f"✓ Made {len(y_pred)} predictions")
print()

# ============================================================================
# CALCULATE METRICS
# ============================================================================
mae = np.mean(np.abs(price_actual - price_pred))
rmse = np.sqrt(np.mean((price_actual - price_pred)**2))
mape = np.mean(np.abs((price_actual - price_pred) / price_actual)) * 100

print("Prediction Metrics:")
print(f"  MAE:  ${mae:.2f}")
print(f"  RMSE: ${rmse:.2f}")
print(f"  MAPE: {mape:.4f}%")
print()

# ============================================================================
# PLOTTING
# ============================================================================
print("Creating visualization...")

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Plot 1: Full Predictions
ax = axes[0]
ax.plot(dates, price_actual, 'b-', linewidth=2, label='Actual Price', alpha=0.8)
ax.plot(dates, price_pred, 'orange', linewidth=2, label='Predicted Price', alpha=0.8)
ax.fill_between(dates, price_actual, price_pred, alpha=0.2, color='gray')
ax.set_title('BTC Price: Actual vs Predicted (Full Series)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=12)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 2: Last 100 candles (zoomed)
ax = axes[1]
if len(dates) > 100:
    idx_start = -100
else:
    idx_start = 0

ax.plot(dates[idx_start:], price_actual[idx_start:], 'b-', linewidth=2.5, label='Actual Price', marker='o', markersize=4, alpha=0.8)
ax.plot(dates[idx_start:], price_pred[idx_start:], 'orange', linewidth=2.5, label='Predicted Price', marker='s', markersize=4, alpha=0.8)
ax.fill_between(dates[idx_start:], price_actual[idx_start:], price_pred[idx_start:], alpha=0.2, color='gray')
ax.set_title(f'BTC Price: Last {-idx_start} Hours (Detailed View)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=12)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 3: Prediction Error Over Time
ax = axes[2]
error = price_actual - price_pred
error_pct = (error / price_actual) * 100

ax.bar(dates, error_pct, color=['green' if e > 0 else 'red' for e in error_pct], alpha=0.7, width=0.03)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.set_title('Prediction Error (Actual - Predicted) in Percentage', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Error (%)', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Add metrics text
error_stats = f"Mean Error: {np.mean(error_pct):.4f}% | Std Dev: {np.std(error_pct):.4f}% | Max: {np.max(np.abs(error_pct)):.4f}%"
ax.text(0.5, 0.95, error_stats, transform=ax.transAxes, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

plt.tight_layout()

# Save figure
output_path = CACHE_DIR / 'prediction_visualization.png'
plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: {output_path}")

plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("PREDICTION SUMMARY")
print("="*80)
print(f"\nData Range: {dates[0]} to {dates[-1]}")
print(f"Predictions Made: {len(y_pred)}")
print(f"\nPrice Range:")
print(f"  Actual Min: ${price_actual.min():,.2f}")
print(f"  Actual Max: ${price_actual.max():,.2f}")
print(f"  Predicted Min: ${price_pred.min():,.2f}")
print(f"  Predicted Max: ${price_pred.max():,.2f}")
print(f"\nError Analysis:")
print(f"  MAE: ${mae:.2f}")
print(f"  RMSE: ${rmse:.2f}")
print(f"  MAPE: {mape:.4f}%")
print(f"  Mean Error: {np.mean(error_pct):.4f}%")
print(f"  Std Dev: {np.std(error_pct):.4f}%")
print(f"  Max Absolute Error: {np.max(np.abs(error_pct)):.4f}%")
print(f"\nDirection Accuracy:")
direction_correct = np.sum((error > 0) == (np.diff(price_actual) > 0))
direction_accuracy = (direction_correct / (len(error) - 1)) * 100
print(f"  {direction_accuracy:.2f}%")
print()
