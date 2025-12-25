#!/usr/bin/env python3
# ============================================================================
# Crypto System v2 - Inference Module
# ============================================================================
# Predicts LOG-RETURN for next period, then converts to price.
# Much more stable than direct price prediction.

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import pickle

class CryptoPricePredictorV2:
    """
    V2 Predictor: predicts LOG-RETURN, then converts to price.
    
    Usage:
        predictor = CryptoPricePredictorV2(
            model_path='/path/to/crypto_v2_final_model.h5',
            scaler_path='/path/to/scalers_v2.pkl'
        )
        pred = predictor.predict(df_recent_60_candles, last_price=100000)
        print(f"Next price: ${pred['price']:.2f}")
        print(f"Expected return: {pred['return_percent']:.4f}%")
    """
    
    def __init__(self, model_path, scaler_path):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model (e.g., 'crypto_v2_final_model.h5')
            scaler_path: Path to pickled scalers (e.g., 'scalers_v2.pkl')
        """
        self.model = tf.keras.models.load_model(model_path)
        self.model_path = Path(model_path)
        
        # Load scalers
        with open(scaler_path, 'rb') as f:
            scalers_dict = pickle.load(f)
            self.scaler_X = scalers_dict['X']
            self.scaler_y = scalers_dict['y']
            self.feature_cols = scalers_dict['feature_cols']
            self.lookback = scalers_dict['lookback']
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Scalers loaded: {scaler_path}")
        print(f"✓ Features: {len(self.feature_cols)} dimensions")
        print(f"✓ Lookback: {self.lookback} periods")
    
    def compute_features(self, df):
        """
        Compute all technical features (same as training).
        
        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
            
        Returns:
            DataFrame with computed features
        """
        if len(df) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} candles, got {len(df)}")
        
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
        
        # Funding rate (in real usage, fetch from Binance API)
        df['funding_rate'] = df['momentum_10'].rolling(20).mean() / (df['close'].rolling(20).std() + 1e-8) * 0.00001
        
        # Open interest change
        df['open_interest_change'] = (df['volume_ratio'] - 1) * (df['volatility'] / 0.01) * 0.1
        
        # Fill NaN
        df = df.ffill().bfill()
        
        return df
    
    def predict(self, df, last_price):
        """
        Predict next period's log-return and convert to price.
        
        Args:
            df: DataFrame with [timestamp, open, high, low, close, volume]
            last_price: Current price (for return conversion)
            
        Returns:
            dict with:
              - 'return': predicted log-return
              - 'return_percent': predicted return as percentage
              - 'price': predicted next price
              - 'confidence': confidence metric (0-1)
        """
        # Compute features
        df_features = self.compute_features(df)
        
        # Get last lookback candles
        df_recent = df_features[self.feature_cols].iloc[-self.lookback:]
        
        if len(df_recent) < self.lookback:
            raise ValueError(f"Need exactly {self.lookback} candles, got {len(df_recent)}")
        
        # Normalize
        X_norm = df_recent.values  # Shape: (lookback, n_features)
        X_norm = self.scaler_X.transform(X_norm)  # Apply scaler
        X_norm = np.expand_dims(X_norm, axis=0)  # Shape: (1, lookback, n_features)
        
        # Predict (normalized)
        return_norm = self.model.predict([X_norm, X_norm], verbose=0)[0, 0]
        
        # Denormalize return
        return_pred = self.scaler_y.inverse_transform([[return_norm]])[0, 0]
        
        # Convert to price
        price_pred = last_price * np.exp(return_pred)
        
        # Confidence: based on recent volatility (lower volatility = higher confidence)
        recent_vol = df_features['volatility'].iloc[-20:].mean()
        confidence = max(0.5, 1.0 - recent_vol * 5)  # Clip between 0.5 and 1.0
        
        result = {
            'return': float(return_pred),
            'return_percent': float(return_pred * 100),
            'price': float(price_pred),
            'confidence': float(confidence),
            'last_price': float(last_price),
            'price_change': float(price_pred - last_price)
        }
        
        return result
    
    def predict_batch(self, list_of_dfs, list_of_prices):
        """
        Predict for multiple symbols.
        
        Args:
            list_of_dfs: List of DataFrames (one per symbol)
            list_of_prices: List of current prices (one per symbol)
            
        Returns:
            List of prediction dicts
        """
        predictions = []
        for df, price in zip(list_of_dfs, list_of_prices):
            try:
                pred = self.predict(df, price)
                predictions.append(pred)
            except Exception as e:
                print(f"Error: {e}")
                predictions.append(None)
        return predictions


def demo():
    """
    Demo with synthetic data.
    """
    from datetime import datetime, timedelta
    
    print("\n" + "="*80)
    print("CRYPTO SYSTEM v2 - INFERENCE DEMO")
    print("="*80 + "\n")
    
    # Generate synthetic test data
    print("Generating test data...")
    np.random.seed(42)
    n = 100
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(n)][::-1]
    returns = np.random.normal(0.0005, 0.015, n)
    prices = 50000 * np.exp(np.cumsum(returns))
    
    df_test = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices.copy(),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n))),
        'close': prices * (1 + np.random.normal(0, 0.005, n)),
        'volume': np.random.uniform(100000, 500000, n)
    })
    
    print(f"Data shape: {df_test.shape}")
    print(f"Current price: ${df_test['close'].iloc[-1]:.2f}\n")
    
    # Load model
    model_path = Path("/content/all_models/v2/crypto_v2_final_model.h5")
    scaler_path = Path("/content/all_models/v2/scalers_v2.pkl")
    
    if not model_path.exists():
        print(f"Model not found. Please train first with:")
        print(f"  python v1/train_v2_production.py")
        return
    
    # Initialize predictor
    print("Loading model and scalers...")
    predictor = CryptoPricePredictorV2(str(model_path), str(scaler_path))
    
    # Make prediction
    print("\nMaking prediction...\n")
    try:
        prediction = predictor.predict(df_test, last_price=df_test['close'].iloc[-1])
        
        print("PREDICTION RESULTS:")
        print(f"  Current Price:     ${prediction['last_price']:.2f}")
        print(f"  Predicted Price:   ${prediction['price']:.2f}")
        print(f"  Expected Change:   ${prediction['price_change']:+.2f} ({prediction['return_percent']:+.4f}%)")
        print(f"  Confidence:        {prediction['confidence']:.2%}")
        print()
        
        # Interpretation
        if prediction['return_percent'] > 0.5:
            signal = "↑ BUY (bullish)"
        elif prediction['return_percent'] < -0.5:
            signal = "↓ SELL (bearish)"
        else:
            signal = "→ HOLD (neutral)"
        
        print(f"  Signal:            {signal}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        print(f"\nMake sure you've trained the model first:")
        print(f"  python v1/train_v2_production.py")


if __name__ == "__main__":
    demo()
