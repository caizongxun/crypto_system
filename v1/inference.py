#!/usr/bin/env python3
# ============================================================================
# Crypto System v1 - Inference Module
# ============================================================================
# Usage:
#   from inference import CryptoPricePredictor
#   predictor = CryptoPricePredictor(model_path, scalers_path)
#   price_pred = predictor.predict(df_recent_60_candles)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path
import json
import pickle

class CryptoPricePredictor:
    """
    Crypto price prediction model for next 1-hour price.
    
    Usage:
        predictor = CryptoPricePredictor(
            model_path='/path/to/model.h5',
            scaler_path='/path/to/scaler.pkl'
        )
        pred = predictor.predict(df_60_candles)
        print(f"Next price: ${pred['price']:.2f}")
        print(f"Confidence: {pred['confidence']:.2f}")
    """
    
    def __init__(self, model_path, scaler_path=None):
        """
        Initialize predictor with trained model and scalers.
        
        Args:
            model_path: Path to trained model (e.g., 'crypto_v1_final_model.h5')
            scaler_path: Path to pickled scalers dict (optional)
        """
        self.model = tf.keras.models.load_model(model_path)
        self.model_path = Path(model_path)
        
        # Feature column names (must match training)
        self.feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'log_return', 'volatility', 'momentum', 'price_range',
            'rsi', 'macd', 'macd_signal', 'volume_ratio', 'atr'
        ]
        
        # Load scalers if provided
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                scalers_dict = pickle.load(f)
                self.scaler_features = scalers_dict['features']
                self.scaler_target = scalers_dict['target']
        else:
            print("Warning: No scalers provided. Using default scaling.")
            self.scaler_features = StandardScaler()
            self.scaler_target = MinMaxScaler()
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Features: {len(self.feature_cols)} dimensions")
    
    def compute_features(self, df):
        """
        Compute technical features from OHLCV data.
        
        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
            
        Returns:
            DataFrame with computed features
        """
        if len(df) < 60:
            raise ValueError(f"Need at least 60 candles, got {len(df)}")
        
        df = df.copy()
        
        # Basic features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(14).std()
        df['momentum'] = df['close'] - df['close'].shift(10)
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
        
        # Fill NaN
        df = df.ffill().bfill()
        
        return df
    
    def predict(self, df, return_confidence=True):
        """
        Predict next price for given recent candles.
        
        Args:
            df: DataFrame with [timestamp, open, high, low, close, volume]
            return_confidence: If True, compute confidence metric
            
        Returns:
            dict with 'price' and optional 'confidence'
        """
        # Compute features
        df_features = self.compute_features(df)
        
        # Get last 60 candles
        df_recent = df_features[self.feature_cols].iloc[-60:]
        
        if len(df_recent) < 60:
            raise ValueError(f"Need exactly 60 candles, got {len(df_recent)}")
        
        # Normalize features
        X_normalized = self.scaler_features.transform(df_recent)
        X_normalized = np.expand_dims(X_normalized, axis=0)  # (1, 60, 14)
        
        # Predict
        y_pred_normalized = self.model.predict([X_normalized, X_normalized], verbose=0)[0, 0]
        
        # Denormalize
        y_pred = self.scaler_target.inverse_transform([[y_pred_normalized]])[0, 0]
        
        result = {'price': float(y_pred)}
        
        # Confidence metric (based on recent volatility)
        if return_confidence:
            recent_volatility = df_features['volatility'].iloc[-20:].mean()
            # Higher volatility = lower confidence
            confidence = max(0.5, 1.0 - recent_volatility)
            result['confidence'] = float(confidence)
        
        return result
    
    def predict_batch(self, list_of_dfs, return_confidence=True):
        """
        Predict for multiple symbols in batch.
        
        Args:
            list_of_dfs: List of DataFrames (one per symbol)
            return_confidence: If True, compute confidence metrics
            
        Returns:
            List of dicts with predictions
        """
        predictions = []
        for df in list_of_dfs:
            try:
                pred = self.predict(df, return_confidence=return_confidence)
                predictions.append(pred)
            except Exception as e:
                print(f"Error predicting: {e}")
                predictions.append({'price': None, 'confidence': None})
        return predictions


def demo():
    """
    Demo: Generate synthetic data and make predictions.
    """
    from datetime import datetime, timedelta
    
    print("\n" + "="*80)
    print("CRYPTO SYSTEM v1 - INFERENCE DEMO")
    print("="*80 + "\n")
    
    # Create synthetic test data
    print("Generating synthetic test data...")
    np.random.seed(42)
    n = 100
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(n)][::-1]
    returns = np.random.normal(0.0005, 0.015, n)
    prices = 90000 * np.exp(np.cumsum(returns))
    
    df_test = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices.copy(),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n))),
        'close': prices * (1 + np.random.normal(0, 0.005, n)),
        'volume': np.random.uniform(100000, 500000, n)
    })
    
    print(f"Test data shape: {df_test.shape}")
    print(f"Latest price: ${df_test['close'].iloc[-1]:.2f}\n")
    
    # Load model
    model_path = Path("/content/all_models/v1/crypto_v1_final_model.h5")
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first.")
        return
    
    # Note: For full functionality, you need to save scalers during training
    # This demo uses default scaling
    print("Loading model...")
    predictor = CryptoPricePredictor(str(model_path))
    
    # Make prediction
    print("\nMaking prediction...")
    try:
        prediction = predictor.predict(df_test, return_confidence=True)
        
        print(f"\nPrediction Result:")
        print(f"  Current Price:  ${df_test['close'].iloc[-1]:.2f}")
        print(f"  Next Hour Pred: ${prediction['price']:.2f}")
        print(f"  Confidence:     {prediction['confidence']:.2%}")
        
        price_change = prediction['price'] - df_test['close'].iloc[-1]
        price_change_pct = (price_change / df_test['close'].iloc[-1]) * 100
        print(f"\n  Expected Change: ${price_change:+.2f} ({price_change_pct:+.2f}%)")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("\nNote: For full functionality, train the model first with:")
        print("  python v1/train_final_optimized.py")


if __name__ == "__main__":
    demo()
