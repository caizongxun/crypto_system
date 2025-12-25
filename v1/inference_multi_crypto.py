#!/usr/bin/env python3
# ============================================================================
# Crypto System v2 - Multi-Crypto Inference Module
# ============================================================================
# Load and use trained models for any cryptocurrency and timeframe

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import pickle
from datetime import datetime

class MultiCryptoPredictor:
    """
    Universal predictor for all trained crypto models.
    
    Usage:
        predictor = MultiCryptoPredictor(
            models_dir='/content/all_models/multi_crypto'
        )
        
        # List all available models
        print(predictor.available_models())
        
        # Make prediction
        pred = predictor.predict(
            symbol='BTC',
            timeframe='1h',
            df_recent=df_60_candles,
            last_price=100000
        )
    """
    
    def __init__(self, models_dir):
        """
        Initialize predictor with models directory.
        
        Args:
            models_dir: Path to directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.loaded_models = {}  # Cache loaded models
        self.loaded_scalers = {}  # Cache loaded scalers
        
        # Discover available models
        self.available_pairs = self._discover_models()
        
        print(f"✓ Multi-Crypto Predictor initialized")
        print(f"  Models directory: {self.models_dir}")
        print(f"  Available models: {len(self.available_pairs)}")
    
    def _discover_models(self):
        """
        Discover all available models in the directory.
        """
        pairs = set()
        
        for model_file in self.models_dir.glob('*_model.h5'):
            # Extract symbol and timeframe from filename
            # Format: {SYMBOL}_{TIMEFRAME}_model.h5
            parts = model_file.stem.split('_')
            if len(parts) >= 2:
                symbol = parts[0]
                timeframe = '_'.join(parts[1:-1])  # Handle '15m' -> '15_m'
                pairs.add((symbol, timeframe))
        
        return sorted(list(pairs))
    
    def available_models(self):
        """
        List all available cryptocurrency-timeframe pairs.
        """
        print("\nAvailable Models:")
        print("-" * 40)
        
        by_symbol = {}
        for symbol, timeframe in self.available_pairs:
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(timeframe)
        
        for symbol in sorted(by_symbol.keys()):
            timeframes = ', '.join(sorted(by_symbol[symbol]))
            print(f"  {symbol:6} - {timeframes}")
        
        print("-" * 40)
        print(f"Total: {len(self.available_pairs)} models\n")
        
        return self.available_pairs
    
    def _load_model(self, symbol, timeframe):
        """
        Load model from disk (with caching).
        """
        key = (symbol, timeframe)
        
        if key in self.loaded_models:
            return self.loaded_models[key]
        
        model_path = self.models_dir / f"{symbol}_{timeframe}_model.h5"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = tf.keras.models.load_model(str(model_path))
        self.loaded_models[key] = model
        return model
    
    def _load_scalers(self, symbol, timeframe):
        """
        Load scalers from disk (with caching).
        """
        key = (symbol, timeframe)
        
        if key in self.loaded_scalers:
            return self.loaded_scalers[key]
        
        scalers_path = self.models_dir / f"{symbol}_{timeframe}_scalers.pkl"
        
        if not scalers_path.exists():
            raise FileNotFoundError(f"Scalers not found: {scalers_path}")
        
        with open(scalers_path, 'rb') as f:
            scalers_dict = pickle.load(f)
        
        self.loaded_scalers[key] = scalers_dict
        return scalers_dict
    
    def _engineer_features(self, df):
        """
        Compute technical features (same as training).
        """
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
    
    def predict(self, symbol, timeframe, df, last_price):
        """
        Make prediction for given symbol and timeframe.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            timeframe: Timeframe (e.g., '1h', '15m')
            df: DataFrame with [timestamp, open, high, low, close, volume]
            last_price: Current price
            
        Returns:
            dict with prediction results
        """
        # Load model and scalers
        model = self._load_model(symbol, timeframe)
        scalers_dict = self._load_scalers(symbol, timeframe)
        
        scaler_X = scalers_dict['X']
        scaler_y = scalers_dict['y']
        feature_cols = scalers_dict['feature_cols']
        lookback = scalers_dict['lookback']
        
        # Compute features
        df_features = self._engineer_features(df)
        
        # Get recent candles
        df_recent = df_features[feature_cols].iloc[-lookback:]
        
        if len(df_recent) < lookback:
            raise ValueError(f"Need {lookback} candles, got {len(df_recent)}")
        
        # Normalize
        X_norm = df_recent.values
        X_norm = scaler_X.transform(X_norm)
        X_norm = np.expand_dims(X_norm, axis=0)  # Shape: (1, lookback, n_features)
        
        # Predict
        return_norm = model.predict([X_norm, X_norm], verbose=0)[0, 0]
        return_pred = scaler_y.inverse_transform([[return_norm]])[0, 0]
        
        # Convert to price
        price_pred = last_price * np.exp(return_pred)
        
        # Confidence based on recent volatility
        recent_vol = df_features['volatility'].iloc[-20:].mean()
        confidence = max(0.5, 1.0 - recent_vol * 5)
        
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now(),
            'return': float(return_pred),
            'return_percent': float(return_pred * 100),
            'price': float(price_pred),
            'confidence': float(confidence),
            'last_price': float(last_price),
            'price_change': float(price_pred - last_price),
            'price_change_percent': float((price_pred - last_price) / last_price * 100)
        }
        
        return result
    
    def predict_batch(self, predictions_specs):
        """
        Make predictions for multiple symbols and timeframes.
        
        Args:
            predictions_specs: List of dicts with keys:
                - symbol: 'BTC', 'ETH', etc.
                - timeframe: '1h', '15m', etc.
                - df: DataFrame
                - last_price: Current price
        
        Returns:
            List of prediction results
        """
        results = []
        
        for spec in predictions_specs:
            try:
                pred = self.predict(
                    symbol=spec['symbol'],
                    timeframe=spec['timeframe'],
                    df=spec['df'],
                    last_price=spec['last_price']
                )
                results.append(pred)
            except Exception as e:
                print(f"Error predicting {spec['symbol']} {spec['timeframe']}: {e}")
                results.append(None)
        
        return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MULTI-CRYPTO PREDICTOR - DEMO")
    print("="*80 + "\n")
    
    # Initialize
    predictor = MultiCryptoPredictor(
        models_dir='/content/all_models/multi_crypto'
    )
    
    # List available models
    predictor.available_models()
    
    print("To use this predictor:")
    print("""    
    predictor = MultiCryptoPredictor('/content/all_models/multi_crypto')
    
    # Get your data from Binance
    df_btc = get_last_60_candles('BTC', '1h')
    
    # Make prediction
    pred = predictor.predict(
        symbol='BTC',
        timeframe='1h',
        df=df_btc,
        last_price=100000
    )
    
    print(f"Next price: ${pred['price']:.2f}")
    print(f"Expected move: {pred['return_percent']:+.4f}%")
    print(f"Confidence: {pred['confidence']:.1%}")
    """)
