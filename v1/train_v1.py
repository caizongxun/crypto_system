# ============================================================================
# Crypto System v1 - Training Script
# ============================================================================
# Main entry point for training the dual-stream LSTM model on Colab GPU
# Usage in Colab:
#   import requests, time
#   url = 'https://raw.githubusercontent.com/caizongxun/crypto_system/main/v1/train_v1.py?t=' + str(int(time.time()))
#   exec(requests.get(url).text)

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Concatenate, Conv1D, Attention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# ============================================================================
# SETUP LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CHECK GPU AND SETUP ENVIRONMENT
# ============================================================================
logger.info("="*80)
logger.info("CRYPTO SYSTEM v1 - TRAINING INITIALIZATION")
logger.info("="*80)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logger.info(f"GPU Available: {len(gpus)} device(s)")
    for gpu in gpus:
        logger.info(f"  {gpu}")
    # Set memory growth to avoid OOM
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    logger.warning("No GPU found. Training will be slow.")

# Create cache directories
CACHE_DIR = Path("/content/all_models/v1")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Cache directory: {CACHE_DIR}")

# ============================================================================
# IMPORT PROJECT MODULES
# ============================================================================
try:
    # For local execution
    from config import *
    from data_fetcher import CryptodataFetcher
    from feature_engineering import FeatureEngineer, process_single_symbol
except ImportError:
    # For remote execution - fetch from GitHub
    logger.info("Importing modules from GitHub...")
    import requests
    
    base_url = "https://raw.githubusercontent.com/caizongxun/crypto_system/main/v1"
    
    config_url = f"{base_url}/config.py?t={int(time.time())}"
    exec(requests.get(config_url).text, globals())
    
    data_fetcher_url = f"{base_url}/data_fetcher.py?t={int(time.time())}"
    exec(requests.get(data_fetcher_url).text, globals())
    
    feature_eng_url = f"{base_url}/feature_engineering.py?t={int(time.time())}"
    exec(requests.get(feature_eng_url).text, globals())

# ============================================================================
# DATA PREPARATION
# ============================================================================
class DataPreparer:
    """
    Prepares data for training.
    """
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.fetcher = CryptodataFetcher(str(cache_dir))
    
    def fetch_and_process_data(self, symbols, timeframes):
        """
        Fetch data from Binance and process features.
        
        Args:
            symbols (list): Trading pairs
            timeframes (dict): {'trend': '1h', 'volatility': '15m'}
            
        Returns:
            dict: Processed data by symbol
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: DATA FETCHING AND PROCESSING")
        logger.info("="*80)
        
        # Fetch data
        logger.info(f"Fetching data for {len(symbols)} symbols...")
        all_data_1h = self.fetcher.fetch_multiple_symbols(
            symbols, timeframes["trend"], limit=1000
        )
        all_data_15m = self.fetcher.fetch_multiple_symbols(
            symbols, timeframes["volatility"], limit=4000  # 4x more for 15m
        )
        
        # Process features
        logger.info(f"Processing features...")
        processed_data = {}
        
        for symbol in all_data_1h.keys():
            if symbol not in all_data_15m:
                logger.warning(f"Skipping {symbol}: missing 15m data")
                continue
            
            df_1h = all_data_1h[symbol]
            df_15m = all_data_15m[symbol]
            
            # Process individual dataframes
            engineer = FeatureEngineer(globals())  # Use global config
            df_1h = process_single_symbol(df_1h, symbol, globals())
            df_15m = process_single_symbol(df_15m, symbol, globals())
            
            # Align timeframes
            df_1h_aligned, df_15m_aligned = engineer.merge_timeframes(
                symbol, df_1h, df_15m
            )
            
            processed_data[symbol] = {
                '1h': df_1h_aligned,
                '15m': df_15m_aligned
            }
        
        logger.info(f"Successfully processed {len(processed_data)} symbols")
        return processed_data

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class DualStreamLSTMModel:
    """
    Dual-stream LSTM model for cryptocurrency price prediction.
    Stream 1: 1h timeframe (captures trends)
    Stream 2: 15m timeframe (captures volatility)
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
    
    def build(self, n_features):
        """
        Build dual-stream model architecture.
        
        Args:
            n_features (int): Number of input features
            
        Returns:
            tf.keras.Model: Compiled model
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: MODEL ARCHITECTURE CONSTRUCTION")
        logger.info("="*80)
        
        # Input 1: 1h trend (60 bars)
        input_1h = Input(shape=(60, n_features), name='input_1h_trend')
        
        # Input 2: 15m volatility (60 bars)
        input_15m = Input(shape=(60, n_features), name='input_15m_volatility')
        
        # Stream 1: 1h LSTM for trend detection
        logger.info("Building 1h trend stream...")
        x1 = LSTM(64, activation='relu', return_sequences=True)(input_1h)
        x1 = Dropout(0.2)(x1)
        x1 = LSTM(32, activation='relu', return_sequences=False)(x1)
        x1 = Dropout(0.2)(x1)
        
        # Stream 2: CNN + LSTM for volatility detection
        logger.info("Building 15m volatility stream...")
        x2 = Conv1D(filters=32, kernel_size=3, activation='relu')(input_15m)
        x2 = Dropout(0.2)(x2)
        x2 = LSTM(32, activation='relu', return_sequences=False)(x2)
        x2 = Dropout(0.2)(x2)
        
        # Concatenate streams
        logger.info("Merging streams...")
        combined = Concatenate()([x1, x2])
        
        # Dense layers
        z = Dense(32, activation='relu')(combined)
        z = Dropout(0.2)(z)
        z = Dense(16, activation='relu')(z)
        
        # Output: Quantile regression (median, lower, upper bounds)
        if self.config.QUANTILE_REGRESSION:
            logger.info("Using quantile regression output (3 predictions)")
            output = Dense(3, name='quantile_predictions')(z)  # [q10, q50, q90]
        else:
            logger.info("Using standard regression output (1 prediction)")
            output = Dense(1, name='price_prediction')(z)
        
        # Compile model
        model = Model(inputs=[input_1h, input_15m], outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.INITIAL_LEARNING_RATE),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info(f"\nModel Summary:")
        model.summary()
        
        self.model = model
        return model

# ============================================================================
# TRAINING LOOP
# ============================================================================
class Trainer:
    """
    Training orchestrator.
    """
    def __init__(self, model, cache_dir):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.history = None
    
    def prepare_training_data(self, processed_data):
        """
        Prepare combined training data from all symbols.
        
        Args:
            processed_data (dict): Processed data by symbol
            
        Returns:
            tuple: (X_1h, X_15m, y)
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: DATA PREPARATION FOR TRAINING")
        logger.info("="*80)
        
        all_X_1h = []
        all_X_15m = []
        all_y = []
        
        for symbol, data in processed_data.items():
            logger.info(f"Preparing sequences for {symbol}...")
            
            df_1h = data['1h']
            df_15m = data['15m']
            
            engineer = FeatureEngineer(globals())
            X_1h, X_15m, y = engineer.create_dual_stream_sequences(
                df_1h, df_15m, lookback=60
            )
            
            all_X_1h.append(X_1h)
            all_X_15m.append(X_15m)
            all_y.append(y)
        
        # Concatenate all symbols
        X_1h = np.concatenate(all_X_1h, axis=0)
        X_15m = np.concatenate(all_X_15m, axis=0)
        y = np.concatenate(all_y, axis=0)
        
        logger.info(f"Combined training data:")
        logger.info(f"  X_1h shape: {X_1h.shape}")
        logger.info(f"  X_15m shape: {X_15m.shape}")
        logger.info(f"  y shape: {y.shape}")
        
        # Train/Val/Test split (time-series aware)
        n_train = int(len(X_1h) * 0.7)
        n_val = int(len(X_1h) * 0.15)
        
        X_1h_train, X_1h_temp = X_1h[:n_train], X_1h[n_train:]
        X_15m_train, X_15m_temp = X_15m[:n_train], X_15m[n_train:]
        y_train, y_temp = y[:n_train], y[n_train:]
        
        X_1h_val, X_1h_test = X_1h_temp[:n_val], X_1h_temp[n_val:]
        X_15m_val, X_15m_test = X_15m_temp[:n_val], X_15m_temp[n_val:]
        y_val, y_test = y_temp[:n_val], y_temp[n_val:]
        
        logger.info(f"\nData split:")
        logger.info(f"  Train: {len(X_1h_train)} samples")
        logger.info(f"  Val:   {len(X_1h_val)} samples")
        logger.info(f"  Test:  {len(X_1h_test)} samples")
        
        return (X_1h_train, X_15m_train, y_train,
                X_1h_val, X_15m_val, y_val,
                X_1h_test, X_15m_test, y_test)
    
    def train(self, X_1h_train, X_15m_train, y_train,
              X_1h_val, X_15m_val, y_val):
        """
        Train the model.
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 4: MODEL TRAINING")
        logger.info("="*80)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.INITIAL_LEARNING_RATE,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                filepath=str(self.cache_dir / 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        self.history = self.model.fit(
            [X_1h_train, X_15m_train], y_train,
            validation_data=([X_1h_val, X_15m_val], y_val),
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed!")
        return self.history
    
    def evaluate(self, X_1h_test, X_15m_test, y_test):
        """
        Evaluate model on test set.
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 5: MODEL EVALUATION")
        logger.info("="*80)
        
        y_pred = self.model.predict([X_1h_test, X_15m_test])
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        logger.info(f"Test Set Metrics:")
        logger.info(f"  MAE:  {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAPE: {mape:.4f}")
        
        return {'mae': mae, 'rmse': rmse, 'mape': mape}
    
    def save_model(self, name="crypto_v1_model"):
        """
        Save trained model.
        """
        filepath = self.cache_dir / f"{name}.h5"
        self.model.save(str(filepath))
        logger.info(f"Model saved to {filepath}")
        return str(filepath)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """
    Main training pipeline.
    """
    try:
        # Data preparation
        preparer = DataPreparer(CACHE_DIR)
        processed_data = preparer.fetch_and_process_data(
            TRADING_PAIRS,
            TIMEFRAMES
        )
        
        if not processed_data:
            logger.error("No data processed. Exiting.")
            return
        
        # Build model
        n_features = processed_data[list(processed_data.keys())[0]]['1h'].shape[1] - 2  # Exclude timestamp, symbol
        model_builder = DualStreamLSTMModel(globals())  # Use global config
        model = model_builder.build(n_features)
        
        # Prepare training data
        trainer = Trainer(model, CACHE_DIR)
        train_data = trainer.prepare_training_data(processed_data)
        X_1h_train, X_15m_train, y_train, X_1h_val, X_15m_val, y_val, X_1h_test, X_15m_test, y_test = train_data
        
        # Train model
        trainer.train(X_1h_train, X_15m_train, y_train,
                     X_1h_val, X_15m_val, y_val)
        
        # Evaluate model
        metrics = trainer.evaluate(X_1h_test, X_15m_test, y_test)
        
        # Save model
        model_path = trainer.save_model()
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Final metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
