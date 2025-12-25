# ============================================================================
# Crypto System v1 - Configuration File
# ============================================================================
# This file contains all hyperparameters, settings, and configuration
# for the v1 cryptocurrency prediction model

import os
from datetime import datetime

# ============================================================================
# 1. PROJECT METADATA
# ============================================================================
VERSION = "v1"
MODEL_NAME = "crypto_dual_lstm_v1"
CREATED_DATE = "2025-12-25"
DESCRIPTION = "Multi-currency LSTM model with dual timeframes (1h, 15m) for volatility handling"

# ============================================================================
# 2. DATA CONFIGURATION
# ============================================================================
# Binance US API Settings
EXCHANGE = "binanceus"
API_KEY = ""  # Leave empty if not using authenticated endpoints
API_SECRET = ""

# Trading Pairs (20+ cryptocurrencies)
TRADING_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
    "DOGE/USDT", "LINK/USDT", "MATIC/USDT", "AVAX/USDT", "DOT/USDT",
    "UNI/USDT", "LTC/USDT", "BCH/USDT", "ETC/USDT", "XLM/USDT",
    "ATOM/USDT", "ICP/USDT", "NEAR/USDT", "ARB/USDT", "SHIB/USDT",
    "OP/USDT", "PEPE/USDT"
]

# Timeframes
TIMEFRAMES = {
    "trend": "1h",      # Long-term trend (60 bars = 60 hours)
    "volatility": "15m" # Short-term volatility (60 bars = 15 hours)
}

# Data Collection Settings
LOOKBACK_PERIOD = 1000  # Fetch 1000 candles per request
MIN_LOOKBACK_DAYS = 365  # Minimum 1 year of historical data
DATA_FORMAT = "parquet"  # Use parquet instead of CSV (faster, smaller)

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
# OHLCV Base Features
BASE_FEATURES = ["open", "high", "low", "close", "volume"]

# Technical Indicators
INDICATORS = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.0,
    "atr_period": 14,
}

# Volatility Features (Key for handling extreme moves)
VOLATILITY_FEATURES = {
    "log_return": True,  # Log returns instead of percentage
    "rolling_volatility": 14,  # 14-period rolling std
    "rolling_range": 14,  # High-Low range
    "close_to_open_ratio": True,  # Intra-bar movement
}

# Data Normalization
NORMALIZATION_METHOD = "z_score"  # "z_score", "minmax", or "log_return"
NORMALIZE_BY_PAIR = True  # Normalize each pair independently

# ============================================================================
# 4. MODEL ARCHITECTURE (LSTM + CNN Dual-Stream)
# ============================================================================
SEQUENCE_LENGTH = 60  # Input window: 60 bars

# LSTM Parameters
LSTM_1H_UNITS = [64, 32]  # Units for trend branch (1h)
LSTM_15M_UNITS = [32, 16]  # Units for volatility branch (15m)
LSTM_DROPOUT = 0.2
LSTM_RECURRENT_DROPOUT = 0.1

# CNN Parameters (for 15m volatility detection)
CNN_FILTERS = [32, 16]
CNN_KERNEL_SIZE = 3
CNN_DROPOUT = 0.2

# Attention Mechanism (Optional)
USE_ATTENTION = True
ATTENTION_HEADS = 4

# Dense Layers
DENSE_UNITS = [32, 16]
DENSE_DROPOUT = 0.2

# ============================================================================
# 5. OUTPUT CONFIGURATION (Quantile Regression for Volatility Prediction)
# ============================================================================
# Instead of predicting single price, predict price distribution
QUANTILE_REGRESSION = True
QUANTILE_LEVELS = [0.1, 0.5, 0.9]  # 10%, 50% (median), 90% percentiles

# Standard output (if not using quantile)
OUTPUT_FEATURES = 1  # Single price prediction

# ============================================================================
# 6. TRAINING CONFIGURATION
# ============================================================================
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LEARNING_RATE = 0.001
LEARNING_RATE_REDUCTION = {
    "factor": 0.5,
    "patience": 5,
    "min_lr": 1e-6
}

# Early Stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_METRIC = "val_loss"

# Validation & Test Split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Train/Val/Test Split Method
SPLIT_METHOD = "time_series"  # Don't shuffle (preserve temporal order)
VALIDATION_FREQUENCY = 1  # Validate every N epochs

# ============================================================================
# 7. LOSS FUNCTION & METRICS
# ============================================================================
# Loss Function: Weighted MSE for volatility sensitivity
LOSS_FUNCTION = "weighted_mse"  # "mse", "mae", "weighted_mse", "quantile_loss"
LOSS_WEIGHTS = {
    "normal_volatility": 1.0,
    "high_volatility": 5.0,  # 5x penalty for volatile periods
    "volatility_threshold": 0.02  # 2% price change threshold
}

# Metrics to Monitor
METRICS = ["mae", "mape"]
CUSTOM_METRICS = ["direction_accuracy", "volatility_rmse"]

# ============================================================================
# 8. DATA AUGMENTATION & HANDLING
# ============================================================================
HANDLE_MISSING_DATA = True
MISSING_DATA_STRATEGY = "forward_fill"  # "forward_fill", "interpolate", or "drop"

HANDLE_OUTLIERS = True
OUTLIER_METHOD = "iqr"  # "iqr" or "zscore"
OUTLIER_THRESHOLD = 3  # 3-sigma for zscore

# ============================================================================
# 9. COLAB & STORAGE CONFIGURATION
# ============================================================================
# Local Storage
LOCAL_CACHE_DIR = "/content/all_models"  # Colab cache folder
LOCAL_DATA_DIR = "/content/crypto_data"

# Google Drive (for persistent storage)
USE_GOOGLE_DRIVE = True
DRIVE_MOUNT_POINT = "/content/drive"
DRIVE_MODEL_DIR = "/content/drive/MyDrive/crypto_system/models"
DRIVE_DATA_DIR = "/content/drive/MyDrive/crypto_system/data"

# Model Checkpointing
SAVE_MODEL_EVERY_N_EPOCHS = 5
SAVE_BEST_MODEL = True
MODEL_CHECKPOINT_DIR = f"{LOCAL_CACHE_DIR}/{VERSION}"

# ============================================================================
# 10. REMOTE EXECUTION (GitHub-based)
# ============================================================================
GITHUB_REPO = "https://github.com/caizongxun/crypto_system"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/caizongxun/crypto_system/main"
CURRENT_VERSION_URL = f"{GITHUB_RAW_URL}/v1/train_v1.py"

# ============================================================================
# 11. BACKTESTING & TRADING LOGIC
# ============================================================================
# Entry/Exit Rules (for future backtesting)
BACKTEST_ENABLED = False
TAKE_PROFIT_PERCENT = 0.02  # 2% profit target
STOP_LOSS_PERCENT = 0.01  # 1% stop loss
LOOKBACK_BARS_FOR_EXTREMA = 10  # Find extreme points in 10-bar window

# Volatility-based Risk Management
VOLATILITY_BASED_RISK = True
HIGH_VOLATILITY_THRESHOLD = 0.03  # 3% volatility pause trading
VOLATILITY_CHECK_PERIOD = 20  # Check volatility over last 20 bars

# ============================================================================
# 12. LOGGING & MONITORING
# ============================================================================
LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING"
LOG_FILE = f"{LOCAL_CACHE_DIR}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
PLOT_TRAINING_HISTORY = True
SAVE_PLOTS = True
PLOT_SAVE_DIR = f"{LOCAL_CACHE_DIR}/plots"

# ============================================================================
# 13. GPU & RESOURCE MANAGEMENT (Colab)
# ============================================================================
GPU_MEMORY_FRACTION = 0.9  # Use 90% of GPU memory
MIXED_PRECISION_TRAINING = True  # Use float16 for faster training

# ============================================================================
# 14. MODEL EVALUATION THRESHOLDS
# ============================================================================
ACCEPTABLE_MAPE = 0.05  # 5% acceptable MAPE
ACCEPTABLE_DIRECTION_ACCURACY = 0.55  # 55% direction accuracy
ACCEPTABLE_SHARPE_RATIO = 1.0  # For backtesting
