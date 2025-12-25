# ============================================================================
# Crypto System v1 - Feature Engineering Module
# ============================================================================
# Generates technical indicators and volatility-specific features
# Key: Log returns and volatility metrics for handling extreme price moves

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Engineering features specifically designed to handle cryptocurrency volatility.
    Focuses on log returns, volatility measures, and technical indicators.
    """
    
    def __init__(self, config):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: Configuration module with all settings
        """
        self.config = config
        self.scaler_by_symbol = {}  # One scaler per symbol
    
    def add_log_returns(self, df):
        """
        Convert prices to log returns (more stable across different price scales).
        This is crucial when training on 20+ coins with vastly different prices.
        BTC @ 90k vs DOGE @ 0.3 need to be on same scale.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: With log return columns
        """
        df['log_close'] = np.log(df['close'])
        df['log_return'] = df['log_close'].diff()
        df['log_return_std'] = df['log_return'].rolling(window=20).std()
        
        logger.info(f"Added log returns (mean: {df['log_return'].mean():.6f}, "
                   f"std: {df['log_return'].std():.6f})")
        return df
    
    def add_volatility_features(self, df):
        """
        Add volatility-specific features to detect extreme moves.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: With volatility features
        """
        period = self.config.VOLATILITY_FEATURES['rolling_volatility']
        
        # Rolling volatility (standard deviation of returns)
        df['volatility'] = df['log_return'].rolling(window=period).std()
        
        # True Range (for ATR-like calculation)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=period).mean()
        
        # Bollinger Band Width (low width = potential breakout)
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volatility ratio (current vol / average vol)
        df['volatility_ma'] = df['volatility'].rolling(window=period).mean()
        df['volatility_ratio'] = df['volatility'] / (df['volatility_ma'] + 1e-8)
        
        logger.info(f"Added {len(df.columns)} volatility features")
        return df
    
    def add_technical_indicators(self, df):
        """
        Add standard technical indicators.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: With technical indicators
        """
        # RSI (Relative Strength Index)
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(14)
        df['momentum_roc'] = df['close'].pct_change(periods=14)
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
        
        logger.info(f"Added technical indicators (RSI, MACD, Momentum)")
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """
        Calculate RSI (Relative Strength Index).
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence).
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        
        return macd, macd_signal, macd_hist
    
    def normalize_features(self, df, symbol=None):
        """
        Normalize features to [0, 1] range using MinMaxScaler.
        Create separate scaler per symbol to handle different price ranges.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            symbol (str): Symbol name for scaler key
            
        Returns:
            pd.DataFrame: Normalized data
        """
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'symbol', 'log_close']]
        
        if symbol and symbol not in self.scaler_by_symbol:
            self.scaler_by_symbol[symbol] = MinMaxScaler()
            self.scaler_by_symbol[symbol].fit(df[feature_cols])
        
        scaler = self.scaler_by_symbol.get(symbol, MinMaxScaler())
        df_normalized = df.copy()
        df_normalized[feature_cols] = scaler.transform(df[feature_cols])
        
        logger.info(f"Normalized {len(feature_cols)} features")
        return df_normalized
    
    def create_sequences(self, df, lookback=60, lookahead=1):
        """
        Create time series sequences for LSTM input.
        
        Args:
            df (pd.DataFrame): Normalized data with features
            lookback (int): Input sequence length (60 bars)
            lookahead (int): Steps ahead to predict (1 bar)
            
        Returns:
            tuple: (X, y) where X is sequences, y is targets
        """
        # Select feature columns (exclude timestamp, symbol, etc.)
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'symbol', 'log_close']]
        
        X = []
        y = []
        
        for i in range(len(df) - lookback - lookahead + 1):
            # Input: 60 bars of features
            X.append(df[feature_cols].iloc[i:i+lookback].values)
            # Target: next bar's close price (or log return)
            y.append(df['close'].iloc[i+lookback+lookahead-1])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def create_dual_stream_sequences(self, df_1h, df_15m, lookback=60):
        """
        Create dual-stream sequences for 1h (trend) + 15m (volatility) architecture.
        
        Args:
            df_1h (pd.DataFrame): 1h data
            df_15m (pd.DataFrame): 15m data
            lookback (int): Input sequence length
            
        Returns:
            tuple: (X_1h, X_15m, y) sequences
        """
        feature_cols = [col for col in df_1h.columns 
                       if col not in ['timestamp', 'symbol', 'log_close']]
        
        X_1h = []
        X_15m = []
        y = []
        
        # For each 1h bar, get corresponding 15m bars (4 bars per 1h)
        min_length = min(len(df_1h), len(df_15m) // 4)
        
        for i in range(min_length - lookback):
            # 1h trend input (60 bars = 60 hours)
            X_1h.append(df_1h[feature_cols].iloc[i:i+lookback].values)
            
            # 15m volatility input (60 bars = 15 hours)
            start_15m = i * 4
            X_15m.append(df_15m[feature_cols].iloc[start_15m:start_15m+lookback*4:4].values)
            
            # Target (next 1h close)
            y.append(df_1h['close'].iloc[i+lookback])
        
        X_1h = np.array(X_1h)
        X_15m = np.array(X_15m)
        y = np.array(y)
        
        logger.info(f"Created dual-stream sequences: ")
        logger.info(f"  X_1h: {X_1h.shape}, X_15m: {X_15m.shape}, y: {y.shape}")
        
        return X_1h, X_15m, y
    
    def handle_missing_values(self, df, method='forward_fill'):
        """
        Handle missing values in data.
        
        Args:
            df (pd.DataFrame): Data with potential missing values
            method (str): 'forward_fill' or 'interpolate'
            
        Returns:
            pd.DataFrame: Clean data
        """
        initial_nans = df.isna().sum().sum()
        
        if method == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'interpolate':
            df = df.interpolate(method='linear').fillna(method='bfill')
        
        final_nans = df.isna().sum().sum()
        logger.info(f"Handled missing values: {initial_nans} -> {final_nans}")
        
        return df
    
    def handle_outliers(self, df, method='iqr', threshold=3):
        """
        Handle outliers in volatility data.
        
        Args:
            df (pd.DataFrame): Data with potential outliers
            method (str): 'iqr' or 'zscore'
            threshold (float): Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Data with outliers handled
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                df.loc[(df[col] < lower) | (df[col] > upper), col] = df[col].median()
            
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                df.loc[abs((df[col] - mean) / std) > threshold, col] = mean
        
        logger.info(f"Handled outliers using {method}")
        return df


def process_single_symbol(df, symbol, config):
    """
    Complete feature engineering pipeline for single symbol.
    
    Args:
        df (pd.DataFrame): Raw OHLCV data
        symbol (str): Trading pair
        config: Configuration object
        
    Returns:
        pd.DataFrame: Fully processed data
    """
    engineer = FeatureEngineer(config)
    
    # Processing pipeline
    logger.info(f"Processing {symbol}...")
    df = engineer.handle_missing_values(df)
    df = engineer.add_log_returns(df)
    df = engineer.add_volatility_features(df)
    df = engineer.add_technical_indicators(df)
    df = engineer.handle_outliers(df)
    df = engineer.normalize_features(df, symbol)
    
    return df
