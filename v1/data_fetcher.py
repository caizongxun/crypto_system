# ============================================================================
# Crypto System v1 - Data Fetcher Module
# ============================================================================
# Fetches cryptocurrency data from Binance US API with proper rate limiting
# and stores in Parquet format for efficiency

import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptodataFetcher:
    """
    Fetches cryptocurrency OHLCV data from Binance US with automatic rate limiting
    and storage optimization using Parquet format.
    """
    
    def __init__(self, cache_dir="/content/all_models"):
        """
        Initialize data fetcher.
        
        Args:
            cache_dir (str): Local cache directory for storing data
        """
        self.exchange = ccxt.binanceus()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.cache_dir / "crypto_data" / "v1"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        
    def _respect_rate_limit(self):
        """Ensure we don't exceed Binance API rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def fetch_ohlcv(self, symbol, timeframe="1h", limit=1000, retries=3):
        """
        Fetch OHLCV data with automatic retry logic.
        
        Args:
            symbol (str): Trading pair (e.g., 'BTC/USDT')
            timeframe (str): '1h', '15m', '4h', '1d', etc.
            limit (int): Number of candles to fetch
            retries (int): Number of retries on failure
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        for attempt in range(retries):
            try:
                self._respect_rate_limit()
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = symbol
                
                logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
                return df
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {symbol}: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {symbol} after {retries} attempts")
                    return None
    
    def fetch_multiple_symbols(self, symbols, timeframe="1h", limit=1000):
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols (list): List of trading pairs
            timeframe (str): Timeframe
            limit (int): Candles per symbol
            
        Returns:
            dict: {symbol: DataFrame}
        """
        all_data = {}
        
        for symbol in symbols:
            logger.info(f"Fetching {symbol} ({timeframe})...")
            df = self.fetch_ohlcv(symbol, timeframe, limit)
            if df is not None:
                all_data[symbol] = df
            else:
                logger.warning(f"Skipping {symbol} due to fetch failure")
        
        logger.info(f"Successfully fetched {len(all_data)}/{len(symbols)} symbols")
        return all_data
    
    def save_to_parquet(self, df, symbol, timeframe, version="v1"):
        """
        Save DataFrame to Parquet format (more efficient than CSV).
        
        Args:
            df (pd.DataFrame): Data to save
            symbol (str): Trading pair
            timeframe (str): Timeframe
            version (str): Version folder
        """
        safe_symbol = symbol.replace('/', '_')
        filepath = self.data_dir / f"{safe_symbol}_{timeframe}.parquet"
        
        try:
            df.to_parquet(filepath, compression='snappy', index=False)
            logger.info(f"Saved {len(df)} rows to {filepath}")
        except Exception as e:
            logger.error(f"Error saving to parquet: {str(e)}")
    
    def load_from_parquet(self, symbol, timeframe):
        """
        Load data from Parquet format.
        
        Args:
            symbol (str): Trading pair
            timeframe (str): Timeframe
            
        Returns:
            pd.DataFrame: Loaded data or None
        """
        safe_symbol = symbol.replace('/', '_')
        filepath = self.data_dir / f"{safe_symbol}_{timeframe}.parquet"
        
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                logger.info(f"Loaded {len(df)} rows from {filepath}")
                return df
            except Exception as e:
                logger.error(f"Error loading from parquet: {str(e)}")
                return None
        else:
            logger.warning(f"File not found: {filepath}")
            return None
    
    def fetch_and_save_batch(self, symbols, timeframes=["1h", "15m"]):
        """
        Fetch and save data for all symbols and timeframes.
        
        Args:
            symbols (list): Trading pairs
            timeframes (list): List of timeframes
        """
        total = len(symbols) * len(timeframes)
        completed = 0
        
        for timeframe in timeframes:
            logger.info(f"\n=== Fetching {timeframe} data ===")
            data = self.fetch_multiple_symbols(symbols, timeframe)
            
            for symbol, df in data.items():
                self.save_to_parquet(df, symbol, timeframe)
                completed += 1
                logger.info(f"Progress: {completed}/{total}")
    
    def merge_timeframes(self, symbol, df_1h, df_15m, fill_method='ffill'):
        """
        Merge 1h and 15m data for dual-stream input.
        
        Args:
            symbol (str): Trading pair
            df_1h (pd.DataFrame): 1h data
            df_15m (pd.DataFrame): 15m data
            fill_method (str): Forward fill or interpolate
            
        Returns:
            tuple: (df_1h_aligned, df_15m_aligned)
        """
        # Ensure same time range
        min_time = max(df_1h['timestamp'].min(), df_15m['timestamp'].min())
        max_time = min(df_1h['timestamp'].max(), df_15m['timestamp'].max())
        
        df_1h_filtered = df_1h[(df_1h['timestamp'] >= min_time) & 
                                (df_1h['timestamp'] <= max_time)].copy()
        df_15m_filtered = df_15m[(df_15m['timestamp'] >= min_time) & 
                                  (df_15m['timestamp'] <= max_time)].copy()
        
        logger.info(f"Aligned {symbol}: {len(df_1h_filtered)} 1h bars, "
                   f"{len(df_15m_filtered)} 15m bars")
        
        return df_1h_filtered, df_15m_filtered


def main():
    """Example usage of data fetcher."""
    from config import TRADING_PAIRS, TIMEFRAMES
    
    fetcher = CryptodataFetcher()
    
    # Fetch data for all pairs and timeframes
    logger.info("Starting batch data fetch...")
    fetcher.fetch_and_save_batch(
        TRADING_PAIRS,
        timeframes=[TIMEFRAMES["trend"], TIMEFRAMES["volatility"]]
    )
    logger.info("Batch fetch completed!")


if __name__ == "__main__":
    main()
