#!/usr/bin/env python3
# ============================================================================
# Walk-Forward Backtesting v2
# ============================================================================
# Tests model performance across rolling windows.
# More realistic than single train/val/test split.

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def walk_forward_test(
    X_sorted, y_sorted, dates_sorted,
    model_callable,
    scaler_X, scaler_y,
    initial_train_size=500,
    window_size=200,
    step_size=100
):
    """
    Walk-forward evaluation.
    
    Args:
        X_sorted, y_sorted, dates_sorted: Chronologically sorted data
        model_callable: Function that builds and trains model
        scaler_X, scaler_y: Pre-fitted scalers
        initial_train_size: Initial training samples
        window_size: Test window size
        step_size: How many samples to roll forward
        
    Returns:
        List of dicts with metrics per window
    """
    results = []
    n_total = len(X_sorted)
    
    print("\n" + "="*80)
    print("WALK-FORWARD BACKTEST")
    print("="*80 + "\n")
    
    idx = initial_train_size
    window_num = 0
    
    while idx + window_size <= n_total:
        window_num += 1
        
        # Define windows
        test_start = idx
        test_end = idx + window_size
        train_end = idx
        train_start = max(0, train_end - initial_train_size * 2)  # Expanding window
        
        # Get data
        X_train_wf = X_sorted[train_start:train_end]
        y_train_wf = y_sorted[train_start:train_end]
        
        X_test_wf = X_sorted[test_start:test_end]
        y_test_wf = y_sorted[test_start:test_end]
        dates_test_wf = dates_sorted[test_start:test_end]
        
        print(f"Window {window_num}:")
        print(f"  Train: {dates_sorted[train_start]} to {dates_sorted[train_end-1]}")
        print(f"  Test:  {dates_test_wf[0]} to {dates_test_wf[-1]}")
        print(f"  Train size: {len(X_train_wf)}, Test size: {len(X_test_wf)}")
        
        try:
            # Train model
            model, history = model_callable(
                X_train_wf, y_train_wf,
                scaler_X, scaler_y
            )
            
            # Test
            from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
            
            X_test_norm = X_test_wf.reshape(-1, X_test_wf.shape[2])
            X_test_norm = scaler_X.transform(X_test_norm)
            X_test_norm = X_test_norm.reshape(X_test_wf.shape)
            
            y_pred_norm = model.predict([X_test_norm, X_test_norm], verbose=0).flatten()
            y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
            
            mae = mean_absolute_error(y_test_wf, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_wf, y_pred))
            mape = mean_absolute_percentage_error(y_test_wf, y_pred)
            
            # Direction accuracy
            direction_correct = np.sum((y_test_wf * y_pred) > 0)
            dir_acc = direction_correct / len(y_test_wf) * 100
            
            result = {
                'window': window_num,
                'train_start': dates_sorted[train_start],
                'train_end': dates_sorted[train_end-1],
                'test_start': dates_test_wf[0],
                'test_end': dates_test_wf[-1],
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'direction_accuracy': dir_acc
            }
            results.append(result)
            
            print(f"  Results: MAPE={mape*100:.4f}%, Dir.Acc={dir_acc:.2f}%")
            print()
            
            # Clean up
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"  Error: {e}")
            print()
        
        idx += step_size
    
    # Summary
    if results:
        df_results = pd.DataFrame(results)
        print("\n" + "="*80)
        print("WALK-FORWARD SUMMARY")
        print("="*80)
        print(f"  Avg MAPE: {df_results['mape'].mean()*100:.4f}%")
        print(f"  Std MAPE: {df_results['mape'].std()*100:.4f}%")
        print(f"  Best MAPE: {df_results['mape'].min()*100:.4f}%")
        print(f"  Worst MAPE: {df_results['mape'].max()*100:.4f}%")
        print(f"  Avg Dir.Acc: {df_results['direction_accuracy'].mean():.2f}%")
        print()
        return df_results
    else:
        print("No results generated.")
        return None


if __name__ == "__main__":
    print("Walk-forward testing requires the full data pipeline.")
    print("Use this module within train_v2_production.py")
