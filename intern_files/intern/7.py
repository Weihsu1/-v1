import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 直接指定.py副檔名
import importlib.util
module_path = os.path.join(os.path.dirname(__file__), '1')
spec = importlib.util.spec_from_file_location("strategy_module", module_path)
strategy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy_module)

from strategy_module import (
    load_txf_data, 
    calculate_bollinger_bands, 
    calculate_rsi, 
    calculate_obv,
    generate_signals,
    calculate_returns,
    optimize_parameters
)
import pandas as pd
import numpy as np

def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    try:
        file_path = "TXF1_Minute_2020-01-01_2025-06-16.txt"
        df = load_txf_data(file_path)
        print(f"Data loaded successfully: {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def test_indicators(df):
    """Test technical indicators calculation."""
    print("\nTesting technical indicators...")
    try:
        # Test Bollinger Bands
        df_bb = calculate_bollinger_bands(df, window=20, num_std=2.0)
        print("Bollinger Bands calculated successfully")
        
        # Test RSI
        df_rsi = calculate_rsi(df_bb, window=14)
        print("RSI calculated successfully")
        
        # Test OBV
        df_obv = calculate_obv(df_rsi)
        print("OBV calculated successfully")
        
        # Check for NaN values
        nan_count = df_obv.isnull().sum()
        print(f"NaN values in indicators: {nan_count.to_dict()}")
        
        return df_obv
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return None

def test_strategy(df):
    """Test strategy signal generation."""
    print("\nTesting strategy signals...")
    try:
        params = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'obv_threshold': 1.2
        }
        
        df_signals = generate_signals(df, params)
        signals = df_signals['signal'].value_counts()
        print(f"Signal distribution: {signals.to_dict()}")
        
        return df_signals
    except Exception as e:
        print(f"Error generating signals: {e}")
        return None

def test_performance(df):
    """Test performance calculation."""
    print("\nTesting performance calculation...")
    try:
        params = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'obv_threshold': 1.2,
            'transaction_cost': 0.0001
        }
        
        metrics = calculate_returns(df, params)
        print("Performance metrics calculated successfully:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return metrics
    except Exception as e:
        print(f"Error calculating performance: {e}")
        return None

def test_optimization(df):
    """Test parameter optimization with reduced parameter space."""
    print("\nTesting parameter optimization...")
    try:
        # Use smaller parameter ranges for testing
        param_ranges = {
            'bb_window': [15, 20],
            'bb_std': [2.0, 2.5],
            'rsi_window': [14],
            'rsi_oversold': [25, 30],
            'rsi_overbought': [70, 75],
            'obv_threshold': [1.2, 1.3]
        }
        
        best_params, best_metrics = optimize_parameters(df, param_ranges)
        
        print("Optimization completed successfully!")
        print("Best parameters:", best_params)
        print("Best metrics:", best_metrics)
        
        return best_params, best_metrics
    except Exception as e:
        print(f"Error in optimization: {e}")
        return None, None

def main():
    """Run all tests."""
    print("="*60)
    print("QUANTITATIVE TRADING SYSTEM - PARAMETER OPTIMIZATION TEST")
    print("="*60)
    
    # Test 1: Data loading
    df = test_data_loading()
    if df is None:
        print("Data loading failed. Exiting.")
        return
    
    # Test 2: Technical indicators
    df_indicators = test_indicators(df)
    if df_indicators is None:
        print("Indicator calculation failed. Exiting.")
        return
    
    # Test 3: Strategy signals
    df_signals = test_strategy(df_indicators)
    if df_signals is None:
        print("Signal generation failed. Exiting.")
        return
    
    # Test 4: Performance calculation
    metrics = test_performance(df_signals)
    if metrics is None:
        print("Performance calculation failed. Exiting.")
        return
    
    # Test 5: Parameter optimization
    best_params, best_metrics = test_optimization(df_indicators)
    if best_params is None:
        print("Optimization failed. Exiting.")
        return
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("The parameter optimization system is ready for production use.")

if __name__ == "__main__":
    main()