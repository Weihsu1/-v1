"""
成本考量（Cost Modeling）模組
- 每筆交易滑價 / 手續費：固定 1.5 點
- 做多與做空不同滑價
- 計算成本後的真實報酬
"""

import pandas as pd
import numpy as np

def apply_cost_model(trades_df, fee=1.5, slippage_long=1.0, slippage_short=2.0):
    """
    套用成本模型到交易紀錄
    
    參數:
    trades_df: 交易紀錄DataFrame，包含 EntryTime, ExitTime, EntryPrice, ExitPrice, Direction
    fee: 手續費（點數）
    slippage_long: 做多滑價（點數）
    slippage_short: 做空滑價（點數）
    
    返回:
    包含成本計算的交易紀錄DataFrame
    """
    
    if len(trades_df) == 0:
        return pd.DataFrame()
    
    # 複製交易紀錄
    result = trades_df.copy()
    
    # 計算毛損益
    result['GrossPnL'] = (result['ExitPrice'] - result['EntryPrice']) * result['Direction']
    
    # 計算成本
    result['EntryCost'] = fee + np.where(result['Direction'] == 1, slippage_long, slippage_short)
    result['ExitCost'] = fee + np.where(result['Direction'] == 1, slippage_long, slippage_short)
    result['TotalCost'] = result['EntryCost'] + result['ExitCost']
    
    # 計算淨損益
    result['NetPnL'] = result['GrossPnL'] - result['TotalCost']
    
    # 計算勝率
    result['Win'] = result['NetPnL'] > 0
    
    return result

def calculate_performance_metrics(trades_df):
    """
    計算績效指標
    
    參數:
    trades_df: 包含NetPnL的交易紀錄DataFrame
    
    返回:
    績效指標字典
    """
    
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
    
    # 基本統計
    total_trades = len(trades_df)
    win_trades = (trades_df['NetPnL'] > 0).sum()
    win_rate = win_trades / total_trades if total_trades > 0 else 0
    
    # 損益統計
    total_pnl = trades_df['NetPnL'].sum()
    avg_pnl = trades_df['NetPnL'].mean()
    
    # 最大回撤（簡化計算）
    cumulative_pnl = trades_df['NetPnL'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - running_max
    max_drawdown = drawdown.min()
    
    # 夏普比率（簡化計算，假設無風險利率為0）
    returns = trades_df['NetPnL']
    sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }

def print_performance_summary(trades_df):
    """
    印出績效摘要
    
    參數:
    trades_df: 交易紀錄DataFrame
    """
    
    if len(trades_df) == 0:
        print("無交易紀錄")
        return
    
    metrics = calculate_performance_metrics(trades_df)
    
    print("="*50)
    print("📊 交易績效摘要")
    print("="*50)
    print(f"總交易次數: {metrics['total_trades']}")
    print(f"勝率: {metrics['win_rate']:.2%}")
    print(f"總損益: {metrics['total_pnl']:.2f} 點")
    print(f"平均損益: {metrics['avg_pnl']:.2f} 點")
    print(f"最大回撤: {metrics['max_drawdown']:.2f} 點")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print("="*50)
    
    # 顯示前幾筆交易
    print("\n前5筆交易:")
    display_cols = ['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'Direction', 'GrossPnL', 'TotalCost', 'NetPnL']
    print(trades_df[display_cols].head()) 