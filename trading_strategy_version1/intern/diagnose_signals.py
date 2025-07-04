"""
診斷交易訊號問題
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 導入主程式的函數
from main_pipeline import resample_to_4h, compute_indicators, generate_entry_signal, generate_exit_signal

def diagnose_signals():
    """診斷為什麼沒有交易訊號"""
    print("🔍 診斷交易訊號問題")
    print("="*50)
    
    # 讀取資料
    print("📖 讀取資料...")
    df = pd.read_csv('../scripts/TXF1_Minute_2020-01-01_2025-06-16.txt')
    df = df.dropna()
    df = df[df['Close'] > 0]
    
    # 轉換為15分鐘K
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('Datetime').resample('15T').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'TotalVolume': 'sum',
        'Date': 'first',
        'Time': 'first'
    }).dropna().reset_index(drop=True)
    
    print(f"資料筆數: {len(df)}")
    
    # 測試參數
    params = {
        'bb_window': 20,
        'bb_std': 2.5,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_exit': 50,
        'rsi_overbought': 70,
        'rsi_exit_short': 50,
        'obv_ma_window': 10,
        'stop_loss': 20,
        'stop_loss_pct': 0.003,
        'require_reversal_kbar': True,
        'max_hold_bars': 5,
        'entry_n': 2
    }
    
    # 計算指標
    print("📊 計算指標...")
    df_4h = resample_to_4h(df)
    df = compute_indicators(df, params, df_4h)
    
    # 檢查四小時RSI
    print(f"\n📈 四小時RSI統計:")
    print(f"四小時RSI平均值: {df['RSI_4H'].mean():.2f}")
    print(f"四小時RSI最小值: {df['RSI_4H'].min():.2f}")
    print(f"四小時RSI最大值: {df['RSI_4H'].max():.2f}")
    print(f"四小時RSI < 50 的次數: {(df['RSI_4H'] < 50).sum()}")
    print(f"四小時RSI > 50 的次數: {(df['RSI_4H'] > 50).sum()}")
    
    # 檢查15分鐘RSI
    print(f"\n📈 15分鐘RSI統計:")
    print(f"15分鐘RSI平均值: {df['RSI'].mean():.2f}")
    print(f"15分鐘RSI < 30 的次數: {(df['RSI'] < 30).sum()}")
    print(f"15分鐘RSI > 70 的次數: {(df['RSI'] > 70).sum()}")
    
    # 檢查布林通道
    print(f"\n📊 布林通道統計:")
    print(f"價格 <= 下軌的次數: {(df['Close'] <= df['BB_LOWER']).sum()}")
    print(f"價格 >= 上軌的次數: {(df['Close'] >= df['BB_UPPER']).sum()}")
    
    # 檢查均線
    print(f"\n📈 均線統計:")
    print(f"MA5 > MA20 的次數: {(df['MA5'] > df['MA20']).sum()}")
    print(f"MA5 < MA20 的次數: {(df['MA5'] < df['MA20']).sum()}")
    
    # 檢查K線型態
    print(f"\n🕯️ K線型態統計:")
    print(f"錘頭次數: {df['Hammer'].sum()}")
    print(f"多頭吞噬次數: {df['Bull_Engulfing'].sum()}")
    print(f"空頭吞噬次數: {df['Bear_Engulfing'].sum()}")
    print(f"墓碑線次數: {df['Gravestone'].sum()}")
    print(f"十字星次數: {df['Doji'].sum()}")
    
    # 產生訊號
    print("\n🎯 產生訊號...")
    df = generate_entry_signal(df, params)
    df = generate_exit_signal(df, params)
    
    # 檢查各個條件
    print(f"\n📊 條件統計:")
    print(f"BB條件(多): {df['BB_Condition_Long'].sum()}")
    print(f"RSI條件(多): {df['RSI_Condition_Long'].sum()}")
    print(f"四小時趨勢條件(多): {df['4H_Trend_Condition_Long'].sum()}")
    print(f"BB條件(空): {df['BB_Condition_Short'].sum()}")
    print(f"RSI條件(空): {df['RSI_Condition_Short'].sum()}")
    print(f"四小時趨勢條件(空): {df['4H_Trend_Condition_Short'].sum()}")
    
    # 檢查進場訊號
    print(f"\n🚀 進場訊號統計:")
    print(f"多單進場訊號: {df['EntrySignal_Long'].sum()}")
    print(f"空單進場訊號: {df['EntrySignal_Short'].sum()}")
    print(f"總進場訊號: {df['EntrySignal_Long'].sum() + df['EntrySignal_Short'].sum()}")
    
    # 檢查條件組合
    print(f"\n📋 條件組合統計:")
    condition_stats_long = df['Conditions_Met_Long'].value_counts().sort_index()
    print("多單條件組合:")
    for conditions, count in condition_stats_long.items():
        print(f"  滿足 {conditions} 個條件: {count} 次")
    
    condition_stats_short = df['Conditions_Met_Short'].value_counts().sort_index()
    print("空單條件組合:")
    for conditions, count in condition_stats_short.items():
        print(f"  滿足 {conditions} 個條件: {count} 次")
    
    # 檢查最近的訊號
    print(f"\n🕐 最近10筆資料的訊號:")
    recent_data = df[['Date', 'Time', 'Close', 'RSI', 'RSI_4H', 'BB_Condition_Long', 'RSI_Condition_Long', '4H_Trend_Condition_Long', 'EntrySignal_Long', 'BB_Condition_Short', 'RSI_Condition_Short', '4H_Trend_Condition_Short', 'EntrySignal_Short']].tail(10)
    print(recent_data)
    
    # 分析四小時趨勢條件的問題
    print(f"\n🔍 四小時趨勢條件分析:")
    long_trend_false = df[~df['4H_Trend_Condition_Long'] & df['BB_Condition_Long'] & df['RSI_Condition_Long']]
    short_trend_false = df[~df['4H_Trend_Condition_Short'] & df['BB_Condition_Short'] & df['RSI_Condition_Short']]
    
    print(f"多單：BB和RSI條件滿足但四小時趨勢不滿足的次數: {len(long_trend_false)}")
    print(f"空單：BB和RSI條件滿足但四小時趨勢不滿足的次數: {len(short_trend_false)}")
    
    if len(long_trend_false) > 0:
        print(f"\n多單四小時趨勢不滿足的範例:")
        sample = long_trend_false[['Date', 'Time', 'Close', 'RSI', 'RSI_4H', 'MA5', '4H_Trend_Condition_Long']].head(5)
        print(sample)
    
    if len(short_trend_false) > 0:
        print(f"\n空單四小時趨勢不滿足的範例:")
        sample = short_trend_false[['Date', 'Time', 'Close', 'RSI', 'RSI_4H', 'MA5', '4H_Trend_Condition_Short']].head(5)
        print(sample)

if __name__ == '__main__':
    diagnose_signals() 