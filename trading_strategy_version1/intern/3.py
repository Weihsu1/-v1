"""
### 6️⃣ **穩健性測試（Robustness）**

* Walk-forward 訓練/測試分段
* 資料洗牌 / 時間打亂
* 噪聲注入（價格加 ±0.2%）
* 年份區間測試（2018–2022 漲跌期）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import random
from sklearn.model_selection import TimeSeriesSplit
import sys
import os

# 連結1.py和2.py的模組
from cost_model import apply_cost_model

warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_palette("husl")

print("穩健性測試環境設定完成！")

# =====================
# 從1.py導入核心函數
# =====================
def resample_to_4h(df):
    """從1.py導入的4小時重採樣函數"""
    df_4h = df.copy()
    df_4h['Datetime'] = pd.to_datetime(df_4h['Date'] + ' ' + df_4h['Time'])
    df_4h = df_4h.set_index('Datetime')
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'TotalVolume': 'sum'
    }
    df_4h = df_4h.resample('4H').agg(agg_dict).dropna().reset_index()
    return df_4h

def compute_indicators(df, params, df_4h=None):
    """從1.py導入的指標計算函數"""
    # 布林通道
    df['BB_MID'] = df['Close'].rolling(params['bb_window']).mean()
    df['BB_STD'] = df['Close'].rolling(params['bb_window']).std()
    df['BB_UPPER'] = df['BB_MID'] + params['bb_std'] * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - params['bb_std'] * df['BB_STD']
    
    # RSI
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(params['rsi_period']).mean()
    avg_loss = pd.Series(loss).rolling(params['rsi_period']).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['TotalVolume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['TotalVolume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    df['OBV_MA'] = df['OBV'].rolling(params['obv_ma_window']).mean()
    
    # 4小時RSI（用於濾波）
    if df_4h is not None:
        delta = df_4h['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(params['rsi_period']).mean()
        avg_loss = pd.Series(loss).rolling(params['rsi_period']).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df_4h['RSI_4H'] = 100 - (100 / (1 + rs))
        # 對齊到15分鐘主圖
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df_4h = df_4h.set_index('Datetime')
        df['RSI_4H'] = df['Datetime'].map(df_4h['RSI_4H'])
    
    return df

def generate_entry_signal(df, params):
    """從1.py導入的進場訊號函數"""
    cond1 = df['Close'] < df['BB_LOWER']
    cond2 = df['RSI'] < params['rsi_oversold']
    cond3 = df['OBV'] > df['OBV_MA']
    # 4小時RSI濾波
    cond4 = df['RSI_4H'] > 50  # 4小時RSI>50才允許進場
    entry = (cond1.astype(int) + cond2.astype(int) + cond3.astype(int)) >= params['entry_n']
    df['EntrySignal'] = entry & cond4
    return df

def generate_exit_signal(df, params):
    """從1.py導入的出場訊號函數"""
    cond1 = df['Close'] > df['BB_MID']
    cond2 = df['RSI'] > params['rsi_exit']
    exit_signal = cond1 & cond2
    df['ExitSignal'] = exit_signal
    return df

def generate_trades_from_signals(df):
    """從2.py導入的交易紀錄產生函數"""
    trades = []
    position = 0
    entry_idx = None
    for i, row in df.iterrows():
        if position == 0 and row.get('EntrySignal', False):
            position = 1
            entry_idx = i
        elif position == 1 and row.get('ExitSignal', False):
            trade = {
                'EntryTime': df.loc[entry_idx, 'Date'] + ' ' + df.loc[entry_idx, 'Time'],
                'ExitTime': row['Date'] + ' ' + row['Time'],
                'EntryPrice': df.loc[entry_idx, 'Close'],
                'ExitPrice': row['Close'],
                'Direction': 1  # 只做多
            }
            trades.append(trade)
            position = 0
            entry_idx = None
    return pd.DataFrame(trades)

# =====================
# 穩健性測試模組
# =====================

def walk_forward_test(df, params, n_splits=5, train_size=0.7):
    """
    Walk-forward 訓練/測試分段
    """
    print("=== Walk-Forward 測試 ===")
    
    # 時間序列分割
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        # 分割訓練和測試資料
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        # 在訓練資料上計算指標
        train_df_4h = resample_to_4h(train_df)
        train_df = compute_indicators(train_df, params, train_df_4h)
        
        # 在測試資料上計算指標（使用訓練資料的參數）
        test_df_4h = resample_to_4h(test_df)
        test_df = compute_indicators(test_df, params, test_df_4h)
        
        # 產生訊號
        train_df = generate_entry_signal(train_df, params)
        train_df = generate_exit_signal(train_df, params)
        test_df = generate_entry_signal(test_df, params)
        test_df = generate_exit_signal(test_df, params)
        
        # 產生交易紀錄
        train_trades = generate_trades_from_signals(train_df)
        test_trades = generate_trades_from_signals(test_df)
        
        # 計算績效
        if len(train_trades) > 0:
            train_result = apply_cost_model(train_trades, fee=1.5, slippage_long=1.0, slippage_short=2.0)
            train_pnl = train_result['NetPnL'].sum()
        else:
            train_pnl = 0
            
        if len(test_trades) > 0:
            test_result = apply_cost_model(test_trades, fee=1.5, slippage_long=1.0, slippage_short=2.0)
            test_pnl = test_result['NetPnL'].sum()
        else:
            test_pnl = 0
        
        results.append({
            'fold': fold + 1,
            'train_pnl': train_pnl,
            'test_pnl': test_pnl,
            'train_trades': len(train_trades),
            'test_trades': len(test_trades)
        })
    
    results_df = pd.DataFrame(results)
    print("Walk-Forward 測試結果：")
    print(results_df)
    print(f"平均測試報酬: {results_df['test_pnl'].mean():.2f}")
    print(f"測試報酬標準差: {results_df['test_pnl'].std():.2f}")
    
    return results_df

def data_shuffle_test(df, params, n_shuffles=5):
    """
    資料洗牌 / 時間打亂測試
    """
    print("\n=== 資料洗牌測試 ===")
    
    results = []
    
    for i in range(n_shuffles):
        print(f"洗牌測試 {i + 1}/{n_shuffles}")
        
        # 隨機打亂資料順序
        shuffled_df = df.sample(frac=1, random_state=i).reset_index(drop=True)
        
        # 重新排序時間（保持時間連續性）
        shuffled_df['Datetime'] = pd.to_datetime(shuffled_df['Date'] + ' ' + shuffled_df['Time'])
        shuffled_df = shuffled_df.sort_values('Datetime').reset_index(drop=True)
        
        # 計算指標和訊號
        shuffled_df_4h = resample_to_4h(shuffled_df)
        shuffled_df = compute_indicators(shuffled_df, params, shuffled_df_4h)
        shuffled_df = generate_entry_signal(shuffled_df, params)
        shuffled_df = generate_exit_signal(shuffled_df, params)
        
        # 產生交易紀錄
        trades = generate_trades_from_signals(shuffled_df)
        
        # 計算績效
        if len(trades) > 0:
            result = apply_cost_model(trades, fee=1.5, slippage_long=1.0, slippage_short=2.0)
            total_pnl = result['NetPnL'].sum()
        else:
            total_pnl = 0
        
        results.append({
            'shuffle': i + 1,
            'total_pnl': total_pnl,
            'trades_count': len(trades)
        })
    
    results_df = pd.DataFrame(results)
    print("資料洗牌測試結果：")
    print(results_df)
    print(f"平均報酬: {results_df['total_pnl'].mean():.2f}")
    print(f"報酬標準差: {results_df['total_pnl'].std():.2f}")
    
    return results_df

def noise_injection_test(df, params, noise_levels=[0.001, 0.002, 0.005]):
    """
    噪聲注入測試（價格加 ±0.2%）
    """
    print("\n=== 噪聲注入測試 ===")
    
    results = []
    
    for noise_level in noise_levels:
        print(f"噪聲水平: ±{noise_level*100:.1f}%")
        
        # 注入噪聲
        noise = np.random.normal(0, noise_level, len(df))
        noisy_df = df.copy()
        noisy_df['Close'] = noisy_df['Close'] * (1 + noise)
        noisy_df['Open'] = noisy_df['Open'] * (1 + noise)
        noisy_df['High'] = noisy_df['High'] * (1 + noise)
        noisy_df['Low'] = noisy_df['Low'] * (1 + noise)
        
        # 計算指標和訊號
        noisy_df_4h = resample_to_4h(noisy_df)
        noisy_df = compute_indicators(noisy_df, params, noisy_df_4h)
        noisy_df = generate_entry_signal(noisy_df, params)
        noisy_df = generate_exit_signal(noisy_df, params)
        
        # 產生交易紀錄
        trades = generate_trades_from_signals(noisy_df)
        
        # 計算績效
        if len(trades) > 0:
            result = apply_cost_model(trades, fee=1.5, slippage_long=1.0, slippage_short=2.0)
            total_pnl = result['NetPnL'].sum()
        else:
            total_pnl = 0
        
        results.append({
            'noise_level': noise_level,
            'noise_percent': f"±{noise_level*100:.1f}%",
            'total_pnl': total_pnl,
            'trades_count': len(trades)
        })
    
    results_df = pd.DataFrame(results)
    print("噪聲注入測試結果：")
    print(results_df)
    
    return results_df

def year_interval_test(df, params, year_ranges=None):
    """
    年份區間測試（2018–2022 漲跌期）
    """
    print("\n=== 年份區間測試 ===")
    
    if year_ranges is None:
        year_ranges = [
            (2020, 2021, "2020-2021 疫情期間"),
            (2021, 2022, "2021-2022 復甦期"),
            (2022, 2023, "2022-2023 調整期"),
            (2023, 2024, "2023-2024 穩定期")
        ]
    
    results = []
    
    for start_year, end_year, period_name in year_ranges:
        print(f"測試期間: {period_name}")
        
        # 篩選年份區間
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        period_df = df[(df['Year'] >= start_year) & (df['Year'] < end_year)].copy()
        
        if len(period_df) == 0:
            print(f"  警告: {period_name} 期間無資料")
            continue
        
        # 計算指標和訊號
        period_df_4h = resample_to_4h(period_df)
        period_df = compute_indicators(period_df, params, period_df_4h)
        period_df = generate_entry_signal(period_df, params)
        period_df = generate_exit_signal(period_df, params)
        
        # 產生交易紀錄
        trades = generate_trades_from_signals(period_df)
        
        # 計算績效
        if len(trades) > 0:
            result = apply_cost_model(trades, fee=1.5, slippage_long=1.0, slippage_short=2.0)
            total_pnl = result['NetPnL'].sum()
            win_rate = (result['NetPnL'] > 0).mean()
        else:
            total_pnl = 0
            win_rate = 0
        
        results.append({
            'period': period_name,
            'start_year': start_year,
            'end_year': end_year,
            'total_pnl': total_pnl,
            'trades_count': len(trades),
            'win_rate': win_rate,
            'data_points': len(period_df)
        })
    
    results_df = pd.DataFrame(results)
    print("年份區間測試結果：")
    print(results_df)
    
    return results_df

def comprehensive_robustness_test(df, params):
    """
    綜合穩健性測試
    """
    print("🚀 開始綜合穩健性測試...")
    
    # 1. Walk-forward 測試
    wf_results = walk_forward_test(df, params)
    
    # 2. 資料洗牌測試
    shuffle_results = data_shuffle_test(df, params)
    
    # 3. 噪聲注入測試
    noise_results = noise_injection_test(df, params)
    
    # 4. 年份區間測試
    year_results = year_interval_test(df, params)
    
    # 綜合報告
    print("\n" + "="*50)
    print("📊 穩健性測試綜合報告")
    print("="*50)
    
    print(f"\n1️⃣ Walk-Forward 測試:")
    print(f"   平均測試報酬: {wf_results['test_pnl'].mean():.2f}")
    print(f"   測試報酬標準差: {wf_results['test_pnl'].std():.2f}")
    print(f"   穩定性評分: {1 - (wf_results['test_pnl'].std() / abs(wf_results['test_pnl'].mean() + 1e-6)):.2f}")
    
    print(f"\n2️⃣ 資料洗牌測試:")
    print(f"   平均報酬: {shuffle_results['total_pnl'].mean():.2f}")
    print(f"   報酬標準差: {shuffle_results['total_pnl'].std():.2f}")
    print(f"   穩定性評分: {1 - (shuffle_results['total_pnl'].std() / abs(shuffle_results['total_pnl'].mean() + 1e-6)):.2f}")
    
    print(f"\n3️⃣ 噪聲注入測試:")
    print(f"   原始報酬: {noise_results.iloc[0]['total_pnl']:.2f}")
    print(f"   最大噪聲報酬: {noise_results.iloc[-1]['total_pnl']:.2f}")
    print(f"   抗噪性評分: {1 - abs(noise_results.iloc[-1]['total_pnl'] - noise_results.iloc[0]['total_pnl']) / abs(noise_results.iloc[0]['total_pnl'] + 1e-6):.2f}")
    
    print(f"\n4️⃣ 年份區間測試:")
    print(f"   最佳期間: {year_results.loc[year_results['total_pnl'].idxmax(), 'period']}")
    print(f"   最差期間: {year_results.loc[year_results['total_pnl'].idxmin(), 'period']}")
    print(f"   期間穩定性: {1 - (year_results['total_pnl'].std() / abs(year_results['total_pnl'].mean() + 1e-6)):.2f}")
    
    return {
        'walk_forward': wf_results,
        'shuffle': shuffle_results,
        'noise': noise_results,
        'year_interval': year_results
    }

# =====================
# 主流程
# =====================
def main():
    # 參數設定（從1.py導入）
    params = {
        'bb_window': 20,
        'bb_std': 2.0,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_exit': 50,
        'obv_ma_window': 10,
        'stop_loss': 20,  # 點數
        'entry_n': 3,     # 滿足幾個條件才進場
    }
    
    # 讀取資料
    print("📖 讀取台指期資料...")
    df = pd.read_csv('../scripts/TXF1_Minute_2020-01-01_2025-06-16.txt')
    
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
    
    print(f"資料處理完成，共 {len(df)} 筆資料")
    
    # 執行綜合穩健性測試
    results = comprehensive_robustness_test(df, params)
    
    print("\n✅ 穩健性測試完成！")
    
    return results

if __name__ == '__main__':
    main() 