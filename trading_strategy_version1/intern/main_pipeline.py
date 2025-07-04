"""
台指期策略完整流程主程式
串聯順序：4 → 5 → 6 → 7 → 1 → 2 → 3 → cost_model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
import sys
import os
from datetime import datetime, timedelta
import time

warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_palette("husl")

print("🚀 台指期策略完整流程開始執行...")
print("="*60)

# =====================
# 導入所有模組
# =====================
print("📦 導入模組中...")

# 導入成本模型
from cost_model import apply_cost_model, calculate_performance_metrics, print_performance_summary

# 導入策略相關函數（從1.py）
def resample_to_4h(df):
    """4小時重採樣函數"""
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
    """指標計算函數"""
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
    
    # 4小時RSI
    if df_4h is not None:
        delta = df_4h['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(params['rsi_period']).mean()
        avg_loss = pd.Series(loss).rolling(params['rsi_period']).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df_4h['RSI_4H'] = 100 - (100 / (1 + rs))
        
        # 將4小時RSI對應到15分鐘資料
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df_4h_temp = df_4h.copy()
        df_4h_temp = df_4h_temp.reset_index()
        df_4h_temp['Datetime'] = pd.to_datetime(df_4h_temp['Datetime'])
        
        # 使用前向填充將4小時RSI對應到15分鐘資料
        df['RSI_4H'] = np.nan
        for i, row in df.iterrows():
            current_time = pd.to_datetime(row['Date'] + ' ' + row['Time'])
            # 找到對應的4小時K線
            matching_4h = df_4h_temp[df_4h_temp['Datetime'] <= current_time]
            if len(matching_4h) > 0:
                df.loc[i, 'RSI_4H'] = matching_4h.iloc[-1]['RSI_4H']
        
        # 如果還有NaN值，使用前向填充
        df['RSI_4H'] = df['RSI_4H'].fillna(method='ffill')
    else:
        # 如果沒有4小時資料，使用15分鐘RSI作為替代
        df['RSI_4H'] = df['RSI']
    
    # ===== 新增指標 =====
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # 均線
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()

    # K線型態
    # 吞噬形態（多頭吞噬/空頭吞噬）
    df['Bull_Engulfing'] = (
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Close'] > df['Open']) &
        (df['Open'] < df['Close'].shift(1)) &
        (df['Close'] > df['Open'].shift(1))
    )
    df['Bear_Engulfing'] = (
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Close'] < df['Open']) &
        (df['Open'] > df['Close'].shift(1)) &
        (df['Close'] < df['Open'].shift(1))
    )
    # 十字星
    df['Doji'] = (abs(df['Close'] - df['Open']) <= (df['High'] - df['Low']) * 0.1)
    # 錘頭
    df['Hammer'] = (
        (df['High'] - df['Low'] > 3 * (df['Open'] - df['Close'])) &
        ((df['Close'] - df['Low']) / (0.001 + df['High'] - df['Low']) > 0.6) &
        ((df['Open'] - df['Low']) / (0.001 + df['High'] - df['Low']) > 0.6)
    )
    # 墓碑線
    df['Gravestone'] = (
        (df['High'] - df['Low'] > 3 * (df['Open'] - df['Close'])) &
        ((df['High'] - df['Close']) / (0.001 + df['High'] - df['Low']) > 0.6) &
        ((df['High'] - df['Open']) / (0.001 + df['High'] - df['Low']) > 0.6)
    )
    # ====================

    return df

def generate_entry_signal(df, params):
    """多空進場訊號函數（K棒型態可選/必須）"""
    # 多單進場條件
    cond1_long = df['Close'] <= df['BB_LOWER']
    cond2_long = df['RSI'] < params['rsi_oversold']
    cond3_long = df['Hammer'] | df['Bull_Engulfing']
    
    # 四小時方向判斷（多頭）- 放寬條件
    df['4H_Trend_Long'] = (df['RSI_4H'] < 60)  # 四小時RSI低於60即可
    
    if params.get('require_reversal_kbar', False):
        entry_long = cond1_long & cond2_long & cond3_long & df['4H_Trend_Long']
    else:
        entry_long = cond1_long & cond2_long & df['4H_Trend_Long']
    
    # 空單進場條件
    cond1_short = df['Close'] >= df['BB_UPPER']
    cond2_short = df['RSI'] > params['rsi_overbought']
    cond3_short = df['Gravestone'] | df['Bear_Engulfing']
    
    # 四小時方向判斷（空頭）- 放寬條件
    df['4H_Trend_Short'] = (df['RSI_4H'] > 40)  # 四小時RSI高於40即可
    
    if params.get('require_reversal_kbar', False):
        entry_short = cond1_short & cond2_short & cond3_short & df['4H_Trend_Short']
    else:
        entry_short = cond1_short & cond2_short & df['4H_Trend_Short']
    
    # 記錄條件
    df['BB_Condition_Long'] = cond1_long
    df['RSI_Condition_Long'] = cond2_long
    df['OBV_Condition_Long'] = df['OBV'] > df['OBV_MA']
    df['MACD_Condition_Long'] = df['MACD'] > df['MACD_signal']
    df['MA_Condition_Long'] = df['MA5'] > df['MA20']
    df['Bull_Engulfing_Long'] = df['Bull_Engulfing']
    df['Hammer_Long'] = df['Hammer']
    df['Doji_Long'] = df['Doji']
    df['4H_Trend_Condition_Long'] = df['4H_Trend_Long']
    df['Conditions_Met_Long'] = (cond1_long.astype(int) + cond2_long.astype(int) + (df['OBV'] > df['OBV_MA']).astype(int) + (df['MACD'] > df['MACD_signal']).astype(int) + (df['MA5'] > df['MA20']).astype(int) + (df['Bull_Engulfing']).astype(int) + (df['Hammer']).astype(int) + (df['Doji']).astype(int) + df['4H_Trend_Long'].astype(int))
    df['EntrySignal_Long'] = entry_long

    df['BB_Condition_Short'] = cond1_short
    df['RSI_Condition_Short'] = cond2_short
    df['OBV_Condition_Short'] = df['OBV'] < df['OBV_MA']
    df['MACD_Condition_Short'] = df['MACD'] < df['MACD_signal']
    df['MA_Condition_Short'] = df['MA5'] < df['MA20']
    df['Bear_Engulfing_Short'] = df['Bear_Engulfing']
    df['Gravestone_Short'] = df['Gravestone']
    df['Doji_Short'] = df['Doji']
    df['4H_Trend_Condition_Short'] = df['4H_Trend_Short']
    df['Conditions_Met_Short'] = (cond1_short.astype(int) + cond2_short.astype(int) + (df['OBV'] < df['OBV_MA']).astype(int) + (df['MACD'] < df['MACD_signal']).astype(int) + (df['MA5'] < df['MA20']).astype(int) + (df['Bear_Engulfing']).astype(int) + (df['Gravestone']).astype(int) + (df['Doji']).astype(int) + df['4H_Trend_Short'].astype(int))
    df['EntrySignal_Short'] = entry_short

    return df

def generate_exit_signal(df, params):
    """多空出場訊號函數"""
    # 多單出場
    cond1_long = df['Close'] > df['BB_MID']
    cond2_long = df['RSI'] > params['rsi_exit']
    cond3_long = df['MACD'] < df['MACD_signal']  # MACD死叉
    cond4_long = df['MA5'] < df['MA20']          # 均線空頭排列
    cond5_long = df['Bear_Engulfing']           # 空頭吞噬
    cond6_long = df['Gravestone']               # 墓碑線
    cond7_long = df['Doji']                     # 十字星
    exit_signal_long = (cond1_long & cond2_long) | cond3_long | cond4_long | cond5_long | cond6_long | cond7_long
    df['ExitSignal_Long'] = exit_signal_long
    # 空單出場
    cond1_short = df['Close'] < df['BB_MID']
    cond2_short = df['RSI'] < params['rsi_exit_short']
    cond3_short = df['MACD'] > df['MACD_signal']  # MACD金叉
    cond4_short = df['MA5'] > df['MA20']          # 均線多頭排列
    cond5_short = df['Bull_Engulfing']           # 多頭吞噬
    cond6_short = df['Hammer']                   # 錘頭
    cond7_short = df['Doji']                     # 十字星
    exit_signal_short = (cond1_short & cond2_short) | cond3_short | cond4_short | cond5_short | cond6_short | cond7_short
    df['ExitSignal_Short'] = exit_signal_short
    return df

def generate_trades_from_signals(df, params):
    """多空交易紀錄產生函數，動態計算止損於前低/高下0.3%"""
    trades = []
    position = 0
    entry_idx = None
    entry_price = None
    entry_direction = None
    initial_capital = 1000000
    current_capital = initial_capital
    stop_loss_pct = params.get('stop_loss_pct', 0.003)
    max_hold_bars = params.get('max_hold_bars', 5)
    for i, row in df.iterrows():
        if position == 0:
            if row.get('EntrySignal_Long', False):
                position = 1
                entry_idx = i
                entry_price = row['Close']
                entry_direction = 1
                # 前低
                prev_low = df['Low'].iloc[max(0, i-1)]
                stop_loss_price = prev_low * (1 - stop_loss_pct)
                entry_indicators = {
                    'BB': row.get('BB_Condition_Long', False),
                    'RSI': row.get('RSI_Condition_Long', False),
                    'Hammer': row.get('Hammer', False),
                    'Bull_Engulfing': row.get('Bull_Engulfing', False),
                    '4H_Trend': row.get('4H_Trend_Condition_Long', False)
                }
            elif row.get('EntrySignal_Short', False):
                position = -1
                entry_idx = i
                entry_price = row['Close']
                entry_direction = -1
                # 前高
                prev_high = df['High'].iloc[max(0, i-1)]
                stop_loss_price = prev_high * (1 + stop_loss_pct)
                entry_indicators = {
                    'BB': row.get('BB_Condition_Short', False),
                    'RSI': row.get('RSI_Condition_Short', False),
                    'Gravestone': row.get('Gravestone', False),
                    'Bear_Engulfing': row.get('Bear_Engulfing', False),
                    '4H_Trend': row.get('4H_Trend_Condition_Short', False)
                }
        elif position == 1:
            exit_reason = None
            exit_price = None
            # 第一目標：BB_MID
            if row['Close'] >= row['BB_MID']:
                exit_reason = 'BB_MID出場(多)'
                exit_price = row['Close']
            # 持有超過max_hold_bars
            elif entry_idx is not None and i - entry_idx >= max_hold_bars:
                exit_reason = '持有超過max_hold_bars(多)'
                exit_price = row['Close']
            # 停損
            elif row['Close'] <= stop_loss_price:
                exit_reason = '停損出場(多)'
                exit_price = stop_loss_price
            if exit_reason and exit_price is not None:
                pnl = (exit_price - entry_price) * 1
                current_capital += pnl * 200
                trade = {
                    'EntryTime': df.loc[entry_idx, 'Date'] + ' ' + df.loc[entry_idx, 'Time'],
                    'ExitTime': row['Date'] + ' ' + row['Time'],
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'Direction': 1,
                    'ExitReason': exit_reason,
                    'StopLossPrice': stop_loss_price,
                    'PnL': pnl,
                    'CurrentCapital': current_capital,
                    'EntryIndicators': entry_indicators
                }
                trades.append(trade)
                position = 0
                entry_idx = None
                entry_price = None
                entry_direction = None
        elif position == -1:
            exit_reason = None
            exit_price = None
            # 第一目標：BB_MID
            if row['Close'] <= row['BB_MID']:
                exit_reason = 'BB_MID出場(空)'
                exit_price = row['Close']
            # 持有超過max_hold_bars
            elif entry_idx is not None and i - entry_idx >= max_hold_bars:
                exit_reason = '持有超過max_hold_bars(空)'
                exit_price = row['Close']
            # 停損
            elif row['Close'] >= stop_loss_price:
                exit_reason = '停損出場(空)'
                exit_price = stop_loss_price
            if exit_reason and exit_price is not None:
                pnl = (entry_price - exit_price) * 1
                current_capital += pnl * 200
                trade = {
                    'EntryTime': df.loc[entry_idx, 'Date'] + ' ' + df.loc[entry_idx, 'Time'],
                    'ExitTime': row['Date'] + ' ' + row['Time'],
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'Direction': -1,
                    'ExitReason': exit_reason,
                    'StopLossPrice': stop_loss_price,
                    'PnL': pnl,
                    'CurrentCapital': current_capital,
                    'EntryIndicators': entry_indicators
                }
                trades.append(trade)
                position = 0
                entry_idx = None
                entry_price = None
                entry_direction = None
    return pd.DataFrame(trades)

# =====================
# 步驟4：資料預處理
# =====================
def step4_data_preprocessing():
    """步驟4：資料預處理"""
    print("\n🔧 步驟4：資料預處理")
    print("-" * 40)
    
    # 讀取原始資料
    print("📖 讀取台指期資料...")
    df = pd.read_csv('../scripts/TXF1_Minute_2020-01-01_2025-06-16.txt')
    print(f"原始資料筆數: {len(df)}")
    
    # 資料清理
    print("🧹 資料清理中...")
    df = df.dropna()
    df = df[df['Close'] > 0]  # 移除異常價格
    
    # 轉換為15分鐘K
    print("⏰ 轉換為15分鐘K線...")
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
    
    print(f"處理後資料筆數: {len(df)}")
    print("✅ 資料預處理完成")
    
    return df

# =====================
# 步驟5：特徵工程
# =====================
def step5_feature_engineering(df):
    """步驟5：特徵工程"""
    print("\n🔬 步驟5：特徵工程")
    print("-" * 40)
    
    print("📊 計算技術指標...")
    
    # 基本價格特徵
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5'] = df['Close'].pct_change(5)
    df['Price_Change_10'] = df['Close'].pct_change(10)
    
    # 波動率特徵
    df['Volatility_5'] = df['Price_Change'].rolling(5).std()
    df['Volatility_10'] = df['Price_Change'].rolling(10).std()
    
    # 成交量特徵
    df['Volume_MA_5'] = df['TotalVolume'].rolling(5).mean()
    df['Volume_MA_10'] = df['TotalVolume'].rolling(10).mean()
    df['Volume_Ratio'] = df['TotalVolume'] / df['Volume_MA_5']
    
    # 時間特徵
    df['Hour'] = pd.to_datetime(df['Time']).dt.hour
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['IsMorning'] = (df['Hour'] >= 9) & (df['Hour'] <= 11)
    df['IsAfternoon'] = (df['Hour'] >= 13) & (df['Hour'] <= 14)
    
    print("✅ 特徵工程完成")
    print(f"新增特徵數量: {len(df.columns) - 8}")  # 減去原始8個欄位
    
    return df

# =====================
# 步驟6：策略優化
# =====================
def step6_strategy_optimization(df):
    """步驟6：策略優化"""
    print("\n⚡ 步驟6：策略優化")
    print("-" * 40)
    
    print("🔍 參數優化中...")
    
    # 定義參數範圍
    param_ranges = {
        'bb_window': [20],
        'bb_std': [2.5],
        'rsi_period': [14],
        'rsi_oversold': [30],
        'rsi_overbought': [70],
        'obv_ma_window': [10],
        'stop_loss_pct': [0.003],
        'require_reversal_kbar': [True, False],
        'max_hold_bars': [5],
        'entry_n': [2]
    }
    
    # 簡單的參數優化（這裡只測試幾個組合）
    best_params = {
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
    
    print("✅ 策略優化完成")
    print(f"最佳參數: {best_params}")
    
    return best_params

def optimize_for_win_rate(df, param_ranges=None):
    """優化策略參數以提升勝率，同時找出總收益最高的組合"""
    print("\n🎯 策略勝率與總收益優化分析")
    print("-" * 50)
    
    if param_ranges is None:
        param_ranges = {
            'bb_window': [20],
            'bb_std': [2.5],
            'rsi_period': [14],
            'rsi_oversold': [30],
            'rsi_overbought': [70],
            'obv_ma_window': [10],
            'stop_loss_pct': [0.003],
            'require_reversal_kbar': [True, False],
            'max_hold_bars': [5],
            'entry_n': [2]
        }
    
    best_win_rate = 0
    best_params = None
    best_metrics = None
    best_pnl = -np.inf
    best_pnl_params = None
    best_pnl_metrics = None
    results_list = []
    
    # 測試參數組合
    total_combinations = 1
    for param, values in param_ranges.items():
        total_combinations *= len(values)
    
    print(f"🔍 測試 {total_combinations} 種參數組合...")
    
    # 簡化測試：只測試關鍵參數組合
    test_combinations = [
        {'bb_window': 20, 'bb_std': 2.5, 'rsi_period': 14, 'rsi_oversold': 30, 'rsi_exit': 50, 'rsi_overbought': 70, 'rsi_exit_short': 50, 'obv_ma_window': 10, 'stop_loss': 20, 'stop_loss_pct': 0.003, 'require_reversal_kbar': True, 'max_hold_bars': 5, 'entry_n': 2},
        {'bb_window': 20, 'bb_std': 2.5, 'rsi_period': 14, 'rsi_oversold': 30, 'rsi_exit': 50, 'rsi_overbought': 70, 'rsi_exit_short': 50, 'obv_ma_window': 10, 'stop_loss': 20, 'stop_loss_pct': 0.003, 'require_reversal_kbar': False, 'max_hold_bars': 5, 'entry_n': 2},
    ]
    
    for i, params in enumerate(test_combinations):
        print(f"測試組合 {i+1}/{len(test_combinations)}: {params}")
        
        try:
            # 計算指標
            df_4h = resample_to_4h(df)
            df_test = compute_indicators(df.copy(), params, df_4h)
            df_test = generate_entry_signal(df_test, params)
            df_test = generate_exit_signal(df_test, params)
            
            # 產生交易
            trades = generate_trades_from_signals(df_test, params)
            
            if len(trades) > 10:
                result = apply_cost_model(trades, fee=1.5, slippage_long=1.0, slippage_short=2.0)
                win_rate = (result['NetPnL'] > 0).mean()
                total_pnl = result['NetPnL'].sum()
                avg_pnl = result['NetPnL'].mean()
                
                results_list.append({
                    'params': params.copy(),
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_pnl': avg_pnl,
                    'trade_count': len(trades)
                })
                
                print(f"  勝率: {win_rate:.2%}, 總損益: {total_pnl:.2f}, 交易次數: {len(trades)}")
                
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_params = params.copy()
                    best_metrics = {
                        'win_rate': win_rate,
                        'total_pnl': total_pnl,
                        'avg_pnl': avg_pnl,
                        'trade_count': len(trades)
                    }
                if total_pnl > best_pnl:
                    best_pnl = total_pnl
                    best_pnl_params = params.copy()
                    best_pnl_metrics = {
                        'win_rate': win_rate,
                        'total_pnl': total_pnl,
                        'avg_pnl': avg_pnl,
                        'trade_count': len(trades)
                    }
            else:
                print(f"  交易次數不足: {len(trades)}")
        except Exception as e:
            print(f"  錯誤: {str(e)}")
    
    # 顯示最佳結果
    if best_params and best_metrics:
        print(f"\n🏆 勝率最高參數組合:")
        print(f"勝率: {best_win_rate:.2%}")
        print(f"總損益: {best_metrics['total_pnl']:.2f}")
        print(f"平均損益: {best_metrics['avg_pnl']:.2f}")
        print(f"交易次數: {best_metrics['trade_count']}")
        print(f"參數: {best_params}")
    if best_pnl_params and best_pnl_metrics:
        print(f"\n💰 總收益最高參數組合:")
        print(f"勝率: {best_pnl_metrics['win_rate']:.2%}")
        print(f"總損益: {best_pnl_metrics['total_pnl']:.2f}")
        print(f"平均損益: {best_pnl_metrics['avg_pnl']:.2f}")
        print(f"交易次數: {best_pnl_metrics['trade_count']}")
        print(f"參數: {best_pnl_params}")
    
    # 顯示前5名結果
    if results_list:
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('win_rate', ascending=False)
        print(f"\n📊 前5名勝率參數組合:")
        for i, (idx, row) in enumerate(results_df.head().iterrows(), 1):
            print(f"{i}. 勝率: {row['win_rate']:.2%}, 總損益: {row['total_pnl']:.2f}, 交易次數: {row['trade_count']}")
            print(f"   參數: {row['params']}")
        results_df = results_df.sort_values('total_pnl', ascending=False)
        print(f"\n📊 前5名總收益參數組合:")
        for i, (idx, row) in enumerate(results_df.head().iterrows(), 1):
            print(f"{i}. 總損益: {row['total_pnl']:.2f}, 勝率: {row['win_rate']:.2%}, 交易次數: {row['trade_count']}")
            print(f"   參數: {row['params']}")
    else:
        results_df = pd.DataFrame()
    
    return best_params, best_metrics, best_pnl_params, best_pnl_metrics, results_df

# =====================
# 步驟7：風險管理
# =====================
def step7_risk_management(df, params):
    """步驟7：風險管理"""
    print("\n🛡️ 步驟7：風險管理")
    print("-" * 40)
    
    print("📈 計算風險指標...")
    
    # 計算指標
    df_4h = resample_to_4h(df)
    df = compute_indicators(df, params, df_4h)
    
    # 產生訊號
    df = generate_entry_signal(df, params)
    df = generate_exit_signal(df, params)
    
    # 風險控制
    df['Position_Size'] = 1.0  # 固定部位大小
    df['Max_Loss'] = params['stop_loss']  # 最大停損
    
    # 計算風險指標
    trades = generate_trades_from_signals(df, params)
    if len(trades) > 0:
        result = apply_cost_model(trades, fee=1.5, slippage_long=1.0, slippage_short=2.0)
        metrics = calculate_performance_metrics(result)
        
        print(f"總交易次數: {metrics['total_trades']}")
        print(f"勝率: {metrics['win_rate']:.2%}")
        print(f"最大回撤: {metrics['max_drawdown']:.2f}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    else:
        print("無交易紀錄")
    
    print("✅ 風險管理完成")
    
    return df

# =====================
# 步驟1：策略執行
# =====================
def step1_strategy_execution(df, params):
    """步驟1：策略執行"""
    print("\n🎯 步驟1：策略執行")
    print("-" * 40)
    
    print("🚀 執行策略...")
    
    # 計算指標和訊號
    df_4h = resample_to_4h(df)
    df = compute_indicators(df, params, df_4h)
    df = generate_entry_signal(df, params)
    df = generate_exit_signal(df, params)
    
    # 顯示訊號統計
    entry_signals = df['EntrySignal_Long'].sum() + df['EntrySignal_Short'].sum()
    exit_signals = df['ExitSignal_Long'].sum() + df['ExitSignal_Short'].sum()
    
    print(f"進場訊號數量: {entry_signals}")
    print(f"出場訊號數量: {exit_signals}")
    
    # 顯示進場條件分析
    if entry_signals > 0:
        print(f"\n📊 進場條件分析:")
        print(f"需要滿足條件數: {params['entry_n']} 個或以上")
        print(f"BB條件滿足次數: {df['BB_Condition_Long'].sum() + df['BB_Condition_Short'].sum()}")
        print(f"RSI條件滿足次數: {df['RSI_Condition_Long'].sum() + df['RSI_Condition_Short'].sum()}")
        print(f"OBV條件滿足次數: {df['OBV_Condition_Long'].sum() + df['OBV_Condition_Short'].sum()}")
        print(f"四小時趨勢條件滿足次數: {df['4H_Trend_Condition_Long'].sum() + df['4H_Trend_Condition_Short'].sum()}")
        
        # 顯示條件組合統計
        condition_stats = df['Conditions_Met_Long'].value_counts().sort_index()
        print(f"\n條件組合統計:")
        for conditions, count in condition_stats.items():
            print(f"  滿足 {conditions} 個條件: {count} 次")
    
    # 顯示最新訊號
    recent_signals = df[['Date', 'Time', 'Close', 'RSI', 'RSI_4H', 'BB_Condition_Long', 'RSI_Condition_Long', 'OBV_Condition_Long', '4H_Trend_Condition_Long', 'Conditions_Met_Long', 'EntrySignal_Long', 'ExitSignal_Long', 'BB_Condition_Short', 'RSI_Condition_Short', 'OBV_Condition_Short', '4H_Trend_Condition_Short', 'Conditions_Met_Short', 'EntrySignal_Short', 'ExitSignal_Short']].tail(10)
    print("\n最新10筆訊號:")
    print(recent_signals)
    
    print("✅ 策略執行完成")
    
    return df

# =====================
# 步驟2：交易執行
# =====================
def step2_trade_execution(df, params):
    """步驟2：交易執行"""
    print("\n💼 步驟2：交易執行")
    print("-" * 40)
    print("📋 產生交易紀錄...")
    # 產生交易紀錄
    trades = generate_trades_from_signals(df, params)
    if len(trades) > 0:
        print(f"產生 {len(trades)} 筆交易")
        print("\n前5筆交易:")
        print(trades[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'Direction']].head())
    else:
        print("無交易紀錄")
    print("✅ 交易執行完成")
    return trades

# =====================
# 步驟3：績效評估
# =====================
def step3_performance_evaluation(trades):
    """步驟3：績效評估"""
    print("\n📊 步驟3：績效評估")
    print("-" * 40)
    
    if len(trades) == 0:
        print("無交易紀錄，跳過績效評估")
        return None
    
    print("📈 計算績效指標...")
    
    # 套用成本模型
    result = apply_cost_model(trades, fee=1.5, slippage_long=1.0, slippage_short=2.0)
    
    # 顯示績效摘要
    print_performance_summary(result)
    
    print("✅ 績效評估完成")
    
    return result

# =====================
# Excel輸出功能
# =====================
def export_to_excel(results, filename='strategy_results.xlsx'):
    """將結果輸出到Excel檔案"""
    print(f"\n📊 輸出結果到Excel檔案: {filename}")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # 1. 策略摘要
        if results['result'] is not None:
            summary_data = {
                '指標': ['總交易次數', '勝率', '總損益(點)', '平均損益(點)', 
                        '最大回撤(點)', '夏普比率', '執行時間(秒)'],
                '數值': [
                    len(results['trades']),
                    f"{results['result']['NetPnL'].gt(0).mean():.2%}",
                    f"{results['result']['NetPnL'].sum():.2f}",
                    f"{results['result']['NetPnL'].mean():.2f}",
                    f"{results['result']['NetPnL'].cumsum().min():.2f}",
                    f"{results['result']['NetPnL'].mean() / results['result']['NetPnL'].std() if results['result']['NetPnL'].std() > 0 else 0:.2f}",
                    f"{results['execution_time']:.2f}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='策略摘要', index=False)
        
        # 2. 詳細交易紀錄（新增欄位）
        if len(results['trades']) > 0:
            detailed_trades = results['result'].copy()
            
            # 添加新欄位
            detailed_trades['交易編號'] = range(1, len(detailed_trades) + 1)
            
            # 日期和時間處理
            detailed_trades['進場日期'] = pd.to_datetime(detailed_trades['EntryTime']).dt.date
            detailed_trades['出場日期'] = pd.to_datetime(detailed_trades['ExitTime']).dt.date
            detailed_trades['進場時間'] = pd.to_datetime(detailed_trades['EntryTime']).dt.time
            detailed_trades['出場時間'] = pd.to_datetime(detailed_trades['ExitTime']).dt.time
            
            # 方向標示
            detailed_trades['方向'] = detailed_trades['Direction'].map({1: '做多', -1: '做空'})
            
            # Time Frame
            detailed_trades['Time_Frame'] = '15分鐘'
            
            # 計算持倉時間
            detailed_trades['持倉時間(小時)'] = (
                pd.to_datetime(detailed_trades['ExitTime']) - 
                pd.to_datetime(detailed_trades['EntryTime'])
            ).dt.total_seconds() / 3600
            
            # Target RR (假設目標風險報酬比為1:2)
            detailed_trades['Target_RR'] = 2.0
            
            # Achieved RR (實際風險報酬比)
            detailed_trades['Achieved_RR'] = detailed_trades['NetPnL'].abs() / 5.0  # 假設每筆交易成本5點
            
            # Profit/Loss USD (假設1點=200台幣，匯率30)
            detailed_trades['Profit_Loss_USD'] = detailed_trades['NetPnL'] * 200 / 30
            
            # 根據實際進場條件顯示使用的指標
            def get_used_indicators(row):
                indicators = ['BB', 'RSI', '4H趨勢']
                if row.get('Hammer', False) or row.get('Bull_Engulfing', False) or row.get('Gravestone', False) or row.get('Bear_Engulfing', False):
                    indicators.append('K線型態')
                return '+'.join(indicators)
            
            detailed_trades['使用指標'] = detailed_trades.apply(get_used_indicators, axis=1)
            
            # 勝負標示
            detailed_trades['勝負'] = detailed_trades['NetPnL'].apply(lambda x: '獲利' if x > 0 else '虧損')
            detailed_trades['損益金額(USD)'] = detailed_trades['Profit_Loss_USD'].apply(
                lambda x: f"${x:.2f}" if x > 0 else f"-${abs(x):.2f}"
            )
            
            # 展開 EntryIndicators
            indicator_cols = ['BB', 'RSI', 'OBV', 'MACD', 'MA', 'Bull_Engulfing', 'Bear_Engulfing', 'Hammer', 'Gravestone', 'Doji', '4H_Trend']
            for col in indicator_cols:
                detailed_trades[col] = detailed_trades['EntryIndicators'].apply(lambda d: d.get(col, False) if isinstance(d, dict) else False)
            # 重新排列欄位
            detailed_trades = detailed_trades[[
                '交易編號', '進場日期', '進場時間', '出場日期', '出場時間', 
                '方向', 'Time_Frame', 'EntryPrice', 'ExitPrice', 
                'Target_RR', 'Achieved_RR', 'GrossPnL', 'TotalCost', 'NetPnL',
                'Profit_Loss_USD', '損益金額(USD)', '使用指標', '勝負', '持倉時間(小時)'
            ] + indicator_cols]
            
            detailed_trades.to_excel(writer, sheet_name='詳細交易紀錄', index=False)
        
        # 3. 指標勝率統計
        if len(results['trades']) > 0:
            # 按指標分組統計
            indicator_cols = ['BB', 'RSI', 'OBV', 'MACD', 'MA', 'Bull_Engulfing', 'Bear_Engulfing', 'Hammer', 'Gravestone', 'Doji', '4H_Trend']
            indicator_stats = {
                '指標名稱': [],
                '使用次數': [],
                '獲利次數': [],
                '勝率': [],
                '平均損益(點)': [],
                '總損益(點)': []
            }
            for col in indicator_cols:
                used = detailed_trades[detailed_trades[col] == True]
                indicator_stats['指標名稱'].append(col)
                indicator_stats['使用次數'].append(len(used))
                indicator_stats['獲利次數'].append((used['NetPnL'] > 0).sum())
                indicator_stats['勝率'].append(f"{(used['NetPnL'] > 0).mean():.2%}" if len(used) > 0 else '0.00%')
                indicator_stats['平均損益(點)'].append(f"{used['NetPnL'].mean():.2f}" if len(used) > 0 else '0.00')
                indicator_stats['總損益(點)'].append(f"{used['NetPnL'].sum():.2f}" if len(used) > 0 else '0.00')
            indicator_df = pd.DataFrame(indicator_stats)
            indicator_df.to_excel(writer, sheet_name='指標勝率統計', index=False)
        
        # 4. 策略參數
        params_list = list(results['params'].items())
        params_df = pd.DataFrame(params_list)
        params_df.columns = ['參數名稱', '參數值']
        params_df.to_excel(writer, sheet_name='策略參數', index=False)
        
        # 5. 訊號統計
        if 'data' in results and results['data'] is not None:
            signal_stats = {
                '統計項目': ['總資料筆數', '進場訊號數量', '出場訊號數量', '實際交易數量'],
                '數量': [
                    len(results['data']),
                    results['data']['EntrySignal_Long'].sum() + results['data']['EntrySignal_Short'].sum(),
                    results['data']['ExitSignal_Long'].sum() + results['data']['ExitSignal_Short'].sum(),
                    len(results['trades'])
                ]
            }
            signal_df = pd.DataFrame(signal_stats)
            signal_df.to_excel(writer, sheet_name='訊號統計', index=False)
        
        # 6. 月度績效分析
        if results['result'] is not None:
            monthly_performance = results['result'].copy()
            monthly_performance['EntryTime'] = pd.to_datetime(monthly_performance['EntryTime'])
            monthly_performance['年月'] = monthly_performance['EntryTime'].dt.to_period('M')
            
            monthly_stats = monthly_performance.groupby('年月').agg({
                'NetPnL': ['count', 'sum', 'mean'],
                'GrossPnL': 'sum',
                'TotalCost': 'sum'
            }).round(2)
            
            monthly_stats.columns = ['交易次數', '淨損益', '平均損益', '毛損益', '總成本']
            monthly_stats.to_excel(writer, sheet_name='月度績效')
        
        # 7. 風險分析
        if results['result'] is not None:
            risk_analysis = {
                '風險指標': [
                    '最大單筆虧損',
                    '最大單筆獲利', 
                    '平均獲利',
                    '平均虧損',
                    '獲利標準差',
                    '虧損標準差',
                    '最大連續虧損次數',
                    '最大連續獲利次數'
                ],
                '數值': [
                    f"{results['result']['NetPnL'].min():.2f}",
                    f"{results['result']['NetPnL'].max():.2f}",
                    f"{results['result']['NetPnL'][results['result']['NetPnL'] > 0].mean():.2f}",
                    f"{results['result']['NetPnL'][results['result']['NetPnL'] < 0].mean():.2f}",
                    f"{results['result']['NetPnL'][results['result']['NetPnL'] > 0].std():.2f}",
                    f"{results['result']['NetPnL'][results['result']['NetPnL'] < 0].std():.2f}",
                    f"{len(results['result']['NetPnL'][results['result']['NetPnL'] < 0].groupby((results['result']['NetPnL'] >= 0).cumsum()).max()):.0f}",
                    f"{len(results['result']['NetPnL'][results['result']['NetPnL'] > 0].groupby((results['result']['NetPnL'] <= 0).cumsum()).max()):.0f}"
                ]
            }
            risk_df = pd.DataFrame(risk_analysis)
            risk_df.to_excel(writer, sheet_name='風險分析', index=False)
        
        # 8. 交易統計摘要
        if results['result'] is not None:
            trade_summary = {
                '統計項目': [
                    '總交易次數',
                    '獲利交易次數',
                    '虧損交易次數',
                    '勝率',
                    '平均獲利(點)',
                    '平均虧損(點)',
                    '最大獲利(點)',
                    '最大虧損(點)',
                    '總損益(點)',
                    '總損益(USD)',
                    '平均持倉時間(小時)'
                ],
                '數值': [
                    len(results['result']),
                    (results['result']['NetPnL'] > 0).sum(),
                    (results['result']['NetPnL'] < 0).sum(),
                    f"{(results['result']['NetPnL'] > 0).mean():.2%}",
                    f"{results['result']['NetPnL'][results['result']['NetPnL'] > 0].mean():.2f}",
                    f"{results['result']['NetPnL'][results['result']['NetPnL'] < 0].mean():.2f}",
                    f"{results['result']['NetPnL'].max():.2f}",
                    f"{results['result']['NetPnL'].min():.2f}",
                    f"{results['result']['NetPnL'].sum():.2f}",
                    f"${results['result']['NetPnL'].sum() * 200 / 30:.2f}",
                    f"{(pd.to_datetime(results['result']['ExitTime']) - pd.to_datetime(results['result']['EntryTime'])).dt.total_seconds().mean() / 3600:.2f}"
                ]
            }
            summary_trade_df = pd.DataFrame(trade_summary)
            summary_trade_df.to_excel(writer, sheet_name='交易統計摘要', index=False)
    
    print(f"✅ Excel檔案已成功輸出: {filename}")
    print(f"📁 檔案位置: {os.path.abspath(filename)}")
    print(f"📊 包含工作表: 策略摘要、詳細交易紀錄、指標勝率統計、策略參數、訊號統計、月度績效、風險分析、交易統計摘要")

# =====================
# 主流程
# =====================
def main():
    """主流程：串聯所有步驟"""
    print("🎬 開始執行完整策略流程")
    print("順序：4 → 5 → 6 → 7 → 1 → 2 → 3 → cost_model")
    print("="*60)
    
    start_time = time.time()
    
    # 產生版本化的檔案名稱
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"7-2回測報表v1_{timestamp}.xlsx"
    
    try:
        # 步驟4：資料預處理
        df = step4_data_preprocessing()
        
        # 步驟5：特徵工程
        df = step5_feature_engineering(df)
        
        # 步驟6：策略優化
        params, best_metrics, best_pnl_params, best_pnl_metrics, results_df = optimize_for_win_rate(df)
        
        # 使用總收益最高的參數組合
        if best_pnl_params is not None:
            print(f"\n🎯 使用總收益最高參數組合執行策略:")
            print(f"總損益: {best_pnl_metrics['total_pnl']:.2f}")
            print(f"勝率: {best_pnl_metrics['win_rate']:.2%}")
            print(f"交易次數: {best_pnl_metrics['trade_count']}")
            params = best_pnl_params
        else:
            print("⚠️ 優化失敗，使用預設參數")
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
        
        # 步驟7：風險管理
        df = step7_risk_management(df, params)
        
        # 步驟1：策略執行
        df = step1_strategy_execution(df, params)
        
        # 步驟2：交易執行
        trades = step2_trade_execution(df, params)
        
        # 步驟3：績效評估
        result = step3_performance_evaluation(trades)
        
        # 成本模型（已在步驟3中執行）
        print("\n💰 成本模型已整合在績效評估中")
        
        execution_time = time.time() - start_time
        print(f"\n⏱️ 總執行時間: {execution_time:.2f} 秒")
        
        print("\n🎉 完整流程執行完成！")
        
        results = {
            'data': df,
            'trades': trades,
            'result': result,
            'params': params,
            'execution_time': execution_time
        }
        
        # 輸出到Excel（使用版本化檔名）
        export_to_excel(results, filename)
        
        return results
        
    except Exception as e:
        print(f"❌ 執行過程中發生錯誤: {str(e)}")
        return None

if __name__ == '__main__':
    results = main() 