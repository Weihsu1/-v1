"""
台指期策略 - Malaysian SNR策略
Malaysian SNR (Signal-to-Noise Ratio) 是一個基於波動率的技術指標
用於識別趨勢強度和反轉點
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

print("🚀 Malaysian SNR策略開始執行...")
print("="*60)

# =====================
# Malaysian SNR 指標計算
# =====================
def calculate_malaysian_snr(df, period=14):
    """
    計算Malaysian SNR指標
    
    Malaysian SNR = (Close - Close_n_periods_ago) / (Sum of absolute price changes)
    
    參數:
    - df: DataFrame with OHLC data
    - period: 計算週期，預設14
    
    返回:
    - SNR值，範圍通常在-1到1之間
    """
    # 計算價格變化
    price_change = df['Close'] - df['Close'].shift(period)
    
    # 計算絕對價格變化總和
    abs_changes = df['Close'].diff().abs()
    sum_abs_changes = abs_changes.rolling(window=period).sum()
    
    # 計算SNR
    snr = price_change / (sum_abs_changes + 1e-9)  # 避免除零
    
    return snr

def calculate_snr_ma(df, snr_column, ma_period=20):
    """計算SNR的移動平均"""
    return df[snr_column].rolling(window=ma_period).mean()

def calculate_snr_std(df, snr_column, std_period=20):
    """計算SNR的標準差"""
    return df[snr_column].rolling(window=std_period).std()

# =====================
# 技術指標計算
# =====================
def compute_indicators_malaysian_snr(df, params):
    """計算Malaysian SNR策略的技術指標"""
    
    # Malaysian SNR
    df['SNR'] = calculate_malaysian_snr(df, params['snr_period'])
    df['SNR_MA'] = calculate_snr_ma(df, 'SNR', params['snr_ma_period'])
    df['SNR_STD'] = calculate_snr_std(df, 'SNR', params['snr_std_period'])
    
    # SNR上下軌道
    df['SNR_UPPER'] = df['SNR_MA'] + params['snr_std_multiplier'] * df['SNR_STD']
    df['SNR_LOWER'] = df['SNR_MA'] - params['snr_std_multiplier'] * df['SNR_STD']
    
    # RSI
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(params['rsi_period']).mean()
    avg_loss = pd.Series(loss).rolling(params['rsi_period']).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 布林通道
    df['BB_MID'] = df['Close'].rolling(params['bb_window']).mean()
    df['BB_STD'] = df['Close'].rolling(params['bb_window']).std()
    df['BB_UPPER'] = df['BB_MID'] + params['bb_std'] * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - params['bb_std'] * df['BB_STD']
    
    # 均線
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    
    # 成交量指標
    df['Volume_MA'] = df['TotalVolume'].rolling(20).mean()
    df['Volume_Ratio'] = df['TotalVolume'] / df['Volume_MA']
    
    # 波動率指標
    df['ATR'] = calculate_atr(df, params['atr_period'])
    df['ATR_MA'] = df['ATR'].rolling(params['atr_ma_period']).mean()
    
    # K線型態
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
    df['Hammer'] = (
        (df['High'] - df['Low'] > 3 * (df['Open'] - df['Close'])) &
        ((df['Close'] - df['Low']) / (0.001 + df['High'] - df['Low']) > 0.6) &
        ((df['Open'] - df['Low']) / (0.001 + df['High'] - df['Low']) > 0.6)
    )
    df['Gravestone'] = (
        (df['High'] - df['Low'] > 3 * (df['Open'] - df['Close'])) &
        ((df['High'] - df['Close']) / (0.001 + df['High'] - df['Low']) > 0.6) &
        ((df['High'] - df['Open']) / (0.001 + df['High'] - df['Low']) > 0.6)
    )
    
    return df

def calculate_atr(df, period=14):
    """計算ATR (Average True Range)"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period).mean()
    
    return atr

# =====================
# 進場訊號生成
# =====================
def generate_entry_signal_malaysian_snr(df, params):
    """生成Malaysian SNR策略的進場訊號"""
    
    # 多單進場條件
    # 1. SNR突破下軌（超賣反轉）
    snr_oversold = df['SNR'] <= df['SNR_LOWER']
    
    # 2. RSI超賣
    rsi_oversold = df['RSI'] < params['rsi_oversold']
    
    # 3. 價格在布林下軌附近
    bb_oversold = df['Close'] <= df['BB_LOWER'] * (1 + params['bb_tolerance'])
    
    # 4. 成交量放大
    volume_surge = df['Volume_Ratio'] > params['volume_threshold']
    
    # 5. 波動率適中
    volatility_ok = (df['ATR'] > df['ATR_MA'] * params['atr_min_ratio']) & \
                   (df['ATR'] < df['ATR_MA'] * params['atr_max_ratio'])
    
    # 6. 均線支撐
    ma_support = df['Close'] > df['MA20']
    
    # 7. K線型態確認
    reversal_pattern = df['Hammer'] | df['Bull_Engulfing']
    
    # 多單最終條件
    if params.get('require_reversal_pattern', False):
        entry_long = snr_oversold & rsi_oversold & bb_oversold & volume_surge & volatility_ok & ma_support & reversal_pattern
    else:
        entry_long = snr_oversold & rsi_oversold & bb_oversold & volume_surge & volatility_ok & ma_support
    
    # 空單進場條件
    # 1. SNR突破上軌（超買反轉）
    snr_overbought = df['SNR'] >= df['SNR_UPPER']
    
    # 2. RSI超買
    rsi_overbought = df['RSI'] > params['rsi_overbought']
    
    # 3. 價格在布林上軌附近
    bb_overbought = df['Close'] >= df['BB_UPPER'] * (1 - params['bb_tolerance'])
    
    # 4. 成交量放大
    volume_surge_short = df['Volume_Ratio'] > params['volume_threshold']
    
    # 5. 波動率適中
    volatility_ok_short = (df['ATR'] > df['ATR_MA'] * params['atr_min_ratio']) & \
                         (df['ATR'] < df['ATR_MA'] * params['atr_max_ratio'])
    
    # 6. 均線阻力
    ma_resistance = df['Close'] < df['MA20']
    
    # 7. K線型態確認
    reversal_pattern_short = df['Gravestone'] | df['Bear_Engulfing']
    
    # 空單最終條件
    if params.get('require_reversal_pattern', False):
        entry_short = snr_overbought & rsi_overbought & bb_overbought & volume_surge_short & volatility_ok_short & ma_resistance & reversal_pattern_short
    else:
        entry_short = snr_overbought & rsi_overbought & bb_overbought & volume_surge_short & volatility_ok_short & ma_resistance
    
    # 記錄條件
    df['SNR_Oversold'] = snr_oversold
    df['RSI_Oversold'] = rsi_oversold
    df['BB_Oversold'] = bb_oversold
    df['Volume_Surge_Long'] = volume_surge
    df['Volatility_OK_Long'] = volatility_ok
    df['MA_Support'] = ma_support
    df['Reversal_Pattern_Long'] = reversal_pattern
    df['EntrySignal_Long'] = entry_long
    
    df['SNR_Overbought'] = snr_overbought
    df['RSI_Overbought'] = rsi_overbought
    df['BB_Overbought'] = bb_overbought
    df['Volume_Surge_Short'] = volume_surge_short
    df['Volatility_OK_Short'] = volatility_ok_short
    df['MA_Resistance'] = ma_resistance
    df['Reversal_Pattern_Short'] = reversal_pattern_short
    df['EntrySignal_Short'] = entry_short
    
    return df

# =====================
# 出場訊號生成
# =====================
def generate_exit_signal_malaysian_snr(df, params):
    """生成Malaysian SNR策略的出場訊號"""
    
    # 多單出場條件
    # 1. SNR回到中軌以上
    snr_exit_long = df['SNR'] >= df['SNR_MA']
    
    # 2. RSI超買
    rsi_exit_long = df['RSI'] > params['rsi_exit_long']
    
    # 3. 價格突破布林中軌
    bb_exit_long = df['Close'] > df['BB_MID']
    
    # 4. 均線死叉
    ma_death_cross = df['MA5'] < df['MA20']
    
    # 5. 反轉型態
    bearish_pattern = df['Gravestone'] | df['Bear_Engulfing']
    
    exit_long = snr_exit_long | rsi_exit_long | bb_exit_long | ma_death_cross | bearish_pattern
    
    # 空單出場條件
    # 1. SNR回到中軌以下
    snr_exit_short = df['SNR'] <= df['SNR_MA']
    
    # 2. RSI超賣
    rsi_exit_short = df['RSI'] < params['rsi_exit_short']
    
    # 3. 價格跌破布林中軌
    bb_exit_short = df['Close'] < df['BB_MID']
    
    # 4. 均線金叉
    ma_golden_cross = df['MA5'] > df['MA20']
    
    # 5. 反轉型態
    bullish_pattern = df['Hammer'] | df['Bull_Engulfing']
    
    exit_short = snr_exit_short | rsi_exit_short | bb_exit_short | ma_golden_cross | bullish_pattern
    
    df['ExitSignal_Long'] = exit_long
    df['ExitSignal_Short'] = exit_short
    
    return df

# =====================
# 交易執行
# =====================
def generate_trades_malaysian_snr(df, params):
    """從訊號產生交易紀錄"""
    trades = []
    position = 0
    entry_idx = None
    entry_price = None
    entry_direction = None
    initial_capital = 1000000
    current_capital = initial_capital
    stop_loss_pct = params.get('stop_loss_pct', 0.005)
    max_hold_bars = params.get('max_hold_bars', 10)
    
    for i, row in df.iterrows():
        if position == 0:
            if row.get('EntrySignal_Long', False):
                position = 1
                entry_idx = i
                entry_price = row['Close']
                entry_direction = 1
                # 動態止損：前低下方
                prev_low = df['Low'].iloc[max(0, i-5):i].min()
                stop_loss_price = prev_low * (1 - stop_loss_pct)
                entry_indicators = {
                    'SNR': row.get('SNR_Oversold', False),
                    'RSI': row.get('RSI_Oversold', False),
                    'BB': row.get('BB_Oversold', False),
                    'Volume': row.get('Volume_Surge_Long', False),
                    'Volatility': row.get('Volatility_OK_Long', False),
                    'MA': row.get('MA_Support', False),
                    'Pattern': row.get('Reversal_Pattern_Long', False)
                }
            elif row.get('EntrySignal_Short', False):
                position = -1
                entry_idx = i
                entry_price = row['Close']
                entry_direction = -1
                # 動態止損：前高上方
                prev_high = df['High'].iloc[max(0, i-5):i].max()
                stop_loss_price = prev_high * (1 + stop_loss_pct)
                entry_indicators = {
                    'SNR': row.get('SNR_Overbought', False),
                    'RSI': row.get('RSI_Overbought', False),
                    'BB': row.get('BB_Overbought', False),
                    'Volume': row.get('Volume_Surge_Short', False),
                    'Volatility': row.get('Volatility_OK_Short', False),
                    'MA': row.get('MA_Resistance', False),
                    'Pattern': row.get('Reversal_Pattern_Short', False)
                }
        elif position == 1:
            exit_reason = None
            exit_price = None
            
            # 出場條件檢查
            if row.get('ExitSignal_Long', False):
                exit_reason = '技術指標出場(多)'
                exit_price = row['Close']
            elif entry_idx is not None and i - entry_idx >= max_hold_bars:
                exit_reason = '持有時間過長(多)'
                exit_price = row['Close']
            elif row['Close'] <= stop_loss_price:
                exit_reason = '止損出場(多)'
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
            
            # 出場條件檢查
            if row.get('ExitSignal_Short', False):
                exit_reason = '技術指標出場(空)'
                exit_price = row['Close']
            elif entry_idx is not None and i - entry_idx >= max_hold_bars:
                exit_reason = '持有時間過長(空)'
                exit_price = row['Close']
            elif row['Close'] >= stop_loss_price:
                exit_reason = '止損出場(空)'
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
# 成本模型
# =====================
def apply_cost_model_malaysian_snr(trades, fee=1.5, slippage_long=1.0, slippage_short=2.0):
    """套用成本模型到交易紀錄"""
    if len(trades) == 0:
        return pd.DataFrame()
    
    result = trades.copy()
    
    # 計算成本
    result['GrossPnL'] = result['PnL']
    result['TotalCost'] = fee + np.where(result['Direction'] == 1, slippage_long, slippage_short)
    result['NetPnL'] = result['GrossPnL'] - result['TotalCost']
    
    return result

def calculate_performance_metrics_malaysian_snr(result):
    """計算績效指標"""
    if len(result) == 0:
        return {}
    
    metrics = {}
    metrics['total_trades'] = len(result)
    metrics['win_rate'] = (result['NetPnL'] > 0).mean()
    metrics['total_pnl'] = result['NetPnL'].sum()
    metrics['avg_pnl'] = result['NetPnL'].mean()
    metrics['max_profit'] = result['NetPnL'].max()
    metrics['max_loss'] = result['NetPnL'].min()
    metrics['profit_factor'] = abs(result[result['NetPnL'] > 0]['NetPnL'].sum() / result[result['NetPnL'] < 0]['NetPnL'].sum()) if result[result['NetPnL'] < 0]['NetPnL'].sum() != 0 else float('inf')
    
    # 最大回撤
    cumulative_pnl = result['NetPnL'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - running_max
    metrics['max_drawdown'] = drawdown.min()
    
    # 夏普比率
    if result['NetPnL'].std() > 0:
        metrics['sharpe_ratio'] = result['NetPnL'].mean() / result['NetPnL'].std()
    else:
        metrics['sharpe_ratio'] = 0
    
    return metrics

# =====================
# 主流程
# =====================
def main_malaysian_snr():
    """Malaysian SNR策略主流程"""
    print("🎬 開始執行Malaysian SNR策略")
    print("="*60)
    
    start_time = time.time()
    
    # 產生版本化的檔案名稱
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"Malaysian_SNR_策略報表_{timestamp}.xlsx"
    
    try:
        # 讀取資料
        print("📖 讀取台指期資料...")
        df = pd.read_csv('../scripts/TXF1_Minute_2020-01-01_2025-06-16.txt')
        df = df.dropna()
        df = df[df['Close'] > 0]
        
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
        
        # 策略參數
        params = {
            'snr_period': 14,
            'snr_ma_period': 20,
            'snr_std_period': 20,
            'snr_std_multiplier': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_exit_long': 60,
            'rsi_exit_short': 40,
            'bb_window': 20,
            'bb_std': 2.0,
            'bb_tolerance': 0.01,
            'volume_threshold': 1.5,
            'atr_period': 14,
            'atr_ma_period': 20,
            'atr_min_ratio': 0.8,
            'atr_max_ratio': 2.0,
            'stop_loss_pct': 0.005,
            'max_hold_bars': 10,
            'require_reversal_pattern': False
        }
        
        # 計算指標
        print("📊 計算Malaysian SNR指標...")
        df = compute_indicators_malaysian_snr(df, params)
        
        # 產生訊號
        print("🎯 產生交易訊號...")
        df = generate_entry_signal_malaysian_snr(df, params)
        df = generate_exit_signal_malaysian_snr(df, params)
        
        # 顯示訊號統計
        entry_signals = df['EntrySignal_Long'].sum() + df['EntrySignal_Short'].sum()
        exit_signals = df['ExitSignal_Long'].sum() + df['ExitSignal_Short'].sum()
        
        print(f"進場訊號數量: {entry_signals}")
        print(f"出場訊號數量: {exit_signals}")
        
        # 產生交易
        print("💼 產生交易紀錄...")
        trades = generate_trades_malaysian_snr(df, params)
        
        if len(trades) > 0:
            print(f"產生 {len(trades)} 筆交易")
            
            # 套用成本模型
            print("💰 套用成本模型...")
            result = apply_cost_model_malaysian_snr(trades, fee=1.5, slippage_long=1.0, slippage_short=2.0)
            
            # 計算績效
            print("📈 計算績效指標...")
            metrics = calculate_performance_metrics_malaysian_snr(result)
            
            # 顯示績效摘要
            print("\n" + "="*50)
            print("📊 Malaysian SNR策略績效摘要")
            print("="*50)
            print(f"總交易次數: {metrics['total_trades']}")
            print(f"勝率: {metrics['win_rate']:.2%}")
            print(f"總損益: {metrics['total_pnl']:.2f} 點")
            print(f"平均損益: {metrics['avg_pnl']:.2f} 點")
            print(f"最大獲利: {metrics['max_profit']:.2f} 點")
            print(f"最大虧損: {metrics['max_loss']:.2f} 點")
            print(f"獲利因子: {metrics['profit_factor']:.2f}")
            print(f"最大回撤: {metrics['max_drawdown']:.2f} 點")
            print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
            
            # 輸出到Excel
            print(f"\n📊 輸出結果到Excel檔案: {filename}")
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 策略摘要
                summary_data = {
                    '指標': ['總交易次數', '勝率', '總損益(點)', '平均損益(點)', 
                            '最大獲利(點)', '最大虧損(點)', '獲利因子', '最大回撤(點)', '夏普比率'],
                    '數值': [
                        metrics['total_trades'],
                        f"{metrics['win_rate']:.2%}",
                        f"{metrics['total_pnl']:.2f}",
                        f"{metrics['avg_pnl']:.2f}",
                        f"{metrics['max_profit']:.2f}",
                        f"{metrics['max_loss']:.2f}",
                        f"{metrics['profit_factor']:.2f}",
                        f"{metrics['max_drawdown']:.2f}",
                        f"{metrics['sharpe_ratio']:.2f}"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='策略摘要', index=False)
                
                # 詳細交易紀錄
                result.to_excel(writer, sheet_name='詳細交易紀錄', index=False)
                
                # 策略參數
                params_list = list(params.items())
                params_df = pd.DataFrame(params_list)
                params_df.columns = ['參數名稱', '參數值']
                params_df.to_excel(writer, sheet_name='策略參數', index=False)
            
            print(f"✅ Excel檔案已成功輸出: {filename}")
            print(f"📁 檔案位置: {os.path.abspath(filename)}")
        else:
            print("❌ 沒有產生任何交易")
        
        execution_time = time.time() - start_time
        print(f"\n⏱️ 總執行時間: {execution_time:.2f} 秒")
        print("\n🎉 Malaysian SNR策略執行完成！")
        
        return {
            'data': df,
            'trades': trades,
            'result': result if len(trades) > 0 else None,
            'params': params,
            'execution_time': execution_time
        }
        
    except Exception as e:
        print(f"❌ 執行過程中發生錯誤: {str(e)}")
        return None

if __name__ == '__main__':
    results = main_malaysian_snr() 