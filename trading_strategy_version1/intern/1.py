"""
台指期策略開發專案起始框架
核心策略：布林通道（BB）＋RSI＋OBV
資料格式：台指期 5 分 K，csv 轉 DataFrame
"""


# 環境設定與函式庫載入
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_palette("husl")

print("優化環境設定完成！")

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cost_model import apply_cost_model

# 讀取台指期資料
df = pd.read_csv('../scripts/TXF1_Minute_2020-01-01_2025-06-16.txt')
print("資料讀取完成，資料筆數:", len(df))
print(df.head())
print("-"*60)
print(type(df))
print(repr(df))
print("-"*60)

# =====================
# 參數設定
# =====================
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

# =====================
# 多時間框架（MTF）模組
# =====================
def resample_to_4h(df):
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

# =====================
# 指標計算模組
# =====================
def compute_indicators(df, params, df_4h=None):
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
    # === 新增：趨勢判斷 ===
    short_window = params.get('trend_short_ma', 10)
    long_window = params.get('trend_long_ma', 50)
    df['MA_SHORT'] = df['Close'].rolling(short_window).mean()
    df['MA_LONG'] = df['Close'].rolling(long_window).mean()
    df['TREND'] = (df['MA_SHORT'] > df['MA_LONG']) | (df['MA_SHORT'] < df['MA_LONG'])
    return df

# =====================
# 訊號產生模組
# =====================
def generate_entry_signal(df, params):
    cond1 = df['Close'] < df['BB_LOWER']
    cond2 = df['RSI'] < params['rsi_oversold']
    cond3 = df['OBV'] > df['OBV_MA']
    # 4小時RSI濾波
    cond4 = df['RSI_4H'] > 50  # 4小時RSI>50才允許進場
    # === 新增：有趨勢才交易 ===
    trend_cond = df['TREND']  # True 表示有趨勢
    entry = (cond1.astype(int) + cond2.astype(int) + cond3.astype(int)) >= params['entry_n']
    df['EntrySignal'] = entry & cond4 & trend_cond
    return df

def generate_exit_signal(df, params):
    cond1 = df['Close'] > df['BB_MID']
    cond2 = df['RSI'] > params['rsi_exit']
    exit_signal = cond1 & cond2
    df['ExitSignal'] = exit_signal
    return df

def apply_stop_loss(df, params):
    # 假設有持倉時，若最大浮虧超過停損點數則出場
    # 這裡僅為範例，實際需根據持倉紀錄計算
    df['StopLossSignal'] = False  # TODO: 實作停損邏輯
    return df

# =====================
# 主流程
# =====================
def main():
    # 讀取資料
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
    # 產生4小時K
    df_4h = resample_to_4h(df)
    # 計算指標（含4小時RSI）
    df = compute_indicators(df, params, df_4h)
    # 產生進出場訊號
    df = generate_entry_signal(df, params)
    df = generate_exit_signal(df, params)
    df = apply_stop_loss(df, params)
    # 顯示部分結果
    output_cols = ['Date', 'Time', 'Close', 'RSI_4H', 'EntrySignal', 'ExitSignal']
    if isinstance(df, pd.DataFrame):
        print(df[output_cols].tail(10))
    else:
        print('df 不是 DataFrame，無法顯示結果')

    # ====== 模擬產生交易紀錄並計算成本 ======
    # 假設這裡有一個簡單的交易紀錄產生器（僅示範，實際應根據訊號產生）
    trades_data = {
        'EntryTime': ['2024-01-01 09:00', '2024-01-01 10:00'],
        'ExitTime':  ['2024-01-01 09:15', '2024-01-01 10:15'],
        'EntryPrice': [18000, 18100],
        'ExitPrice':  [18020, 18080],
        'Direction':  [1, -1],  # 1=多, -1=空
    }
    trades_df = pd.DataFrame(trades_data)
    result = apply_cost_model(trades_df, fee=1.5, slippage_long=1.0, slippage_short=2.0)
    print('\n含成本的交易紀錄：')
    print(result[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'Direction', 'GrossPnL', 'TotalCost', 'NetPnL']])

if __name__ == '__main__':
    main()
