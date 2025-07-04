"""
成本考量（Cost Modeling）模組
- 每筆交易滑價 / 手續費：固定 1.5 點
- 做多與做空不同滑價
- 計算成本後的真實報酬
- 交易紀錄自動由策略訊號產生
"""
import pandas as pd
from cost_model import apply_cost_model

# 交易紀錄產生器：根據 EntrySignal/ExitSignal 產生交易紀錄
# 假設只做多，遇到 EntrySignal=True 進場，ExitSignal=True 出場

def generate_trades_from_signals(df):
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

if __name__ == '__main__':
    # 讀取scripts資料夾中的台指期資料
    df = pd.read_csv('../scripts/TXF1_Minute_2020-01-01_2025-06-16.txt')
    # 這裡省略指標與訊號計算，假設df已包含EntrySignal/ExitSignal
    # 若需完整流程，請將1.py的指標與訊號計算複製過來
    # 產生交易紀錄
    trades_df = generate_trades_from_signals(df)
    # 計算成本
    result = apply_cost_model(trades_df, fee=1.5, slippage_long=1.0, slippage_short=2.0)
    if len(result) == 0:
        print('無交易紀錄')
    else:
        print(result[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'Direction', 'GrossPnL', 'TotalCost', 'NetPnL']])
