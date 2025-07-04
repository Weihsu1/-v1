"""
分析10萬美元實際交易的回報
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_returns():
    """分析實際回報"""
    print("💰 分析10萬美元實際交易回報...")
    print("="*60)
    
    # 讀取交易記錄
    try:
        df_trades = pd.read_excel('AAPL_tradingview_multi_timeframe_trades.xlsx')
        print(f"📊 讀取到 {len(df_trades)} 筆交易記錄")
    except Exception as e:
        print(f"❌ 無法讀取交易記錄: {e}")
        return
    
    # 設定初始資金
    initial_capital = 100000  # 10萬美元
    
    # 計算實際回報
    total_pnl = df_trades['pnl'].sum()
    final_capital = df_trades['capital'].iloc[-1] if len(df_trades) > 0 else initial_capital
    
    # 計算百分比回報
    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
    
    # 基本統計
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades['pnl'] > 0])
    losing_trades = len(df_trades[df_trades['pnl'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 平均損益
    avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    
    # 最大回撤
    cumulative_pnl = df_trades['pnl'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = (cumulative_pnl - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # 計算年化回報率
    if len(df_trades) > 0:
        first_trade = pd.to_datetime(df_trades['entry_time'].iloc[0])
        last_trade = pd.to_datetime(df_trades['exit_time'].iloc[-1])
        trading_days = (last_trade - first_trade).days
        if trading_days > 0:
            annual_return = (total_return_pct / trading_days) * 365
        else:
            annual_return = 0
    else:
        annual_return = 0
    
    # 顯示結果
    print(f"💰 初始資金: ${initial_capital:,.2f}")
    print(f"💰 最終資金: ${final_capital:,.2f}")
    print(f"💰 總損益: ${total_pnl:,.2f}")
    print(f"💰 總回報率: {total_return_pct:.2f}%")
    print(f"💰 年化回報率: {annual_return:.2f}%")
    print(f"📊 總交易次數: {total_trades}")
    print(f"📊 獲利交易: {winning_trades}")
    print(f"📊 虧損交易: {losing_trades}")
    print(f"📊 勝率: {win_rate:.2%}")
    print(f"📊 平均獲利: ${avg_win:,.2f}")
    print(f"📊 平均虧損: ${avg_loss:,.2f}")
    print(f"📊 最大回撤: {max_drawdown:.2f}%")
    
    # 計算風險調整後回報
    if df_trades['pnl'].std() > 0:
        sharpe_ratio = df_trades['pnl'].mean() / df_trades['pnl'].std()
        print(f"📊 夏普比率: {sharpe_ratio:.2f}")
    
    # 時間框架分析
    if 'timeframe' in df_trades.columns:
        print("\n📊 各時間框架表現:")
        timeframe_stats = df_trades.groupby('timeframe').agg({
            'pnl': ['count', 'sum', 'mean'],
            'exit_reason': lambda x: (x == 'take_profit').sum() / len(x)
        }).round(2)
        print(timeframe_stats)
    
    # 月度分析
    print("\n📊 月度表現分析:")
    df_trades['exit_month'] = pd.to_datetime(df_trades['exit_time']).dt.to_period('M')
    monthly_returns = df_trades.groupby('exit_month')['pnl'].sum()
    monthly_returns_pct = (monthly_returns / initial_capital) * 100
    
    print("月份\t\t損益($)\t\t回報率(%)")
    print("-" * 40)
    for month, pnl in monthly_returns.items():
        pct = monthly_returns_pct[month]
        print(f"{month}\t${pnl:,.2f}\t\t{pct:.2f}%")
    
    # 繪製回報曲線
    plt.figure(figsize=(15, 10))
    
    # 子圖1: 資金曲線
    plt.subplot(2, 2, 1)
    capital_curve = df_trades['capital'].values
    trade_dates = pd.to_datetime(df_trades['exit_time'])
    plt.plot(trade_dates, capital_curve, linewidth=2, color='blue')
    plt.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.7, label='初始資金')
    plt.fill_between(trade_dates, capital_curve, initial_capital, 
                     where=(capital_curve >= initial_capital), alpha=0.3, color='green')
    plt.fill_between(trade_dates, capital_curve, initial_capital, 
                     where=(capital_curve < initial_capital), alpha=0.3, color='red')
    plt.title(f'資金曲線 (初始: ${initial_capital:,.0f})', fontsize=12, fontweight='bold')
    plt.ylabel('資金 ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子圖2: 累積回報率
    plt.subplot(2, 2, 2)
    cumulative_return = ((capital_curve - initial_capital) / initial_capital) * 100
    plt.plot(trade_dates, cumulative_return, linewidth=2, color='green')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('累積回報率 (%)', fontsize=12, fontweight='bold')
    plt.ylabel('回報率 (%)')
    plt.grid(True, alpha=0.3)
    
    # 子圖3: 月度回報
    plt.subplot(2, 2, 3)
    monthly_returns_pct.plot(kind='bar', color=['green' if x > 0 else 'red' for x in monthly_returns_pct.values])
    plt.title('月度回報率 (%)', fontsize=12, fontweight='bold')
    plt.ylabel('回報率 (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 子圖4: 交易分布
    plt.subplot(2, 2, 4)
    plt.hist(df_trades['pnl'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.title('交易損益分布', fontsize=12, fontweight='bold')
    plt.xlabel('損益 ($)')
    plt.ylabel('交易次數')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('AAPL_100k_returns_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 風險分析
    print("\n⚠️ 風險分析:")
    print(f"最大單筆虧損: ${df_trades['pnl'].min():,.2f}")
    print(f"最大單筆獲利: ${df_trades['pnl'].max():,.2f}")
    print(f"損益標準差: ${df_trades['pnl'].std():,.2f}")
    
    # 連續虧損分析
    consecutive_losses = 0
    max_consecutive_losses = 0
    for pnl in df_trades['pnl']:
        if pnl < 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0
    
    print(f"最大連續虧損次數: {max_consecutive_losses}")
    
    # 實際建議
    print("\n💡 實際交易建議:")
    if total_return_pct > 0:
        print("✅ 系統在測試期間表現良好")
        if annual_return > 20:
            print("🚀 年化回報率優秀 (>20%)")
        elif annual_return > 10:
            print("👍 年化回報率良好 (10-20%)")
        else:
            print("📈 年化回報率一般 (<10%)")
    else:
        print("❌ 系統在測試期間虧損")
    
    if max_drawdown < -20:
        print("⚠️ 最大回撤較大，建議調整風險管理")
    elif max_drawdown < -10:
        print("📊 回撤在可接受範圍內")
    else:
        print("✅ 回撤控制良好")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    analyze_returns() 