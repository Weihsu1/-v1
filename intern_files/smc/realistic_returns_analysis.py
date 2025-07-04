"""
修正後的10萬美元實際交易回報分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def realistic_returns_analysis():
    """現實的回報分析"""
    print("💰 10萬美元實際交易回報分析 (修正版)")
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
    
    # 修正損益計算 - 使用合理的風險管理
    print("\n🔧 修正損益計算...")
    
    # 重新計算每筆交易的損益
    corrected_trades = []
    current_capital = initial_capital
    
    for idx, trade in df_trades.iterrows():
        # 使用2%風險管理
        risk_amount = current_capital * 0.02
        
        # 計算實際損益 (基於風險金額)
        if trade['pnl'] > 0:
            # 獲利交易 - 假設2:1風險報酬比
            actual_pnl = risk_amount * 2
        else:
            # 虧損交易 - 損失風險金額
            actual_pnl = -risk_amount
        
        # 更新資金
        current_capital += actual_pnl
        
        # 創建修正後的交易記錄
        corrected_trade = trade.copy()
        corrected_trade['corrected_pnl'] = actual_pnl
        corrected_trade['corrected_capital'] = current_capital
        corrected_trades.append(corrected_trade)
    
    df_corrected = pd.DataFrame(corrected_trades)
    
    # 計算修正後的統計
    total_pnl = df_corrected['corrected_pnl'].sum()
    final_capital = df_corrected['corrected_capital'].iloc[-1]
    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
    
    # 基本統計
    total_trades = len(df_corrected)
    winning_trades = len(df_corrected[df_corrected['corrected_pnl'] > 0])
    losing_trades = len(df_corrected[df_corrected['corrected_pnl'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 平均損益
    avg_win = df_corrected[df_corrected['corrected_pnl'] > 0]['corrected_pnl'].mean() if winning_trades > 0 else 0
    avg_loss = df_corrected[df_corrected['corrected_pnl'] < 0]['corrected_pnl'].mean() if losing_trades > 0 else 0
    
    # 最大回撤
    cumulative_pnl = df_corrected['corrected_pnl'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = (cumulative_pnl - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # 計算年化回報率
    if len(df_corrected) > 0:
        first_trade = pd.to_datetime(df_corrected['entry_time'].iloc[0])
        last_trade = pd.to_datetime(df_corrected['exit_time'].iloc[-1])
        trading_days = (last_trade - first_trade).days
        if trading_days > 0:
            annual_return = (total_return_pct / trading_days) * 365
        else:
            annual_return = 0
    else:
        annual_return = 0
    
    # 顯示修正後的結果
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
    if df_corrected['corrected_pnl'].std() > 0:
        sharpe_ratio = df_corrected['corrected_pnl'].mean() / df_corrected['corrected_pnl'].std()
        print(f"📊 夏普比率: {sharpe_ratio:.2f}")
    
    # 時間框架分析
    if 'timeframe' in df_corrected.columns:
        print("\n📊 各時間框架表現:")
        timeframe_stats = df_corrected.groupby('timeframe').agg({
            'corrected_pnl': ['count', 'sum', 'mean']
        }).round(2)
        print(timeframe_stats)
    
    # 月度分析
    print("\n📊 月度表現分析:")
    df_corrected['exit_month'] = pd.to_datetime(df_corrected['exit_time']).dt.to_period('M')
    monthly_returns = df_corrected.groupby('exit_month')['corrected_pnl'].sum()
    monthly_returns_pct = (monthly_returns / initial_capital) * 100
    
    print("月份\t\t損益($)\t\t回報率(%)")
    print("-" * 40)
    for month, pnl in monthly_returns.items():
        pct = monthly_returns_pct[month]
        print(f"{month}\t${pnl:,.2f}\t\t{pct:.2f}%")
    
    # 繪製修正後的回報曲線
    plt.figure(figsize=(15, 10))
    
    # 子圖1: 資金曲線
    plt.subplot(2, 2, 1)
    capital_curve = df_corrected['corrected_capital'].values
    trade_dates = pd.to_datetime(df_corrected['exit_time'])
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
    plt.hist(df_corrected['corrected_pnl'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.title('交易損益分布', fontsize=12, fontweight='bold')
    plt.xlabel('損益 ($)')
    plt.ylabel('交易次數')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('AAPL_100k_realistic_returns.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 風險分析
    print("\n⚠️ 風險分析:")
    print(f"最大單筆虧損: ${df_corrected['corrected_pnl'].min():,.2f}")
    print(f"最大單筆獲利: ${df_corrected['corrected_pnl'].max():,.2f}")
    print(f"損益標準差: ${df_corrected['corrected_pnl'].std():,.2f}")
    
    # 連續虧損分析
    consecutive_losses = 0
    max_consecutive_losses = 0
    for pnl in df_corrected['corrected_pnl']:
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
    
    # 實際可行性評估
    print("\n🎯 實際可行性評估:")
    print(f"📊 交易頻率: {total_trades} 筆 / 約5個月 = {total_trades/5:.1f} 筆/月")
    print(f"📊 平均每筆交易損益: ${total_pnl/total_trades:,.2f}")
    print(f"📊 資金利用率: 每筆交易使用2%資金")
    
    if total_trades > 1000:
        print("⚠️ 交易頻率過高，實際執行可能有困難")
    elif total_trades > 500:
        print("📊 交易頻率中等，需要較多時間管理")
    else:
        print("✅ 交易頻率合理，適合實際執行")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    realistic_returns_analysis() 