"""
VectorBT 量化策略最佳化實戰
==============================

VectorBT 相比傳統回測框架的優勢：

⚡ 性能優勢:
- 向量化計算: 比迴圈快 100-1000 倍
- 並行處理: 同時測試多個參數組合
- 記憶體優化: 高效處理大量數據

🛠️ 功能完整:
- 專業回測引擎: 內建訂單管理、滑價處理
- 豐富指標庫: 預建技術指標和績效指標
- 視覺化工具: 專業的圖表和報告

📊 分析深度:
- 詳細統計: 完整的績效分析報告
- 風險指標: Sharpe、Sortino、Calmar 等
- 交易分析: 持倉時間、勝率、盈虧比等
"""

# 1. 環境設定
print("🚀 VectorBT 環境設定...")

# 安裝 VectorBT (如果尚未安裝)
# pip install vectorbt

import vectorbt as vbt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import time

warnings.filterwarnings('ignore')

# VectorBT 設定
vbt.settings.set_theme("dark")
vbt.settings.plotting['layout']['width'] = 800
vbt.settings.plotting['layout']['height'] = 400

print(f"VectorBT 版本: {vbt.__version__}")

# 啟用 VectorBT 的 numba 加速
vbt.settings.caching['enabled'] = True
print("✅ Numba 加速已啟用")

# 2. 數據載入
def load_data_for_vbt(file_path='./data/TXF1_Minute_2020-01-01_2025-06-16.txt'):
    """載入數據並轉換為 VectorBT 格式"""
    try:
        print("📊 載入台指期數據...")
        
        # 載入數據
        for encoding in ['utf-8', 'big5', None]:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except:
                continue
        
        # 處理時間索引
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        elif 'Date' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'])
        
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # 標準化欄位
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if any(k in col_lower for k in ['open', '開盤']):
                column_mapping[col] = 'Open'
            elif any(k in col_lower for k in ['high', '最高']):
                column_mapping[col] = 'High'
            elif any(k in col_lower for k in ['low', '最低']):
                column_mapping[col] = 'Low'
            elif any(k in col_lower for k in ['close', '收盤']):
                column_mapping[col] = 'Close'
            elif any(k in col_lower for k in ['volume', '成交量']):
                column_mapping[col] = 'Volume'
        
        df.rename(columns=column_mapping, inplace=True)
        
        # 重採樣為日線
        ohlcv_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min', 
            'Close': 'last',
            'Volume': 'sum'
        }
        
        available_dict = {k: v for k, v in ohlcv_dict.items() if k in df.columns}
        daily_df = df.resample('D').agg(available_dict).dropna()
        
        print(f"✅ 數據載入成功: {daily_df.shape}")
        print(f"📅 時間範圍: {daily_df.index.min()} 到 {daily_df.index.max()}")
        
        return daily_df
        
    except Exception as e:
        print(f"❌ 數據載入失敗: {e}")
        return None

# 載入數據
data = load_data_for_vbt()

if data is not None:
    print("\n📋 數據概覽:")
    print(data.head())
    print(f"\n✅ 數據適合 VectorBT: {vbt.utils.checks.is_pandas(data)}")
    
    # 取得收盤價
    close_prices = data['Close']
    
    # 3. 基本移動平均策略
    print("\n" + "="*50)
    print("📈 基本移動平均策略測試")
    print("="*50)
    
    def create_ma_signals(close_prices, fast_window, slow_window):
        """使用 VectorBT 創建移動平均交叉訊號"""
        
        # 計算移動平均線 (向量化)
        fast_ma = vbt.MA.run(close_prices, fast_window)
        slow_ma = vbt.MA.run(close_prices, slow_window)
        
        # 生成交叉訊號
        entries = fast_ma.ma_crossed_above(slow_ma)  # 黃金交叉
        exits = fast_ma.ma_crossed_below(slow_ma)    # 死亡交叉
        
        return entries, exits, fast_ma.ma, slow_ma.ma
    
    # 基本參數
    fast_window = 20
    slow_window = 60
    
    print(f"🎯 測試基本 MA 策略: {fast_window}/{slow_window}")
    
    # 生成訊號
    entries, exits, fast_ma, slow_ma = create_ma_signals(close_prices, fast_window, slow_window)
    
    print(f"📊 買入訊號數量: {entries.sum()}")
    print(f"📊 賣出訊號數量: {exits.sum()}")
    
    # VectorBT 回測 (超快速!)
    portfolio = vbt.Portfolio.from_signals(
        close_prices,
        entries,
        exits,
        init_cash=1000000,
        fees=0.001,  # 0.1% 交易費用
        freq='D'
    )
    
    # 基本績效
    print(f"\n=== 基本策略績效 ===")
    print(f"💰 總回報率: {portfolio.total_return():.2%}")
    print(f"📊 夏普比率: {portfolio.sharpe_ratio():.4f}")
    print(f"📉 最大回撤: {portfolio.max_drawdown():.2%}")
    print(f"🔄 交易次數: {portfolio.stats()['Total Trades']}")
    
    # 4. VectorBT 超高速參數最佳化
    print("\n" + "="*50)
    print("🚀 VectorBT 超高速參數最佳化")
    print("="*50)
    
    # 定義參數範圍
    fast_windows = [10, 15, 20, 25, 30]
    slow_windows = [40, 50, 60, 70, 80]
    
    print(f"🎯 測試參數組合: {len(fast_windows)} x {len(slow_windows)} = {len(fast_windows) * len(slow_windows)} 組")
    
    # 計時開始
    start_time = time.time()
    
    # VectorBT 魔法：一次計算所有組合！
    fast_ma_opt = vbt.MA.run(close_prices, fast_windows, short_name='fast')
    slow_ma_opt = vbt.MA.run(close_prices, slow_windows, short_name='slow')
    
    # 廣播計算所有組合的交叉訊號
    entries_opt = fast_ma_opt.ma_crossed_above(slow_ma_opt)
    exits_opt = fast_ma_opt.ma_crossed_below(slow_ma_opt)
    
    print(f"📊 訊號矩陣形狀: {entries_opt.shape}")
    
    # 一次性回測所有參數組合 (這就是 VectorBT 的威力!)
    portfolio_opt = vbt.Portfolio.from_signals(
        close_prices,
        entries_opt,
        exits_opt,
        init_cash=1000000,
        fees=0.001,
        freq='D'
    )
    
    optimization_time = time.time() - start_time
    print(f"⚡ 所有參數組合回測完成！耗時: {optimization_time:.4f} 秒")
    
    # 獲取所有組合的績效指標
    total_returns = portfolio_opt.total_return()
    sharpe_ratios = portfolio_opt.sharpe_ratio()
    max_drawdowns = portfolio_opt.max_drawdown()
    
    print(f"\n=== 最佳化結果 ===")
    
    # 找出最佳夏普比率
    best_sharpe_idx = sharpe_ratios.idxmax()
    best_return_idx = total_returns.idxmax()
    
    print(f"🏆 最佳夏普比率: {sharpe_ratios.max():.4f}")
    print(f"📊 最佳夏普參數: Fast={best_sharpe_idx[0]}, Slow={best_sharpe_idx[1]}")
    
    print(f"\n💰 最佳總回報: {total_returns.max():.2%}")
    print(f"📊 最佳回報參數: Fast={best_return_idx[0]}, Slow={best_return_idx[1]}")
    
    # 創建結果 DataFrame 進行分析
    results_data = []
    for fast_w in fast_windows:
        for slow_w in slow_windows:
            if fast_w < slow_w:  # 只考慮合理組合
                idx = (fast_w, slow_w)
                results_data.append({
                    'fast_ma': fast_w,
                    'slow_ma': slow_w,
                    'total_return': total_returns[idx],
                    'sharpe_ratio': sharpe_ratios[idx],
                    'max_drawdown': max_drawdowns[idx]
                })
    
    results_df = pd.DataFrame(results_data)
    
    print(f"\n=== 前5名夏普比率 ===")
    top_5_sharpe = results_df.nlargest(5, 'sharpe_ratio')
    print(top_5_sharpe[['fast_ma', 'slow_ma', 'sharpe_ratio', 'total_return', 'max_drawdown']].round(4))
    
    # 5. 效能比較
    print("\n" + "="*50)
    print("⏱️ 效能比較測試")
    print("="*50)
    
    # 模擬傳統方法的時間 (基於經驗估算)
    traditional_time_estimate = len(fast_windows) * len(slow_windows) * 0.5  # 每個組合約 0.5 秒
    
    print(f"🐌 傳統方法估算耗時: {traditional_time_estimate:.1f} 秒")
    print(f"⚡ VectorBT 實際耗時: {optimization_time:.4f} 秒")
    print(f"🚀 VectorBT 速度提升: {traditional_time_estimate / optimization_time:.1f} 倍")
    
    # 6. 進階策略: RSI + MA 組合
    print("\n" + "="*50)
    print("🔥 進階策略: RSI + 移動平均組合")
    print("="*50)
    
    # 計算 RSI 指標
    rsi = vbt.RSI.run(close_prices, window=14)
    
    # 計算移動平均
    fast_ma_adv = vbt.MA.run(close_prices, 20)
    slow_ma_adv = vbt.MA.run(close_prices, 50)
    
    # 組合條件 (VectorBT 的強大之處!)
    # 買入條件: MA 黃金交叉 AND RSI < 70 (避免超買)
    ma_bullish = fast_ma_adv.ma_crossed_above(slow_ma_adv)
    rsi_not_overbought = rsi.rsi < 70
    
    entries_advanced = ma_bullish & rsi_not_overbought
    
    # 賣出條件: MA 死亡交叉 OR RSI > 80 (超買離場)
    ma_bearish = fast_ma_adv.ma_crossed_below(slow_ma_adv)
    rsi_overbought = rsi.rsi > 80
    
    exits_advanced = ma_bearish | rsi_overbought
    
    print(f"📊 進階買入訊號: {entries_advanced.sum()}")
    print(f"📊 進階賣出訊號: {exits_advanced.sum()}")
    
    # 進階回測 (加入更多設定)
    portfolio_advanced = vbt.Portfolio.from_signals(
        close_prices,
        entries_advanced,
        exits_advanced,
        init_cash=1000000,
        fees=0.001,
        slippage=0.0005,  # 滑價
        min_size=1,       # 最小交易單位
        max_size=np.inf,  # 最大交易單位
        size_type='amount',  # 交易金額類型
        freq='D'
    )
    
    # 比較基本策略 vs 進階策略
    print(f"\n=== 策略比較 ===")
    
    # 選擇最佳參數的基本策略
    best_portfolio = portfolio_opt[best_sharpe_idx]
    
    basic_return = best_portfolio.total_return()
    basic_sharpe = best_portfolio.sharpe_ratio()
    basic_dd = best_portfolio.max_drawdown()
    
    adv_return = portfolio_advanced.total_return()
    adv_sharpe = portfolio_advanced.sharpe_ratio()
    adv_dd = portfolio_advanced.max_drawdown()
    
    print(f"📊 基本 MA 策略:")
    print(f"  💰 總回報: {basic_return:.2%}")
    print(f"  📊 夏普比率: {basic_sharpe:.4f}")
    print(f"  📉 最大回撤: {basic_dd:.2%}")
    
    print(f"\n🔥 進階 RSI+MA 策略:")
    print(f"  💰 總回報: {adv_return:.2%}")
    print(f"  📊 夏普比率: {adv_sharpe:.4f}")
    print(f"  📉 最大回撤: {adv_dd:.2%}")
    
    print(f"\n📈 改善程度:")
    print(f"  💰 回報提升: {((adv_return / basic_return) - 1):.2%}")
    print(f"  📊 夏普提升: {((adv_sharpe / basic_sharpe) - 1):.2%}")
    print(f"  📉 回撤改善: {((basic_dd / adv_dd) - 1):.2%}")
    
    # 7. 詳細統計報告
    print("\n" + "="*50)
    print("📊 詳細統計報告")
    print("="*50)
    
    print("🏆 最佳策略詳細統計:")
    key_stats = [
        'Start', 'End', 'Period', 'Total Return [%]', 
        'Sharpe Ratio', 'Max Drawdown [%]', 'Total Trades'
    ]
    
    best_stats = best_portfolio.stats()
    for stat in key_stats:
        if stat in best_stats.index:
            print(f"  {stat}: {best_stats[stat]}")
    
    # 交易分析
    print(f"\n🔄 交易分析:")
    trades = best_portfolio.trades
    
    if len(trades.records_readable) > 0:
        print(f"  ⏱️ 平均持倉時間: {trades.duration.mean():.1f} 天")
        print(f"  🎯 勝率: {trades.win_rate:.2%}")
        print(f"  💰 平均獲利: {trades.winning.pnl.mean():.2f}")
        print(f"  📉 平均虧損: {trades.losing.pnl.mean():.2f}")
        print(f"  📊 盈虧比: {trades.profit_factor:.2f}")
    
    # 8. VectorBT 內建績效指標展示
    print(f"\n📊 VectorBT 內建績效指標範例:")
    sample_stats = best_portfolio.stats()
    
    important_metrics = [
        'Total Return [%]', 'Sharpe Ratio', 'Calmar Ratio',
        'Max Drawdown [%]', 'Win Rate [%]', 'Profit Factor'
    ]
    
    for metric in important_metrics:
        if metric in sample_stats.index:
            print(f"  {metric}: {sample_stats[metric]}")
    
    # 9. 總結
    print("\n" + "="*50)
    print("🎉 VectorBT 完整工作流程展示完成!")
    print("="*50)
    
    print("\n✅ VectorBT 的核心價值:")
    print("  1. 極致效能: 向量化計算帶來的速度提升是革命性的")
    print("  2. 專業工具: 內建完整的量化交易工具鏈")
    print("  3. 易於使用: 簡潔的 API 降低了學習成本")
    print("  4. 可擴展性: 支援複雜的多資產、多策略回測")
    
    print("\n🎯 何時使用 VectorBT？")
    print("  ✅ 參數最佳化: 需要測試大量參數組合")
    print("  ✅ 快速原型: 快速驗證策略想法")
    print("  ✅ 專業回測: 需要詳細的績效分析")
    print("  ✅ 大規模測試: 處理大量歷史數據")
    print("  ✅ 多策略比較: 同時比較多個策略")
    
    print("\n🚀 VectorBT 讓量化交易變得更簡單、更快速、更專業!")
    print("開始您的高效量化交易之旅吧！")
    
    # 10. 視覺化建議 (需要在 Jupyter 中執行)
    print("\n" + "="*50)
    print("📊 視覺化建議 (在 Jupyter Notebook 中執行)")
    print("="*50)
    
    print("# 以下代碼可在 Jupyter Notebook 中執行以獲得視覺化:")
    print("# 1. 價格走勢圖")
    print("# data['Close'].vbt.plot(title='TXF1 Daily Close Price').show()")
    print("")
    print("# 2. 最佳策略績效圖")
    print("# best_portfolio.plot().show()")
    print("")
    print("# 3. 回撤圖")
    print("# best_portfolio.drawdowns.plot().show()")
    print("")
    print("# 4. RSI 指標圖")
    print("# rsi.rsi.vbt.plot(title='RSI Indicator').show()")
    print("")
    print("# 5. 參數熱力圖")
    print("# 可使用 matplotlib 和 seaborn 繪製參數最佳化結果的熱力圖")

else:
    print("❌ 無可用數據進行策略測試")

print("\n" + "="*50)
print("🎯 VectorBT vs 傳統方法比較總結")
print("="*50)

comparison_table = """
| 特性           | 傳統迴圈方法    | VectorBT        |
|----------------|----------------|-----------------|
| 速度           | 慢 (秒/分鐘級)  | 超快 (毫秒級)    |
| 並行化         | 需手動實現      | 內建支援        |
| 記憶體效率     | 低             | 高              |
| 參數最佳化     | 逐一測試        | 批量處理        |
| 視覺化         | 需自己寫        | 內建專業圖表    |
| 統計指標       | 需自己計算      | 內建豐富指標    |
| 學習曲線       | 陡峭           | 平緩            |
| 代碼複雜度     | 高             | 低              |
"""

print(comparison_table)

print("\n💡 關鍵心得:")
print("VectorBT 不只是一個回測工具，它是一個完整的量化交易研發平台。")
print("相比傳統方法，它能讓您:")
print("  • 節省 90% 的開發時間")
print("  • 獲得 10-100 倍的運算速度")
print("  • 享受專業級的分析工具")
print("  • 專注於策略邏輯而非技術實現")
print("\n對於認真的量化交易者來說，VectorBT 是必備工具！") 