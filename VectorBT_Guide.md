# VectorBT 量化策略最佳化指南

## 🚀 為什麼選擇 VectorBT？

### ⚡ 性能優勢
- **向量化計算**: 比迴圈快 100-1000 倍
- **並行處理**: 同時測試多個參數組合
- **記憶體優化**: 高效處理大量數據

### 🛠️ 功能完整
- **專業回測引擎**: 內建訂單管理、滑價處理
- **豐富指標庫**: 預建技術指標和績效指標
- **視覺化工具**: 專業的圖表和報告

### 📊 分析深度
- **詳細統計**: 完整的績效分析報告
- **風險指標**: Sharpe、Sortino、Calmar 等
- **交易分析**: 持倉時間、勝率、盈虧比等

## 📦 安裝方式

```bash
pip install vectorbt
```

## 🎯 核心功能展示

### 1. 基本移動平均策略

```python
import vectorbt as vbt
import pandas as pd

# 計算移動平均線 (向量化)
fast_ma = vbt.MA.run(close_prices, 20)
slow_ma = vbt.MA.run(close_prices, 60)

# 生成交叉訊號
entries = fast_ma.ma_crossed_above(slow_ma)  # 黃金交叉
exits = fast_ma.ma_crossed_below(slow_ma)    # 死亡交叉

# VectorBT 回測
portfolio = vbt.Portfolio.from_signals(
    close_prices,
    entries,
    exits,
    init_cash=1000000,
    fees=0.001
)
```

### 2. 超高速參數最佳化

```python
# 定義參數範圍
fast_windows = [10, 15, 20, 25, 30]
slow_windows = [40, 50, 60, 70, 80]

# VectorBT 魔法：一次計算所有組合！
fast_ma_opt = vbt.MA.run(close_prices, fast_windows)
slow_ma_opt = vbt.MA.run(close_prices, slow_windows)

# 廣播計算所有組合的交叉訊號
entries_opt = fast_ma_opt.ma_crossed_above(slow_ma_opt)
exits_opt = fast_ma_opt.ma_crossed_below(slow_ma_opt)

# 一次性回測所有參數組合
portfolio_opt = vbt.Portfolio.from_signals(
    close_prices,
    entries_opt,
    exits_opt,
    init_cash=1000000,
    fees=0.001
)

# 獲取最佳參數
sharpe_ratios = portfolio_opt.sharpe_ratio()
best_params = sharpe_ratios.idxmax()
```

### 3. 進階策略組合

```python
# 計算 RSI 指標
rsi = vbt.RSI.run(close_prices, window=14)

# 組合條件
ma_bullish = fast_ma.ma_crossed_above(slow_ma)
rsi_not_overbought = rsi.rsi < 70

# 買入條件: MA 黃金交叉 AND RSI < 70
entries_advanced = ma_bullish & rsi_not_overbought

# 賣出條件: MA 死亡交叉 OR RSI > 80
ma_bearish = fast_ma.ma_crossed_below(slow_ma)
rsi_overbought = rsi.rsi > 80
exits_advanced = ma_bearish | rsi_overbought
```

## 📊 效能比較

| 特性 | 傳統迴圈方法 | VectorBT |
|------|-------------|----------|
| **速度** | 慢 (秒/分鐘級) | 超快 (毫秒級) |
| **並行化** | 需手動實現 | 內建支援 |
| **記憶體效率** | 低 | 高 |
| **參數最佳化** | 逐一測試 | 批量處理 |
| **視覺化** | 需自己寫 | 內建專業圖表 |
| **統計指標** | 需自己計算 | 內建豐富指標 |

## 💡 實際效能測試

```python
# 傳統方法: 測試 25 個參數組合可能需要 30-60 秒
# VectorBT: 測試 25 個參數組合只需要 1-3 秒
# 效能提升: 10-50 倍！
```

## 🎯 何時使用 VectorBT？

- ✅ **參數最佳化**: 需要測試大量參數組合
- ✅ **快速原型**: 快速驗證策略想法
- ✅ **專業回測**: 需要詳細的績效分析
- ✅ **大規模測試**: 處理大量歷史數據
- ✅ **多策略比較**: 同時比較多個策略

## 📈 內建績效指標

VectorBT 自動計算超過 50 種績效指標：

```python
stats = portfolio.stats()

# 主要指標包括:
# - Total Return [%]
# - Sharpe Ratio
# - Calmar Ratio
# - Max Drawdown [%]
# - Win Rate [%]
# - Profit Factor
# - Average Trade Duration
# - 等等...
```

## 🎨 視覺化功能

```python
# 1. 策略績效圖
portfolio.plot().show()

# 2. 回撤圖
portfolio.drawdowns.plot().show()

# 3. 價格走勢圖
close_prices.vbt.plot().show()

# 4. 指標圖
rsi.rsi.vbt.plot().show()
```

## 🚀 進階功能

### 1. 多資產回測
```python
# 同時回測多個資產
multi_portfolio = vbt.Portfolio.from_signals(
    multi_asset_prices,  # DataFrame with multiple columns
    multi_entries,
    multi_exits
)
```

### 2. 自定義指標
```python
# 創建自定義技術指標
@vbt.IF(
    class_name='CustomIndicator',
    short_name='ci'
)
def custom_indicator(close, param1, param2):
    # 自定義計算邏輯
    return result
```

### 3. 複雜訂單類型
```python
portfolio = vbt.Portfolio.from_signals(
    close_prices,
    entries,
    exits,
    init_cash=1000000,
    fees=0.001,
    slippage=0.0005,     # 滑價
    min_size=1,          # 最小交易單位
    max_size=np.inf,     # 最大交易單位
    size_type='amount'   # 交易金額類型
)
```

## 🎓 學習資源

1. **官方文檔**: https://vectorbt.dev/
2. **範例集**: https://github.com/polakowo/vectorbt
3. **社群論壇**: GitHub Issues 和 Discord

## 💼 實際應用建議

### 1. 整合到交易系統
- 結合實時數據源
- 建立自動化策略選擇
- 整合風險管理模組

### 2. 持續最佳化
- 定期重新最佳化參數
- 加入更多市場指標
- 考慮機器學習增強

### 3. 風險管理
- 使用 VectorBT 的風險指標
- 設定停損停利機制
- 監控回撤和波動率

## 🎉 總結

VectorBT 不只是一個回測工具，它是一個完整的量化交易研發平台。相比傳統方法，它能讓您：

- **節省 90% 的開發時間**
- **獲得 10-100 倍的運算速度**
- **享受專業級的分析工具**
- **專注於策略邏輯而非技術實現**

對於認真的量化交易者來說，VectorBT 是必備工具！

## 🔗 相關檔案

- `vbt_strategy_demo.py` - 完整的 VectorBT 策略示範代碼
- `optimized_strategy_backtest.ipynb` - 傳統方法的參數最佳化範例

---

*建議先執行 `vbt_strategy_demo.py` 體驗 VectorBT 的強大功能！* 