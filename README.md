# 量化交易策略回測專案

這是一個使用 Python 和 VectorBT 進行量化交易策略回測的專案，採用分段上傳的方式來管理不同功能模組。

## 📁 專案結構

### 🎯 核心策略模組 (Core Strategy Modules)
```
scripts/
├── strategy_backtest.ipynb              # 基礎策略回測 - 入門級策略測試
├── optimized_strategy_backtest.ipynb    # 優化策略回測 - 進階策略優化
├── vbt_optimized_strategy.ipynb         # VectorBT 優化策略 - 高效能回測
├── vbt_strategy_demo.py                 # VectorBT 策略演示 - 完整策略範例
└── VectorBT_Guide.md                    # VectorBT 使用指南 - 詳細教學文件
```

### 📚 學習材料 (Learning Materials)
```
material/
├── 00_introduction.md                   # 量化交易介紹
├── 01_what_is_quant_trading.md          # 什麼是量化交易
├── 02_toolbox.md                        # 量化交易工具箱
├── 03_data_preparation.md               # 數據準備
├── 04_strategy_development.md           # 策略開發
├── 05_performance_measurement.md        # 績效衡量
└── 06_next_steps.md                     # 下一步學習
```

### 🔧 實習專案 (Internship Projects)
```
intern_files/
├── intern/                              # 實習任務檔案
│   ├── 1.py - 7.py                     # 基礎 Python 練習
│   ├── cost_model.py                    # 成本模型計算
│   ├── diagnose_signals.py              # 信號診斷工具
│   ├── main_pipeline.py                 # 主要處理流程
│   ├── parameter_dashboard.py           # 參數儀表板
│   ├── *.pine                          # TradingView Pine Script
│   └── *.xlsx                          # 回測報表
├── smc/                                 # Smart Money Concepts 策略
│   ├── multi_tf_smc.py                 # 多時間框架 SMC 策略
│   ├── multi_tf_smc_system.py          # SMC 系統整合
│   ├── simple_multi_tf_smc.py          # 簡化版多時間框架 SMC
│   ├── tradingview_multi_tf_smc.py     # TradingView 整合 SMC
│   ├── analyze_returns.py              # 收益分析工具
│   ├── realistic_returns_analysis.py   # 真實收益分析
│   ├── *.png                           # 策略結果圖表
│   └── *.xlsx                          # 交易結果報表
└── snr test/                           # Signal-to-Noise Ratio 測試
    ├── malaysian_snr_strategy.py       # 馬來西亞 SNR 策略
    ├── snr_indicator.pine              # SNR 指標 (Pine Script)
    └── snr_strategy.pine               # SNR 策略 (Pine Script)
```

## 🚀 主要功能模組

### 1. **基礎策略回測** (`strategy_backtest.ipynb`)
- 入門級量化策略實現
- 基本技術指標應用
- 簡單買賣信號生成
- 適合初學者學習

### 2. **優化策略回測** (`optimized_strategy_backtest.ipynb`)
- 進階策略優化技術
- 參數自動調優
- 多策略組合測試
- 風險管理整合

### 3. **VectorBT 高效回測** (`vbt_optimized_strategy.ipynb`)
- 使用 VectorBT 框架
- 向量化計算加速
- 大規模策略測試
- 專業級回測工具

### 4. **SMC 策略系統** (`intern_files/smc/`)
- Smart Money Concepts 實現
- 多時間框架分析
- 機構資金流向追蹤
- 高級技術分析

### 5. **SNR 指標系統** (`intern_files/snr test/`)
- Signal-to-Noise Ratio 計算
- 市場噪音過濾
- 信號品質評估
- 交易時機優化

## 🛠️ 環境需求

- Python 3.8+
- VectorBT
- pandas
- numpy
- matplotlib
- jupyter

## 📦 安裝

```bash
pip install vectorbt pandas numpy matplotlib jupyter
```

## 📖 使用方式

### 1. 基礎學習
```bash
# 開啟 Jupyter Notebook
jupyter notebook

# 依序執行：
# 1. strategy_backtest.ipynb (基礎策略)
# 2. optimized_strategy_backtest.ipynb (優化策略)
# 3. vbt_optimized_strategy.ipynb (VectorBT 策略)
```

### 2. 進階應用
```bash
# 執行 VectorBT 演示
python vbt_strategy_demo.py

# 運行 SMC 策略
cd intern_files/smc/
python multi_tf_smc.py

# 執行 SNR 測試
cd intern_files/snr\ test/
python malaysian_snr_strategy.py
```

## 📊 數據說明

**注意**: 大數據檔案 (如 TXF1_Minute_*.txt) 因 GitHub 檔案大小限制未包含在此倉庫中。

如需數據檔案，請：
1. 自行準備台指期貨分鐘數據
2. 或聯繫專案維護者獲取數據檔案

## ⚠️ 重要提醒

- 回測結果僅供學習和研究使用
- 實際交易請謹慎評估風險
- 策略參數需要根據市場情況調整
- 建議在模擬環境中充分測試

## 📝 授權

本專案僅供學習和研究使用。

## 🔄 更新日誌

- **v1.0**: 初始版本，包含基礎策略回測
- **v1.1**: 新增 SMC 和 SNR 策略模組
- **v1.2**: 整合 VectorBT 高效回測框架 