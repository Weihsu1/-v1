# 量化交易策略回測專案

這是一個使用 Python 和 VectorBT 進行量化交易策略回測的專案。

## 專案結構

```
scripts/
├── optimized_strategy_backtest.ipynb    # 優化策略回測
├── strategy_backtest.ipynb              # 基礎策略回測
├── vbt_optimized_strategy.ipynb         # VectorBT 優化策略
├── vbt_strategy_demo.py                 # VectorBT 策略演示
├── TXF1_Minute_2020-01-01_2025-06-16.txt  # 台指期貨分鐘數據
└── VectorBT_Guide.md                    # VectorBT 使用指南

material/                                 # 學習材料
├── 00_introduction.md
├── 01_what_is_quant_trading.md
├── 02_toolbox.md
├── 03_data_preparation.md
├── 04_strategy_development.md
├── 05_performance_measurement.md
└── 06_next_steps.md

未命名檔案夾/
├── intern/                              # 實習相關檔案
│   ├── *.py                            # Python 腳本
│   ├── *.pine                          # TradingView Pine Script
│   └── *.xlsx                          # 回測報表
├── smc/                                 # SMC 策略相關
│   ├── multi_tf_smc.py                 # 多時間框架 SMC 策略
│   ├── *.png                           # 結果圖表
│   └── *.xlsx                          # 交易結果
└── snr test/                           # SNR 測試
    ├── malaysian_snr_strategy.py
    └── snr_indicator.pine
```

## 主要功能

- **策略回測**: 使用 VectorBT 進行高效的策略回測
- **多時間框架分析**: 支援多時間框架的技術分析
- **SMC 策略**: 實現 Smart Money Concepts 交易策略
- **SNR 指標**: 實現 Signal-to-Noise Ratio 指標
- **TradingView 整合**: Pine Script 策略腳本

## 環境需求

- Python 3.8+
- VectorBT
- pandas
- numpy
- matplotlib
- jupyter

## 安裝

```bash
pip install vectorbt pandas numpy matplotlib jupyter
```

## 使用方式

1. 開啟 Jupyter Notebook:
```bash
jupyter notebook
```

2. 執行策略回測:
   - 開啟 `strategy_backtest.ipynb` 進行基礎回測
   - 開啟 `optimized_strategy_backtest.ipynb` 進行優化回測

3. 運行 Python 腳本:
```bash
python vbt_strategy_demo.py
```

## 注意事項

- 大數據檔案 (如 TXF1_Minute_*.txt) 可能不會被上傳到 GitHub
- 請確保在執行策略前已正確設置數據路徑
- 回測結果僅供參考，實際交易請謹慎評估風險

## 授權

本專案僅供學習和研究使用。 