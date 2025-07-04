import streamlit as st
import pandas as pd
import numpy as np
import os
from main_pipeline import main, step4_data_preprocessing, step5_feature_engineering, compute_indicators, generate_entry_signal, generate_exit_signal, generate_trades_from_signals, step3_performance_evaluation

def run_backtest(params):
    df = step4_data_preprocessing()
    df = step5_feature_engineering(df)
    df = compute_indicators(df, params)
    df = generate_entry_signal(df, params)
    df = generate_exit_signal(df, params)
    trades = generate_trades_from_signals(df, params)
    result = step3_performance_evaluation(trades)
    return trades, result

st.title('策略參數即時調整與回測')

with st.form("param_form"):
    st.subheader("策略參數設定")
    bb_window = st.slider('布林帶期間', 10, 40, 20)
    bb_std = st.slider('布林帶標準差', 1.0, 4.0, 2.5, 0.1)
    rsi_period = st.slider('RSI期間', 5, 30, 14)
    rsi_oversold = st.slider('RSI超賣', 10, 50, 30)
    rsi_overbought = st.slider('RSI超買', 50, 90, 70)
    stop_loss_pct = st.slider('止損百分比(%)', 0.1, 2.0, 0.3, 0.05) / 100
    require_reversal_kbar = st.checkbox('K棒型態必須', value=True)
    max_hold_bars = st.slider('最大持有K棒數', 1, 20, 5)
    submit = st.form_submit_button('執行回測')

params = {
    'bb_window': bb_window,
    'bb_std': bb_std,
    'rsi_period': rsi_period,
    'rsi_oversold': rsi_oversold,
    'rsi_overbought': rsi_overbought,
    'obv_ma_window': 10,
    'stop_loss_pct': stop_loss_pct,
    'require_reversal_kbar': require_reversal_kbar,
    'max_hold_bars': max_hold_bars,
    'entry_n': 2,
    'rsi_exit': 50,
    'rsi_exit_short': 50,
    'stop_loss': 20
}

if submit:
    with st.spinner('回測中...'):
        trades, result = run_backtest(params)
    st.success('回測完成！')
    if result is not None:
        st.write('績效摘要:')
        st.write(result.describe())
        st.write('前10筆交易:')
        st.dataframe(trades.head(10))
    else:
        st.warning('無交易紀錄，請調整參數。') 