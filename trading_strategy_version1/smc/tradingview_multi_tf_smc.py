"""
TradingView多時間框架SMC短線交易系統
結合5分鐘、15分鐘、1小時時間框架進行交易
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
import requests
import json

warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TradingViewMultiTimeframeSMC:
    """TradingView多時間框架SMC交易系統"""
    
    def __init__(self, symbol='AAPL', start_date='2024-01-01', end_date='2024-06-01'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.timeframes = {'5m': 5, '15m': 15, '1h': 60}
        self.data = {}
        self.signals = []
        self.trades = []
    
    def fetch_data(self):
        """獲取多時間框架資料"""
        print(f"📊 獲取 {self.symbol} 的多時間框架資料...")
        
        for tf_name, tf_minutes in self.timeframes.items():
            print(f"🔄 處理 {tf_name} 時間框架...")
            
            # 生成模擬資料（實際使用時可替換為TradingView API）
            df = self._generate_mock_data(tf_minutes)
            df = self._add_indicators(df)
            self.data[tf_name] = df
            
            print(f"✅ {tf_name}: {len(df)} 筆資料")
        
        return self.data
    
    def _generate_mock_data(self, timeframe_minutes):
        """生成模擬資料"""
        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)
        
        # 根據時間框架設定頻率
        if timeframe_minutes == 5:
            freq = '5T'
        elif timeframe_minutes == 15:
            freq = '15T'
        elif timeframe_minutes == 60:
            freq = '1H'
        else:
            freq = '1H'
        
        # 生成時間序列
        date_range = pd.date_range(start=start_dt, end=end_dt, freq=freq)
        
        # 生成價格資料
        np.random.seed(42)
        base_price = 100.0
        returns = np.random.normal(0, 0.002, len(date_range))
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # 生成OHLC資料
        data = []
        for dt, price in zip(date_range, prices):
            volatility = 0.01
            open_price = price * (1 + np.random.uniform(-volatility/2, volatility/2))
            high_price = max(open_price, price) * (1 + np.random.uniform(0, volatility))
            low_price = min(open_price, price) * (1 - np.random.uniform(0, volatility))
            close_price = price
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=date_range)
        df.index.name = 'Datetime'
        return df
    
    def _add_indicators(self, df):
        """添加技術指標"""
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 移動平均線
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # 成交量指標
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        return df
    
    def detect_order_blocks(self, df):
        """檢測訂單塊"""
        bullish_blocks = []
        bearish_blocks = []
        
        for i in range(20, len(df) - 1):
            current = df.iloc[i]
            next_candle = df.iloc[i + 1]
            
            # 看漲訂單塊
            if (current['Close'] > current['Open'] and 
                current['Volume'] > current['Volume_SMA'] * 1.2 and
                next_candle['High'] > current['High']):
                
                for j in range(i + 2, min(i + 15, len(df))):
                    test = df.iloc[j]
                    if (test['Low'] <= current['High'] and 
                        test['High'] >= current['Low']):
                        
                        block_size = current['High'] - current['Low']
                        if block_size >= 0.3:
                            bullish_blocks.append({
                                'start_idx': i,
                                'end_idx': j,
                                'high': current['High'],
                                'low': current['Low'],
                                'strength': block_size,
                                'volume_ratio': current['Volume_Ratio']
                            })
                        break
            
            # 看跌訂單塊
            elif (current['Close'] < current['Open'] and 
                  current['Volume'] > current['Volume_SMA'] * 1.2 and
                  next_candle['Low'] < current['Low']):
                
                for j in range(i + 2, min(i + 15, len(df))):
                    test = df.iloc[j]
                    if (test['Low'] <= current['High'] and 
                        test['High'] >= current['Low']):
                        
                        block_size = current['High'] - current['Low']
                        if block_size >= 0.3:
                            bearish_blocks.append({
                                'start_idx': i,
                                'end_idx': j,
                                'high': current['High'],
                                'low': current['Low'],
                                'strength': block_size,
                                'volume_ratio': current['Volume_Ratio']
                            })
                        break
        
        return bullish_blocks, bearish_blocks
    
    def detect_fvgs(self, df):
        """檢測FVG"""
        fvgs = []
        
        for i in range(1, len(df) - 1):
            current = df.iloc[i]
            prev = df.iloc[i - 1]
            next_candle = df.iloc[i + 1]
            
            # 看漲FVG
            if (prev['Low'] > current['High'] and
                next_candle['High'] > prev['Low']):
                
                gap_size = prev['Low'] - current['High']
                if gap_size >= 0.1:
                    fvgs.append({
                        'type': 'bullish_fvg',
                        'idx': i,
                        'top': prev['Low'],
                        'bottom': current['High'],
                        'size': gap_size
                    })
            
            # 看跌FVG
            elif (current['Low'] > prev['High'] and
                  next_candle['Low'] < current['High']):
                
                gap_size = current['Low'] - prev['High']
                if gap_size >= 0.1:
                    fvgs.append({
                        'type': 'bearish_fvg',
                        'idx': i,
                        'top': current['Low'],
                        'bottom': prev['High'],
                        'size': gap_size
                    })
        
        return fvgs
    
    def generate_signals(self):
        """生成多時間框架訊號"""
        print("\n🎯 生成多時間框架交易訊號...")
        
        all_signals = []
        
        for tf_name, df in self.data.items():
            print(f"📊 分析 {tf_name} 時間框架...")
            
            # 檢測訂單塊
            bullish_ob, bearish_ob = self.detect_order_blocks(df)
            
            # 檢測FVG
            fvgs = self.detect_fvgs(df)
            
            # 生成訊號
            for ob in bullish_ob:
                if ob['end_idx'] < len(df) - 1:
                    all_signals.append({
                        'datetime': df.index[ob['end_idx']],
                        'type': 'order_block_bullish',
                        'timeframe': tf_name,
                        'price': df.iloc[ob['end_idx']]['Close'],
                        'strength': ob['strength'],
                        'direction': 'long',
                        'reason': f"{tf_name}看漲訂單塊回測完成"
                    })
            
            for ob in bearish_ob:
                if ob['end_idx'] < len(df) - 1:
                    all_signals.append({
                        'datetime': df.index[ob['end_idx']],
                        'type': 'order_block_bearish',
                        'timeframe': tf_name,
                        'price': df.iloc[ob['end_idx']]['Close'],
                        'strength': ob['strength'],
                        'direction': 'short',
                        'reason': f"{tf_name}看跌訂單塊回測完成"
                    })
            
            for fvg in fvgs:
                if fvg['idx'] < len(df) - 1:
                    if fvg['type'] == 'bullish_fvg':
                        all_signals.append({
                            'datetime': df.index[fvg['idx']],
                            'type': 'fvg_bullish',
                            'timeframe': tf_name,
                            'price': df.iloc[fvg['idx']]['Close'],
                            'strength': fvg['size'],
                            'direction': 'long',
                            'reason': f"{tf_name}看漲FVG形成"
                        })
                    else:
                        all_signals.append({
                            'datetime': df.index[fvg['idx']],
                            'type': 'fvg_bearish',
                            'timeframe': tf_name,
                            'price': df.iloc[fvg['idx']]['Close'],
                            'strength': fvg['size'],
                            'direction': 'short',
                            'reason': f"{tf_name}看跌FVG形成"
                        })
        
        # 按時間排序並過濾
        all_signals.sort(key=lambda x: x['datetime'])
        
        # 時間框架權重過濾
        timeframe_weights = {'5m': 0.3, '15m': 0.5, '1h': 0.8}
        filtered_signals = []
        
        for signal in all_signals:
            weight = timeframe_weights.get(signal['timeframe'], 0.5)
            signal['weight'] = weight
            signal['composite_strength'] = signal['strength'] * weight
            filtered_signals.append(signal)
        
        self.signals = filtered_signals
        print(f"📊 總共生成 {len(filtered_signals)} 個訊號")
        
        return filtered_signals
    
    def backtest(self, initial_capital=100000):
        """執行回測"""
        print("\n📈 開始交易回測...")
        
        if '5m' not in self.data:
            print("❌ 缺少5分鐘資料")
            return
        
        df_5m = self.data['5m']
        capital = initial_capital
        position = None
        trades = []
        
        for signal in self.signals:
            signal_time = pd.to_datetime(signal['datetime'])
            time_diff = abs(df_5m.index - signal_time)
            current_idx = time_diff.argmin()
            
            if current_idx >= len(df_5m) - 1:
                continue
            
            current_price = df_5m.iloc[current_idx]['Close']
            
            # 進場邏輯
            if position is None:
                risk_per_trade = capital * 0.02
                stop_loss_pct = 0.03
                
                if signal['direction'] == 'long':
                    stop_loss_price = current_price * (1 - stop_loss_pct)
                    position_size = risk_per_trade / (current_price - stop_loss_price)
                    
                    position = {
                        'type': 'long',
                        'entry_price': current_price,
                        'entry_time': signal_time,
                        'position_size': position_size,
                        'stop_loss': stop_loss_price,
                        'take_profit': current_price * (1 + stop_loss_pct * 2),
                        'time_stop': signal_time + timedelta(hours=4)
                    }
                
                elif signal['direction'] == 'short':
                    stop_loss_price = current_price * (1 + stop_loss_pct)
                    position_size = risk_per_trade / (stop_loss_price - current_price)
                    
                    position = {
                        'type': 'short',
                        'entry_price': current_price,
                        'entry_time': signal_time,
                        'position_size': position_size,
                        'stop_loss': stop_loss_price,
                        'take_profit': current_price * (1 - stop_loss_pct * 2),
                        'time_stop': signal_time + timedelta(hours=4)
                    }
            
            # 平倉邏輯
            elif position is not None:
                exit_reason = None
                exit_price = None
                
                if position['type'] == 'long':
                    if current_price <= position['stop_loss']:
                        exit_reason = 'stop_loss'
                        exit_price = position['stop_loss']
                    elif current_price >= position['take_profit']:
                        exit_reason = 'take_profit'
                        exit_price = current_price
                    elif signal_time >= position['time_stop']:
                        exit_reason = 'time_stop'
                        exit_price = current_price
                
                elif position['type'] == 'short':
                    if current_price >= position['stop_loss']:
                        exit_reason = 'stop_loss'
                        exit_price = position['stop_loss']
                    elif current_price <= position['take_profit']:
                        exit_reason = 'take_profit'
                        exit_price = current_price
                    elif signal_time >= position['time_stop']:
                        exit_reason = 'time_stop'
                        exit_price = current_price
                
                if exit_reason:
                    if position['type'] == 'long':
                        pnl = (exit_price - position['entry_price']) * position['position_size']
                    else:
                        pnl = (position['entry_price'] - exit_price) * position['position_size']
                    
                    capital += pnl
                    
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': signal_time,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'direction': position['type'],
                        'pnl': pnl,
                        'capital': capital,
                        'exit_reason': exit_reason,
                        'timeframe': signal['timeframe']
                    }
                    
                    trades.append(trade)
                    position = None
        
        self.trades = trades
        print(f"📊 完成回測，共執行 {len(trades)} 筆交易")
        
        return trades
    
    def calculate_performance(self):
        """計算績效"""
        if not self.trades:
            return None
        
        df_trades = pd.DataFrame(self.trades)
        
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = df_trades['pnl'].sum()
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # 最大回撤
        cumulative_pnl = df_trades['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        performance = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf'),
            'max_drawdown': max_drawdown,
            'final_capital': df_trades['capital'].iloc[-1] if len(df_trades) > 0 else 100000
        }
        
        return performance
    
    def print_results(self):
        """列印結果"""
        print("\n" + "="*60)
        print("📊 TradingView多時間框架SMC交易結果")
        print("="*60)
        
        performance = self.calculate_performance()
        
        if performance:
            print(f"總交易次數: {performance['total_trades']}")
            print(f"獲利交易: {performance['winning_trades']}")
            print(f"虧損交易: {performance['losing_trades']}")
            print(f"勝率: {performance['win_rate']:.2%}")
            print(f"總損益: ${performance['total_pnl']:,.2f}")
            print(f"平均獲利: ${performance['avg_win']:,.2f}")
            print(f"平均虧損: ${performance['avg_loss']:,.2f}")
            print(f"獲利因子: {performance['profit_factor']:.2f}")
            print(f"最大回撤: {performance['max_drawdown']:.2f}%")
            print(f"最終資金: ${performance['final_capital']:,.2f}")
        else:
            print("無交易記錄")
        
        print("="*60)
    
    def plot_results(self):
        """繪製結果"""
        if not self.trades or '5m' not in self.data:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 價格圖
        df_5m = self.data['5m']
        ax1.plot(df_5m.index, df_5m['Close'], label='收盤價', alpha=0.7)
        
        # 交易點位
        df_trades = pd.DataFrame(self.trades)
        if len(df_trades) > 0:
            entry_times = pd.to_datetime(df_trades['entry_time'])
            entry_prices = df_trades['entry_price']
            colors = ['green' if d == 'long' else 'red' for d in df_trades['direction']]
            
            ax1.scatter(entry_times, entry_prices, color=colors, marker='o', s=50, label='交易進場')
            
            exit_times = pd.to_datetime(df_trades['exit_time'])
            exit_prices = df_trades['exit_price']
            ax1.scatter(exit_times, exit_prices, color='black', marker='x', s=50, label='交易出場')
        
        ax1.set_title(f'{self.symbol} TradingView多時間框架SMC交易結果', fontsize=14, fontweight='bold')
        ax1.set_ylabel('價格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 損益圖
        if len(df_trades) > 0:
            cumulative_pnl = df_trades['pnl'].cumsum()
            exit_times = pd.to_datetime(df_trades['exit_time'])
            
            ax2.plot(exit_times, cumulative_pnl, label='累積損益', linewidth=2)
            ax2.fill_between(exit_times, cumulative_pnl, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_title('累積損益曲線', fontsize=12, fontweight='bold')
            ax2.set_ylabel('損益')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.symbol}_tradingview_multi_timeframe_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """儲存結果"""
        print("\n💾 儲存結果...")
        
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            df_trades.to_excel(f'{self.symbol}_tradingview_multi_timeframe_trades.xlsx', index=False)
            print(f"✅ 交易記錄已儲存")
        
        if self.signals:
            df_signals = pd.DataFrame(self.signals)
            df_signals.to_excel(f'{self.symbol}_tradingview_multi_timeframe_signals.xlsx', index=False)
            print(f"✅ 訊號記錄已儲存")
    
    def run_analysis(self):
        """執行完整分析"""
        print("🚀 開始TradingView多時間框架SMC交易系統分析...")
        print("="*60)
        
        # 1. 獲取資料
        self.fetch_data()
        
        # 2. 生成訊號
        self.generate_signals()
        
        # 3. 執行回測
        if self.signals:
            self.backtest()
            
            # 4. 顯示結果
            self.print_results()
            self.plot_results()
            self.save_results()
        else:
            print("⚠️ 未生成任何交易訊號")

# 主程式執行
if __name__ == "__main__":
    # 建立交易系統實例
    trading_system = TradingViewMultiTimeframeSMC(
        symbol='AAPL',
        start_date='2024-01-01',
        end_date='2024-06-01'
    )
    
    # 執行完整分析
    trading_system.run_analysis()
    perf = trading_system.calculate_performance()
    print(perf['win_rate']) 