"""
量化交易系統 - 基於訂單塊、破壞塊、FVG和流動性獵取
使用yfinance資料進行回測
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import time

warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (15, 10)
sns.set_palette("husl")

class OrderBlockDetector:
    """訂單塊檢測器"""
    
    def __init__(self, lookback_period=20, min_block_size=0.5):
        self.lookback_period = lookback_period
        self.min_block_size = min_block_size
    
    def detect_bullish_order_blocks(self, df):
        """檢測看漲訂單塊"""
        bullish_blocks = []
        
        for i in range(self.lookback_period, len(df) - 1):
            # 尋找強烈的看漲蠟燭
            current_candle = df.iloc[i]
            next_candle = df.iloc[i + 1]
            
            # 條件：當前蠟燭收盤價高於開盤價，且下一個蠟燭突破當前蠟燭高點
            if (current_candle['Close'] > current_candle['Open'] and 
                next_candle['High'] > current_candle['High']):
                
                # 檢查回測是否在訂單塊範圍內
                for j in range(i + 2, min(i + 20, len(df))):
                    test_candle = df.iloc[j]
                    
                    # 如果價格回測到訂單塊範圍內
                    if (test_candle['Low'] <= current_candle['High'] and 
                        test_candle['High'] >= current_candle['Low']):
                        
                        block_size = current_candle['High'] - current_candle['Low']
                        if block_size >= self.min_block_size:
                            bullish_blocks.append({
                                'start_idx': i,
                                'end_idx': j,
                                'high': current_candle['High'],
                                'low': current_candle['Low'],
                                'strength': block_size,
                                'type': 'bullish'
                            })
                        break
        
        return bullish_blocks
    
    def detect_bearish_order_blocks(self, df):
        """檢測看跌訂單塊"""
        bearish_blocks = []
        
        for i in range(self.lookback_period, len(df) - 1):
            # 尋找強烈的看跌蠟燭
            current_candle = df.iloc[i]
            next_candle = df.iloc[i + 1]
            
            # 條件：當前蠟燭收盤價低於開盤價，且下一個蠟燭跌破當前蠟燭低點
            if (current_candle['Close'] < current_candle['Open'] and 
                next_candle['Low'] < current_candle['Low']):
                
                # 檢查回測是否在訂單塊範圍內
                for j in range(i + 2, min(i + 20, len(df))):
                    test_candle = df.iloc[j]
                    
                    # 如果價格回測到訂單塊範圍內
                    if (test_candle['Low'] <= current_candle['High'] and 
                        test_candle['High'] >= current_candle['Low']):
                        
                        block_size = current_candle['High'] - current_candle['Low']
                        if block_size >= self.min_block_size:
                            bearish_blocks.append({
                                'start_idx': i,
                                'end_idx': j,
                                'high': current_candle['High'],
                                'low': current_candle['Low'],
                                'strength': block_size,
                                'type': 'bearish'
                            })
                        break
        
        return bearish_blocks

class BreakOfStructureDetector:
    """破壞塊檢測器"""
    
    def __init__(self, swing_period=10):
        self.swing_period = swing_period
    
    def detect_swing_highs_lows(self, df):
        """檢測擺動高點和低點"""
        swing_highs = []
        swing_lows = []
        
        for i in range(self.swing_period, len(df) - self.swing_period):
            current_high = df.iloc[i]['High']
            current_low = df.iloc[i]['Low']
            
            # 檢查是否為擺動高點
            is_swing_high = True
            for j in range(i - self.swing_period, i + self.swing_period + 1):
                if j != i and df.iloc[j]['High'] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append({
                    'idx': i,
                    'price': current_high,
                    'datetime': df.iloc[i]['Datetime']
                })
            
            # 檢查是否為擺動低點
            is_swing_low = True
            for j in range(i - self.swing_period, i + self.swing_period + 1):
                if j != i and df.iloc[j]['Low'] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append({
                    'idx': i,
                    'price': current_low,
                    'datetime': df.iloc[i]['Datetime']
                })
        
        return swing_highs, swing_lows
    
    def detect_bos(self, df):
        """檢測破壞塊"""
        swing_highs, swing_lows = self.detect_swing_highs_lows(df)
        
        bos_events = []
        
        # 檢測向上破壞
        for i, swing_high in enumerate(swing_highs[:-1]):
            next_swing_high = swing_highs[i + 1]
            
            # 如果下一個擺動高點高於當前擺動高點，且中間有回調
            if next_swing_high['price'] > swing_high['price']:
                # 尋找中間的最低點
                min_low = float('inf')
                min_idx = None
                
                for j in range(swing_high['idx'], next_swing_high['idx']):
                    if df.iloc[j]['Low'] < min_low:
                        min_low = df.iloc[j]['Low']
                        min_idx = j
                
                if min_low < swing_high['price']:
                    bos_events.append({
                        'type': 'bullish_bos',
                        'break_level': swing_high['price'],
                        'break_idx': next_swing_high['idx'],
                        'pullback_low': min_low,
                        'pullback_idx': min_idx
                    })
        
        # 檢測向下破壞
        for i, swing_low in enumerate(swing_lows[:-1]):
            next_swing_low = swing_lows[i + 1]
            
            # 如果下一個擺動低點低於當前擺動低點，且中間有反彈
            if next_swing_low['price'] < swing_low['price']:
                # 尋找中間的最高點
                max_high = float('-inf')
                max_idx = None
                
                for j in range(swing_low['idx'], next_swing_low['idx']):
                    if df.iloc[j]['High'] > max_high:
                        max_high = df.iloc[j]['High']
                        max_idx = j
                
                if max_high > swing_low['price']:
                    bos_events.append({
                        'type': 'bearish_bos',
                        'break_level': swing_low['price'],
                        'break_idx': next_swing_low['idx'],
                        'pullback_high': max_high,
                        'pullback_idx': max_idx
                    })
        
        return bos_events

class FVGDetector:
    """公平價值缺口檢測器"""
    
    def __init__(self, min_gap_size=0.1):
        self.min_gap_size = min_gap_size
    
    def detect_fvg(self, df):
        """檢測公平價值缺口"""
        fvgs = []
        
        for i in range(1, len(df) - 1):
            current_candle = df.iloc[i]
            prev_candle = df.iloc[i - 1]
            next_candle = df.iloc[i + 1]
            
            # 檢測看漲FVG
            if (prev_candle['Low'] > current_candle['High'] and
                next_candle['High'] > prev_candle['Low']):
                
                gap_size = prev_candle['Low'] - current_candle['High']
                if gap_size >= self.min_gap_size:
                    fvgs.append({
                        'type': 'bullish_fvg',
                        'idx': i,
                        'top': prev_candle['Low'],
                        'bottom': current_candle['High'],
                        'size': gap_size,
                        'filled': False
                    })
            
            # 檢測看跌FVG
            elif (current_candle['Low'] > prev_candle['High'] and
                  next_candle['Low'] < current_candle['High']):
                
                gap_size = current_candle['Low'] - prev_candle['High']
                if gap_size >= self.min_gap_size:
                    fvgs.append({
                        'type': 'bearish_fvg',
                        'idx': i,
                        'top': current_candle['Low'],
                        'bottom': prev_candle['High'],
                        'size': gap_size,
                        'filled': False
                    })
        
        return fvgs
    
    def check_fvg_fill(self, df, fvgs):
        """檢查FVG是否被填補"""
        for fvg in fvgs:
            if fvg['filled']:
                continue
            
            for i in range(fvg['idx'] + 1, len(df)):
                candle = df.iloc[i]
                
                if fvg['type'] == 'bullish_fvg':
                    if candle['Low'] <= fvg['bottom']:
                        fvg['filled'] = True
                        fvg['fill_idx'] = i
                        break
                else:  # bearish_fvg
                    if candle['High'] >= fvg['top']:
                        fvg['filled'] = True
                        fvg['fill_idx'] = i
                        break
        
        return fvgs

class LiquidityGrabDetector:
    """流動性獵取檢測器"""
    
    def __init__(self, lookback_period=20, min_grab_size=0.3):
        self.lookback_period = lookback_period
        self.min_grab_size = min_grab_size
    
    def detect_liquidity_grabs(self, df):
        """檢測流動性獵取"""
        liquidity_grabs = []
        
        for i in range(self.lookback_period, len(df) - 1):
            current_candle = df.iloc[i]
            
            # 檢測上方流動性獵取（假突破高點）
            recent_highs = df.iloc[i-self.lookback_period:i]['High'].values
            if len(recent_highs) > 0:
                max_recent_high = np.max(recent_highs)
                
                # 如果當前蠟燭突破近期高點但收盤價回落
                if (current_candle['High'] > max_recent_high and 
                    current_candle['Close'] < max_recent_high):
                    
                    grab_size = current_candle['High'] - max_recent_high
                    if grab_size >= self.min_grab_size:
                        liquidity_grabs.append({
                            'type': 'upper_liquidity_grab',
                            'idx': i,
                            'grab_level': max_recent_high,
                            'grab_size': grab_size,
                            'direction': 'bearish'  # 假突破通常看跌
                        })
            
            # 檢測下方流動性獵取（假跌破低點）
            recent_lows = df.iloc[i-self.lookback_period:i]['Low'].values
            if len(recent_lows) > 0:
                min_recent_low = np.min(recent_lows)
                
                # 如果當前蠟燭跌破近期低點但收盤價反彈
                if (current_candle['Low'] < min_recent_low and 
                    current_candle['Close'] > min_recent_low):
                    
                    grab_size = min_recent_low - current_candle['Low']
                    if grab_size >= self.min_grab_size:
                        liquidity_grabs.append({
                            'type': 'lower_liquidity_grab',
                            'idx': i,
                            'grab_level': min_recent_low,
                            'grab_size': grab_size,
                            'direction': 'bullish'  # 假跌破通常看漲
                        })
        
        return liquidity_grabs

class QuantTradingSystem:
    """量化交易系統主類"""
    
    def __init__(self, symbol='AAPL', start_date='2023-01-01', end_date='2024-01-01'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.df = None
        self.trades = []
        
        # 初始化檢測器
        self.order_block_detector = OrderBlockDetector()
        self.bos_detector = BreakOfStructureDetector()
        self.fvg_detector = FVGDetector()
        self.liquidity_grab_detector = LiquidityGrabDetector()
    
    def fetch_data(self):
        """獲取yfinance資料"""
        print(f"📊 正在獲取 {self.symbol} 的資料...")
        
        ticker = yf.Ticker(self.symbol)
        self.df = ticker.history(start=self.start_date, end=self.end_date, interval='1d')
        
        # 重置索引並添加Datetime列
        self.df = self.df.reset_index()
        self.df['Datetime'] = self.df['Date']
        
        print(f"✅ 成功獲取 {len(self.df)} 筆資料")
        print(f"📅 資料期間: {self.df['Date'].min()} 到 {self.df['Date'].max()}")
        
        return self.df
    
    def analyze_market_structure(self):
        """分析市場結構"""
        print("\n🔍 分析市場結構...")
        
        # 檢測訂單塊
        bullish_ob = self.order_block_detector.detect_bullish_order_blocks(self.df)
        bearish_ob = self.order_block_detector.detect_bearish_order_blocks(self.df)
        
        # 檢測破壞塊
        bos_events = self.bos_detector.detect_bos(self.df)
        
        # 檢測FVG
        fvgs = self.fvg_detector.detect_fvg(self.df)
        fvgs = self.fvg_detector.check_fvg_fill(self.df, fvgs)
        
        # 檢測流動性獵取
        liquidity_grabs = self.liquidity_grab_detector.detect_liquidity_grabs(self.df)
        
        print(f"📈 檢測到 {len(bullish_ob)} 個看漲訂單塊")
        print(f"📉 檢測到 {len(bearish_ob)} 個看跌訂單塊")
        print(f"🔄 檢測到 {len(bos_events)} 個破壞塊事件")
        print(f"⚡ 檢測到 {len(fvgs)} 個FVG")
        print(f"🎯 檢測到 {len(liquidity_grabs)} 個流動性獵取")
        
        return {
            'bullish_ob': bullish_ob,
            'bearish_ob': bearish_ob,
            'bos_events': bos_events,
            'fvgs': fvgs,
            'liquidity_grabs': liquidity_grabs
        }
    
    def generate_signals(self, market_structure):
        """生成交易訊號"""
        print("\n🎯 生成交易訊號...")
        
        signals = []
        
        # 基於訂單塊的訊號
        for ob in market_structure['bullish_ob']:
            if ob['end_idx'] < len(self.df) - 1:
                signals.append({
                    'type': 'order_block_bullish',
                    'idx': ob['end_idx'],
                    'price': self.df.iloc[ob['end_idx']]['Close'],
                    'strength': ob['strength'],
                    'direction': 'long',
                    'reason': f"看漲訂單塊回測完成，強度: {ob['strength']:.2f}"
                })
        
        for ob in market_structure['bearish_ob']:
            if ob['end_idx'] < len(self.df) - 1:
                signals.append({
                    'type': 'order_block_bearish',
                    'idx': ob['end_idx'],
                    'price': self.df.iloc[ob['end_idx']]['Close'],
                    'strength': ob['strength'],
                    'direction': 'short',
                    'reason': f"看跌訂單塊回測完成，強度: {ob['strength']:.2f}"
                })
        
        # 基於破壞塊的訊號
        for bos in market_structure['bos_events']:
            if bos['type'] == 'bullish_bos' and bos['break_idx'] < len(self.df) - 1:
                signals.append({
                    'type': 'bos_bullish',
                    'idx': bos['break_idx'],
                    'price': self.df.iloc[bos['break_idx']]['Close'],
                    'strength': 1.0,
                    'direction': 'long',
                    'reason': f"向上破壞塊確認，突破位: {bos['break_level']:.2f}"
                })
            elif bos['type'] == 'bearish_bos' and bos['break_idx'] < len(self.df) - 1:
                signals.append({
                    'type': 'bos_bearish',
                    'idx': bos['break_idx'],
                    'price': self.df.iloc[bos['break_idx']]['Close'],
                    'strength': 1.0,
                    'direction': 'short',
                    'reason': f"向下破壞塊確認，突破位: {bos['break_level']:.2f}"
                })
        
        # 基於FVG的訊號
        for fvg in market_structure['fvgs']:
            if not fvg['filled'] and fvg['idx'] < len(self.df) - 1:
                if fvg['type'] == 'bullish_fvg':
                    signals.append({
                        'type': 'fvg_bullish',
                        'idx': fvg['idx'],
                        'price': self.df.iloc[fvg['idx']]['Close'],
                        'strength': fvg['size'],
                        'direction': 'long',
                        'reason': f"看漲FVG形成，缺口大小: {fvg['size']:.2f}"
                    })
                else:
                    signals.append({
                        'type': 'fvg_bearish',
                        'idx': fvg['idx'],
                        'price': self.df.iloc[fvg['idx']]['Close'],
                        'strength': fvg['size'],
                        'direction': 'short',
                        'reason': f"看跌FVG形成，缺口大小: {fvg['size']:.2f}"
                    })
        
        # 基於流動性獵取的訊號
        for lg in market_structure['liquidity_grabs']:
            if lg['idx'] < len(self.df) - 1:
                signals.append({
                    'type': 'liquidity_grab',
                    'idx': lg['idx'],
                    'price': self.df.iloc[lg['idx']]['Close'],
                    'strength': lg['grab_size'],
                    'direction': lg['direction'],
                    'reason': f"流動性獵取完成，獵取大小: {lg['grab_size']:.2f}"
                })
        
        # 按時間排序
        signals.sort(key=lambda x: x['idx'])
        
        print(f"📊 生成 {len(signals)} 個交易訊號")
        
        return signals
    
    def backtest(self, signals, initial_capital=100000):
        """回測交易策略"""
        print("\n📈 開始回測...")
        
        capital = initial_capital
        position = None
        trades = []
        
        for i, signal in enumerate(signals):
            current_price = signal['price']
            current_idx = signal['idx']
            
            # 如果沒有持倉，考慮進場
            if position is None:
                # 計算倉位大小（基於風險管理）
                risk_per_trade = capital * 0.02  # 每筆交易風險2%
                stop_loss_pct = 0.05  # 5%停損
                
                if signal['direction'] == 'long':
                    stop_loss_price = current_price * (1 - stop_loss_pct)
                    position_size = risk_per_trade / (current_price - stop_loss_price)
                    
                    position = {
                        'type': 'long',
                        'entry_price': current_price,
                        'entry_idx': current_idx,
                        'entry_time': self.df.iloc[current_idx]['Date'],
                        'position_size': position_size,
                        'stop_loss': stop_loss_price,
                        'signal_strength': signal['strength'],
                        'signal_type': signal['type'],
                        'reason': signal['reason']
                    }
                
                elif signal['direction'] == 'short':
                    stop_loss_price = current_price * (1 + stop_loss_pct)
                    position_size = risk_per_trade / (stop_loss_price - current_price)
                    
                    position = {
                        'type': 'short',
                        'entry_price': current_price,
                        'entry_idx': current_idx,
                        'entry_time': self.df.iloc[current_idx]['Date'],
                        'position_size': position_size,
                        'stop_loss': stop_loss_price,
                        'signal_strength': signal['strength'],
                        'signal_type': signal['type'],
                        'reason': signal['reason']
                    }
            
            # 檢查是否需要平倉
            elif position is not None:
                exit_reason = None
                exit_price = None
                
                # 檢查停損
                if position['type'] == 'long' and current_price <= position['stop_loss']:
                    exit_reason = 'stop_loss'
                    exit_price = position['stop_loss']
                elif position['type'] == 'short' and current_price >= position['stop_loss']:
                    exit_reason = 'stop_loss'
                    exit_price = position['stop_loss']
                
                # 檢查獲利了結（當價格移動2倍停損距離時）
                elif position['type'] == 'long':
                    profit_target = position['entry_price'] + (position['entry_price'] - position['stop_loss']) * 2
                    if current_price >= profit_target:
                        exit_reason = 'take_profit'
                        exit_price = current_price
                
                elif position['type'] == 'short':
                    profit_target = position['entry_price'] - (position['stop_loss'] - position['entry_price']) * 2
                    if current_price <= profit_target:
                        exit_reason = 'take_profit'
                        exit_price = current_price
                
                # 檢查時間停損（持倉超過20個交易日）
                elif current_idx - position['entry_idx'] > 20:
                    exit_reason = 'time_stop'
                    exit_price = current_price
                
                # 平倉
                if exit_reason:
                    if position['type'] == 'long':
                        pnl = (exit_price - position['entry_price']) * position['position_size']
                    else:
                        pnl = (position['entry_price'] - exit_price) * position['position_size']
                    
                    capital += pnl
                    
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': self.df.iloc[current_idx]['Date'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'direction': position['type'],
                        'position_size': position['position_size'],
                        'pnl': pnl,
                        'capital': capital,
                        'exit_reason': exit_reason,
                        'signal_type': position['signal_type'],
                        'signal_strength': position['signal_strength'],
                        'reason': position['reason']
                    }
                    
                    trades.append(trade)
                    position = None
        
        self.trades = trades
        print(f"📊 完成回測，共執行 {len(trades)} 筆交易")
        
        return trades
    
    def calculate_performance(self):
        """計算績效指標"""
        if not self.trades:
            return None
        
        df_trades = pd.DataFrame(self.trades)
        
        # 基本統計
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 損益統計
        total_pnl = df_trades['pnl'].sum()
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # 最大回撤
        cumulative_pnl = df_trades['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # 夏普比率（簡化版）
        returns = df_trades['pnl'] / 100000  # 假設初始資金10萬
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
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
            'sharpe_ratio': sharpe_ratio,
            'final_capital': df_trades['capital'].iloc[-1] if len(df_trades) > 0 else 100000
        }
        
        return performance
    
    def plot_results(self, market_structure, signals):
        """繪製結果圖表"""
        print("\n📊 繪製結果圖表...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 主圖：價格和訊號
        ax1.plot(self.df['Date'], self.df['Close'], label='收盤價', alpha=0.7)
        
        # 繪製訂單塊
        for ob in market_structure['bullish_ob']:
            ax1.axvspan(self.df.iloc[ob['start_idx']]['Date'], 
                       self.df.iloc[ob['end_idx']]['Date'], 
                       alpha=0.3, color='green', label='看漲訂單塊' if ob == market_structure['bullish_ob'][0] else "")
        
        for ob in market_structure['bearish_ob']:
            ax1.axvspan(self.df.iloc[ob['start_idx']]['Date'], 
                       self.df.iloc[ob['end_idx']]['Date'], 
                       alpha=0.3, color='red', label='看跌訂單塊' if ob == market_structure['bearish_ob'][0] else "")
        
        # 繪製交易訊號
        long_signals = [s for s in signals if s['direction'] == 'long']
        short_signals = [s for s in signals if s['direction'] == 'short']
        
        if long_signals:
            long_dates = [self.df.iloc[s['idx']]['Date'] for s in long_signals]
            long_prices = [s['price'] for s in long_signals]
            ax1.scatter(long_dates, long_prices, color='green', marker='^', s=100, label='做多訊號')
        
        if short_signals:
            short_dates = [self.df.iloc[s['idx']]['Date'] for s in short_signals]
            short_prices = [s['price'] for s in short_signals]
            ax1.scatter(short_dates, short_prices, color='red', marker='v', s=100, label='做空訊號')
        
        # 繪製交易結果
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            
            # 進場點
            entry_dates = pd.to_datetime(df_trades['entry_time'])
            entry_prices = df_trades['entry_price']
            colors = ['green' if d == 'long' else 'red' for d in df_trades['direction']]
            
            ax1.scatter(entry_dates, entry_prices, color=colors, marker='o', s=50, alpha=0.7, label='交易進場')
            
            # 出場點
            exit_dates = pd.to_datetime(df_trades['exit_time'])
            exit_prices = df_trades['exit_price']
            ax1.scatter(exit_dates, exit_prices, color='black', marker='x', s=50, alpha=0.7, label='交易出場')
        
        ax1.set_title(f'{self.symbol} 量化交易系統結果', fontsize=14, fontweight='bold')
        ax1.set_ylabel('價格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子圖：累積損益
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            cumulative_pnl = df_trades['pnl'].cumsum()
            exit_dates = pd.to_datetime(df_trades['exit_time'])
            
            ax2.plot(exit_dates, cumulative_pnl, label='累積損益', linewidth=2)
            ax2.fill_between(exit_dates, cumulative_pnl, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_title('累積損益曲線', fontsize=12, fontweight='bold')
            ax2.set_ylabel('損益')
            ax2.set_xlabel('日期')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.symbol}_trading_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_performance_summary(self, performance):
        """列印績效摘要"""
        print("\n" + "="*60)
        print("📊 交易績效摘要")
        print("="*60)
        
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
            print(f"夏普比率: {performance['sharpe_ratio']:.2f}")
            print(f"最終資金: ${performance['final_capital']:,.2f}")
        else:
            print("無交易記錄")
        
        print("="*60)
    
    def run_complete_analysis(self):
        """執行完整分析"""
        print("🚀 開始量化交易系統分析...")
        print("="*60)
        
        # 1. 獲取資料
        self.fetch_data()
        
        # 2. 分析市場結構
        market_structure = self.analyze_market_structure()
        
        # 3. 生成訊號
        signals = self.generate_signals(market_structure)
        
        # 4. 回測
        if signals:
            self.backtest(signals)
            
            # 5. 計算績效
            performance = self.calculate_performance()
            
            # 6. 列印結果
            self.print_performance_summary(performance)
            
            # 7. 繪製圖表
            self.plot_results(market_structure, signals)
            
            # 8. 儲存結果
            self.save_results(performance)
        else:
            print("⚠️ 未生成任何交易訊號")
    
    def save_results(self, performance):
        """儲存結果到檔案"""
        print("\n💾 儲存結果...")
        
        # 儲存交易記錄
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            df_trades.to_excel(f'{self.symbol}_trades.xlsx', index=False)
            print(f"✅ 交易記錄已儲存至 {self.symbol}_trades.xlsx")
        
        # 儲存績效摘要
        if performance:
            performance_df = pd.DataFrame([performance])
            performance_df.to_excel(f'{self.symbol}_performance.xlsx', index=False)
            print(f"✅ 績效摘要已儲存至 {self.symbol}_performance.xlsx")

# 主程式執行
if __name__ == "__main__":
    # 建立交易系統實例
    trading_system = QuantTradingSystem(
        symbol='AAPL',  # 可以更改為其他股票代碼
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    # 執行完整分析
    trading_system.run_complete_analysis()
