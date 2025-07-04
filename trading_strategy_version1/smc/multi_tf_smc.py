"""
多時間框架SMC短線交易系統
結合1分鐘、5分鐘、15分鐘時間框架進行短線交易
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (15, 10)

class MultiTimeframeDataManager:
    """多時間框架資料管理器"""
    
    def __init__(self, symbol: str, start_date: str, end_date: str):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.timeframes = {
            '1m': '1m',
            '5m': '5m', 
            '15m': '15m'
        }
        self.data = {}
    
    def fetch_multi_timeframe_data(self) -> Dict[str, pd.DataFrame]:
        """獲取多時間框架資料"""
        print(f"📊 正在獲取 {self.symbol} 的多時間框架資料...")
        
        ticker = yf.Ticker(self.symbol)
        
        for tf_name, tf_interval in self.timeframes.items():
            print(f"🔄 獲取 {tf_name} 時間框架資料...")
            
            try:
                # 獲取資料
                df = ticker.history(start=self.start_date, end=self.end_date, interval=tf_interval)
                
                if len(df) == 0:
                    print(f"⚠️ {tf_name} 時間框架無資料，跳過")
                    continue
                
                # 重置索引並添加Datetime列
                df = df.reset_index()
                df['Datetime'] = df['Date']
                
                # 添加技術指標
                df = self.add_technical_indicators(df)
                
                self.data[tf_name] = df
                
                print(f"✅ {tf_name} 時間框架: {len(df)} 筆資料")
                print(f"📅 期間: {df['Date'].min()} 到 {df['Date'].max()}")
                
            except Exception as e:
                print(f"❌ 獲取 {tf_name} 時間框架資料失敗: {e}")
        
        return self.data
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技術指標"""
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        
        # 移動平均線
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # 成交量指標
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # ATR (平均真實範圍)
        df['ATR'] = self.calculate_atr(df, 14)
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """計算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """計算ATR"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

class OrderBlockDetector:
    """訂單塊檢測器"""
    
    def __init__(self, lookback_period: int = 20, min_block_size: float = 0.3):
        self.lookback_period = lookback_period
        self.min_block_size = min_block_size
    
    def detect_order_blocks(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """檢測訂單塊"""
        bullish_blocks = []
        bearish_blocks = []
        
        for i in range(self.lookback_period, len(df) - 1):
            current_candle = df.iloc[i]
            next_candle = df.iloc[i + 1]
            
            # 看漲訂單塊條件
            if (current_candle['Close'] > current_candle['Open'] and 
                current_candle['Volume'] > current_candle['Volume_SMA'] * 1.2 and
                next_candle['High'] > current_candle['High']):
                
                # 檢查回測
                for j in range(i + 2, min(i + 15, len(df))):
                    test_candle = df.iloc[j]
                    
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
                                'volume_ratio': current_candle['Volume_Ratio'],
                                'type': 'bullish'
                            })
                        break
            
            # 看跌訂單塊條件
            elif (current_candle['Close'] < current_candle['Open'] and 
                  current_candle['Volume'] > current_candle['Volume_SMA'] * 1.2 and
                  next_candle['Low'] < current_candle['Low']):
                
                # 檢查回測
                for j in range(i + 2, min(i + 15, len(df))):
                    test_candle = df.iloc[j]
                    
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
                                'volume_ratio': current_candle['Volume_Ratio'],
                                'type': 'bearish'
                            })
                        break
        
        return bullish_blocks, bearish_blocks

class FVGDetector:
    """FVG檢測器"""
    
    def __init__(self, min_gap_size: float = 0.1):
        self.min_gap_size = min_gap_size
    
    def detect_fvgs(self, df: pd.DataFrame) -> List[Dict]:
        """檢測FVG"""
        fvgs = []
        
        for i in range(1, len(df) - 1):
            current_candle = df.iloc[i]
            prev_candle = df.iloc[i - 1]
            next_candle = df.iloc[i + 1]
            
            # 看漲FVG
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
                        'filled': False,
                        'volume_ratio': current_candle['Volume_Ratio']
                    })
            
            # 看跌FVG
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
                        'filled': False,
                        'volume_ratio': current_candle['Volume_Ratio']
                    })
        
        return fvgs

class LiquidityGrabDetector:
    """流動性獵取檢測器"""
    
    def __init__(self, lookback_period: int = 15, min_grab_size: float = 0.2):
        self.lookback_period = lookback_period
        self.min_grab_size = min_grab_size
    
    def detect_liquidity_grabs(self, df: pd.DataFrame) -> List[Dict]:
        """檢測流動性獵取"""
        liquidity_grabs = []
        
        for i in range(self.lookback_period, len(df) - 1):
            current_candle = df.iloc[i]
            
            # 上方流動性獵取
            recent_highs = df.iloc[i-self.lookback_period:i]['High'].values
            if len(recent_highs) > 0:
                max_recent_high = np.max(recent_highs)
                
                if (current_candle['High'] > max_recent_high and 
                    current_candle['Close'] < max_recent_high and
                    current_candle['Volume'] > current_candle['Volume_SMA'] * 1.1):
                    
                    grab_size = current_candle['High'] - max_recent_high
                    if grab_size >= self.min_grab_size:
                        liquidity_grabs.append({
                            'type': 'upper_liquidity_grab',
                            'idx': i,
                            'grab_level': max_recent_high,
                            'grab_size': grab_size,
                            'direction': 'bearish',
                            'volume_ratio': current_candle['Volume_Ratio']
                        })
            
            # 下方流動性獵取
            recent_lows = df.iloc[i-self.lookback_period:i]['Low'].values
            if len(recent_lows) > 0:
                min_recent_low = np.min(recent_lows)
                
                if (current_candle['Low'] < min_recent_low and 
                    current_candle['Close'] > min_recent_low and
                    current_candle['Volume'] > current_candle['Volume_SMA'] * 1.1):
                    
                    grab_size = min_recent_low - current_candle['Low']
                    if grab_size >= self.min_grab_size:
                        liquidity_grabs.append({
                            'type': 'lower_liquidity_grab',
                            'idx': i,
                            'grab_level': min_recent_low,
                            'grab_size': grab_size,
                            'direction': 'bullish',
                            'volume_ratio': current_candle['Volume_Ratio']
                        })
        
        return liquidity_grabs

class MultiTimeframeSignalGenerator:
    """多時間框架訊號生成器"""
    
    def __init__(self):
        self.order_block_detector = OrderBlockDetector()
        self.fvg_detector = FVGDetector()
        self.liquidity_grab_detector = LiquidityGrabDetector()
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """生成多時間框架訊號"""
        print("\n🎯 生成多時間框架交易訊號...")
        
        all_signals = []
        
        # 對每個時間框架生成訊號
        for tf_name, df in data.items():
            print(f"📊 分析 {tf_name} 時間框架...")
            
            # 檢測訂單塊
            bullish_ob, bearish_ob = self.order_block_detector.detect_order_blocks(df)
            
            # 檢測FVG
            fvgs = self.fvg_detector.detect_fvgs(df)
            
            # 檢測流動性獵取
            liquidity_grabs = self.liquidity_grab_detector.detect_liquidity_grabs(df)
            
            # 生成訊號
            tf_signals = self._create_timeframe_signals(
                df, tf_name, bullish_ob, bearish_ob, fvgs, liquidity_grabs
            )
            
            all_signals.extend(tf_signals)
            print(f"✅ {tf_name} 時間框架: {len(tf_signals)} 個訊號")
        
        # 按時間排序
        all_signals.sort(key=lambda x: x['datetime'])
        
        # 過濾和優化訊號
        filtered_signals = self._filter_signals(all_signals)
        
        print(f"📊 總共生成 {len(filtered_signals)} 個有效訊號")
        
        return filtered_signals
    
    def _create_timeframe_signals(self, df: pd.DataFrame, tf_name: str, 
                                 bullish_ob: List[Dict], bearish_ob: List[Dict],
                                 fvgs: List[Dict], liquidity_grabs: List[Dict]) -> List[Dict]:
        """為單個時間框架創建訊號"""
        signals = []
        
        # 訂單塊訊號
        for ob in bullish_ob:
            if ob['end_idx'] < len(df) - 1:
                signals.append({
                    'datetime': df.iloc[ob['end_idx']]['Datetime'],
                    'type': 'order_block_bullish',
                    'timeframe': tf_name,
                    'price': df.iloc[ob['end_idx']]['Close'],
                    'strength': ob['strength'],
                    'volume_ratio': ob['volume_ratio'],
                    'direction': 'long',
                    'reason': f"{tf_name}看漲訂單塊回測完成",
                    'rsi': df.iloc[ob['end_idx']]['RSI'],
                    'atr': df.iloc[ob['end_idx']]['ATR']
                })
        
        for ob in bearish_ob:
            if ob['end_idx'] < len(df) - 1:
                signals.append({
                    'datetime': df.iloc[ob['end_idx']]['Datetime'],
                    'type': 'order_block_bearish',
                    'timeframe': tf_name,
                    'price': df.iloc[ob['end_idx']]['Close'],
                    'strength': ob['strength'],
                    'volume_ratio': ob['volume_ratio'],
                    'direction': 'short',
                    'reason': f"{tf_name}看跌訂單塊回測完成",
                    'rsi': df.iloc[ob['end_idx']]['RSI'],
                    'atr': df.iloc[ob['end_idx']]['ATR']
                })
        
        # FVG訊號
        for fvg in fvgs:
            if not fvg['filled'] and fvg['idx'] < len(df) - 1:
                if fvg['type'] == 'bullish_fvg':
                    signals.append({
                        'datetime': df.iloc[fvg['idx']]['Datetime'],
                        'type': 'fvg_bullish',
                        'timeframe': tf_name,
                        'price': df.iloc[fvg['idx']]['Close'],
                        'strength': fvg['size'],
                        'volume_ratio': fvg['volume_ratio'],
                        'direction': 'long',
                        'reason': f"{tf_name}看漲FVG形成",
                        'rsi': df.iloc[fvg['idx']]['RSI'],
                        'atr': df.iloc[fvg['idx']]['ATR']
                    })
                else:
                    signals.append({
                        'datetime': df.iloc[fvg['idx']]['Datetime'],
                        'type': 'fvg_bearish',
                        'timeframe': tf_name,
                        'price': df.iloc[fvg['idx']]['Close'],
                        'strength': fvg['size'],
                        'volume_ratio': fvg['volume_ratio'],
                        'direction': 'short',
                        'reason': f"{tf_name}看跌FVG形成",
                        'rsi': df.iloc[fvg['idx']]['RSI'],
                        'atr': df.iloc[fvg['idx']]['ATR']
                    })
        
        # 流動性獵取訊號
        for lg in liquidity_grabs:
            if lg['idx'] < len(df) - 1:
                signals.append({
                    'datetime': df.iloc[lg['idx']]['Datetime'],
                    'type': 'liquidity_grab',
                    'timeframe': tf_name,
                    'price': df.iloc[lg['idx']]['Close'],
                    'strength': lg['grab_size'],
                    'volume_ratio': lg['volume_ratio'],
                    'direction': lg['direction'],
                    'reason': f"{tf_name}流動性獵取完成",
                    'rsi': df.iloc[lg['idx']]['RSI'],
                    'atr': df.iloc[lg['idx']]['ATR']
                })
        
        return signals
    
    def _filter_signals(self, signals: List[Dict]) -> List[Dict]:
        """過濾和優化訊號"""
        filtered_signals = []
        
        for signal in signals:
            # 基本過濾條件
            if (pd.notna(signal['rsi']) and 
                pd.notna(signal['atr']) and 
                pd.notna(signal['volume_ratio'])):
                
                # RSI過濾
                if signal['direction'] == 'long' and signal['rsi'] < 70:
                    if signal['direction'] == 'short' and signal['rsi'] > 30:
                        continue
                
                # 成交量過濾
                if signal['volume_ratio'] < 1.0:
                    continue
                
                # 時間框架權重
                timeframe_weights = {'1m': 0.3, '5m': 0.5, '15m': 0.8}
                signal['weight'] = timeframe_weights.get(signal['timeframe'], 0.5)
                
                # 綜合強度計算
                signal['composite_strength'] = (
                    signal['strength'] * signal['weight'] * signal['volume_ratio']
                )
                
                filtered_signals.append(signal)
        
        return filtered_signals

class ShortTermTradingSystem:
    """短線交易系統"""
    
    def __init__(self, symbol: str = 'AAPL', start_date: str = '2024-01-01', 
                 end_date: str = '2024-06-01', initial_capital: float = 100000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        # 初始化組件
        self.data_manager = MultiTimeframeDataManager(symbol, start_date, end_date)
        self.signal_generator = MultiTimeframeSignalGenerator()
        
        self.data = {}
        self.signals = []
        self.trades = []
    
    def run_analysis(self):
        """執行完整分析"""
        print("🚀 開始多時間框架SMC短線交易系統分析...")
        print("="*60)
        
        # 1. 獲取多時間框架資料
        self.data = self.data_manager.fetch_multi_timeframe_data()
        
        if not self.data:
            print("❌ 無法獲取任何時間框架資料")
            return
        
        # 2. 生成訊號
        self.signals = self.signal_generator.generate_signals(self.data)
        
        # 3. 執行回測
        if self.signals:
            self.backtest()
            
            # 4. 計算績效
            performance = self.calculate_performance()
            
            # 5. 顯示結果
            self.print_performance_summary(performance)
            self.plot_results()
            self.save_results(performance)
        else:
            print("⚠️ 未生成任何交易訊號")
    
    def backtest(self):
        """執行回測"""
        print("\n📈 開始短線交易回測...")
        
        capital = self.initial_capital
        position = None
        trades = []
        
        # 使用1分鐘資料進行回測
        if '1m' not in self.data:
            print("❌ 缺少1分鐘資料，無法進行回測")
            return
        
        df_1m = self.data['1m']
        
        for signal in self.signals:
            # 找到對應的1分鐘資料索引
            signal_time = pd.to_datetime(signal['datetime'])
            df_1m['Datetime'] = pd.to_datetime(df_1m['Datetime'])
            
            # 找到最接近的時間點
            time_diff = abs(df_1m['Datetime'] - signal_time)
            current_idx = time_diff.idxmin()
            
            if current_idx >= len(df_1m) - 1:
                continue
            
            current_price = df_1m.iloc[current_idx]['Close']
            
            # 如果沒有持倉，考慮進場
            if position is None:
                # 短線交易風險管理
                risk_per_trade = capital * 0.01  # 每筆交易風險1%
                atr = signal['atr']
                stop_loss_pct = min(0.03, atr / current_price * 2)  # 基於ATR的動態停損
                
                if signal['direction'] == 'long':
                    stop_loss_price = current_price * (1 - stop_loss_pct)
                    position_size = risk_per_trade / (current_price - stop_loss_price)
                    
                    position = {
                        'type': 'long',
                        'entry_price': current_price,
                        'entry_time': signal_time,
                        'entry_idx': current_idx,
                        'position_size': position_size,
                        'stop_loss': stop_loss_price,
                        'take_profit': current_price * (1 + stop_loss_pct * 2),  # 2:1風險報酬比
                        'signal': signal,
                        'time_stop': signal_time + timedelta(minutes=30)  # 30分鐘時間停損
                    }
                
                elif signal['direction'] == 'short':
                    stop_loss_price = current_price * (1 + stop_loss_pct)
                    position_size = risk_per_trade / (stop_loss_price - current_price)
                    
                    position = {
                        'type': 'short',
                        'entry_price': current_price,
                        'entry_time': signal_time,
                        'entry_idx': current_idx,
                        'position_size': position_size,
                        'stop_loss': stop_loss_price,
                        'take_profit': current_price * (1 - stop_loss_pct * 2),
                        'signal': signal,
                        'time_stop': signal_time + timedelta(minutes=30)
                    }
            
            # 檢查是否需要平倉
            elif position is not None:
                exit_reason = None
                exit_price = None
                current_time = signal_time
                
                # 檢查停損
                if position['type'] == 'long' and current_price <= position['stop_loss']:
                    exit_reason = 'stop_loss'
                    exit_price = position['stop_loss']
                elif position['type'] == 'short' and current_price >= position['stop_loss']:
                    exit_reason = 'stop_loss'
                    exit_price = position['stop_loss']
                
                # 檢查獲利了結
                elif position['type'] == 'long' and current_price >= position['take_profit']:
                    exit_reason = 'take_profit'
                    exit_price = current_price
                elif position['type'] == 'short' and current_price <= position['take_profit']:
                    exit_reason = 'take_profit'
                    exit_price = current_price
                
                # 檢查時間停損
                elif current_time >= position['time_stop']:
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
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'direction': position['type'],
                        'position_size': position['position_size'],
                        'pnl': pnl,
                        'capital': capital,
                        'exit_reason': exit_reason,
                        'signal_type': position['signal']['type'],
                        'timeframe': position['signal']['timeframe'],
                        'signal_strength': position['signal']['composite_strength'],
                        'reason': position['signal']['reason']
                    }
                    
                    trades.append(trade)
                    position = None
        
        self.trades = trades
        print(f"📊 完成回測，共執行 {len(trades)} 筆交易")
    
    def calculate_performance(self) -> Optional[Dict]:
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
        
        # 夏普比率
        returns = df_trades['pnl'] / self.initial_capital
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # 時間框架分析
        timeframe_stats = df_trades.groupby('timeframe').agg({
            'pnl': ['count', 'sum', 'mean'],
            'exit_reason': lambda x: (x == 'take_profit').sum() / len(x)
        }).round(4)
        
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
            'final_capital': df_trades['capital'].iloc[-1] if len(df_trades) > 0 else self.initial_capital,
            'timeframe_stats': timeframe_stats
        }
        
        return performance
    
    def print_performance_summary(self, performance: Optional[Dict]):
        """列印績效摘要"""
        print("\n" + "="*60)
        print("📊 多時間框架SMC短線交易績效摘要")
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
            
            print("\n📊 時間框架統計:")
            print(performance['timeframe_stats'])
        else:
            print("無交易記錄")
        
        print("="*60)
    
    def plot_results(self):
        """繪製結果圖表"""
        print("\n📊 繪製結果圖表...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 主圖：價格和訊號
        if '1m' in self.data:
            df_1m = self.data['1m']
            ax1.plot(df_1m['Datetime'], df_1m['Close'], label='收盤價', alpha=0.7)
            
            # 繪製交易訊號
            if self.signals:
                long_signals = [s for s in self.signals if s['direction'] == 'long']
                short_signals = [s for s in self.signals if s['direction'] == 'short']
                
                if long_signals:
                    long_times = [s['datetime'] for s in long_signals]
                    long_prices = [s['price'] for s in long_signals]
                    ax1.scatter(long_times, long_prices, color='green', marker='^', s=50, label='做多訊號')
                
                if short_signals:
                    short_times = [s['datetime'] for s in short_signals]
                    short_prices = [s['price'] for s in short_signals]
                    ax1.scatter(short_times, short_prices, color='red', marker='v', s=50, label='做空訊號')
            
            # 繪製交易結果
            if self.trades:
                df_trades = pd.DataFrame(self.trades)
                
                # 進場點
                entry_times = pd.to_datetime(df_trades['entry_time'])
                entry_prices = df_trades['entry_price']
                colors = ['green' if d == 'long' else 'red' for d in df_trades['direction']]
                
                ax1.scatter(entry_times, entry_prices, color=colors, marker='o', s=30, alpha=0.7, label='交易進場')
                
                # 出場點
                exit_times = pd.to_datetime(df_trades['exit_time'])
                exit_prices = df_trades['exit_price']
                ax1.scatter(exit_times, exit_prices, color='black', marker='x', s=30, alpha=0.7, label='交易出場')
        
        ax1.set_title(f'{self.symbol} 多時間框架SMC短線交易結果', fontsize=14, fontweight='bold')
        ax1.set_ylabel('價格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子圖2：累積損益
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            cumulative_pnl = df_trades['pnl'].cumsum()
            exit_times = pd.to_datetime(df_trades['exit_time'])
            
            ax2.plot(exit_times, cumulative_pnl, label='累積損益', linewidth=2)
            ax2.fill_between(exit_times, cumulative_pnl, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_title('累積損益曲線', fontsize=12, fontweight='bold')
            ax2.set_ylabel('損益')
            ax2.grid(True, alpha=0.3)
        
        # 子圖3：時間框架分析
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            timeframe_pnl = df_trades.groupby('timeframe')['pnl'].sum()
            
            ax3.bar(timeframe_pnl.index, timeframe_pnl.values, color=['blue', 'green', 'red'])
            ax3.set_title('各時間框架損益', fontsize=12, fontweight='bold')
            ax3.set_ylabel('損益')
            ax3.grid(True, alpha=0.3)
        
        # 子圖4：出場原因分析
        if self.trades:
            exit_reasons = df_trades['exit_reason'].value_counts()
            
            ax4.pie(exit_reasons.values, labels=exit_reasons.index, autopct='%1.1f%%')
            ax4.set_title('出場原因分布', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.symbol}_multi_timeframe_trading_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, performance: Optional[Dict]):
        """儲存結果"""
        print("\n💾 儲存結果...")
        
        # 儲存交易記錄
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            df_trades.to_excel(f'{self.symbol}_multi_timeframe_trades.xlsx', index=False)
            print(f"✅ 交易記錄已儲存至 {self.symbol}_multi_timeframe_trades.xlsx")
        
        # 儲存訊號記錄
        if self.signals:
            df_signals = pd.DataFrame(self.signals)
            df_signals.to_excel(f'{self.symbol}_multi_timeframe_signals.xlsx', index=False)
            print(f"✅ 訊號記錄已儲存至 {self.symbol}_multi_timeframe_signals.xlsx")
        
        # 儲存績效摘要
        if performance:
            # 移除無法序列化的DataFrame
            perf_copy = performance.copy()
            if 'timeframe_stats' in perf_copy:
                perf_copy['timeframe_stats'] = perf_copy['timeframe_stats'].to_dict()
            
            performance_df = pd.DataFrame([perf_copy])
            performance_df.to_excel(f'{self.symbol}_multi_timeframe_performance.xlsx', index=False)
            print(f"✅ 績效摘要已儲存至 {self.symbol}_multi_timeframe_performance.xlsx")

# 主程式執行
if __name__ == "__main__":
    # 建立短線交易系統實例
    trading_system = ShortTermTradingSystem(
        symbol='AAPL',  # 可以更改為其他股票代碼
        start_date='2024-01-01',
        end_date='2024-06-01'
    )
    
    # 執行完整分析
    trading_system.run_analysis() 