import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import itertools
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_txf_data(file_path: str) -> pd.DataFrame:
    """
    Load TXF 5-minute K data from txt format and convert to datetime index.
    
    Args:
        file_path: Path to the TXF data file
        
    Returns:
        DataFrame with datetime index and OHLCV data
    """
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'TotalVolume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']  # Standardize column names for technical indicators
    return df

def convert_to_4h_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 1-minute data to 4-hour (240-minute) data.
    
    Args:
        df: DataFrame with 1-minute OHLCV data
        
    Returns:
        DataFrame with 4-hour OHLCV data
    """
    df = df.copy()
    
    # Reset index to get datetime as column for grouping
    df = df.reset_index()
    
    # Create 4-hour periods
    # Group by date and 4-hour blocks (240 minutes)
    df['date'] = df['Datetime'].dt.date
    df['hour_block'] = df['Datetime'].dt.hour // 4  # 0-5 for 4-hour blocks
    df['period'] = df['date'].astype(str) + '_' + df['hour_block'].astype(str)
    
    # Group by period and aggregate
    grouped = df.groupby('period').agg({
        'Datetime': 'first',  # First datetime of the period
        'open': 'first',      # First open price
        'high': 'max',        # Maximum high price
        'low': 'min',         # Minimum low price
        'close': 'last',      # Last close price
        'volume': 'sum'       # Sum of volume
    })
    
    # Set datetime as index
    grouped = grouped.set_index('Datetime')
    
    # Sort by datetime
    grouped = grouped.sort_index()
    
    print(f"Converted {len(df)} 1-minute records to {len(grouped)} 4-hour records")
    print(f"Date range: {grouped.index[0]} to {grouped.index[-1]}")
    
    return grouped

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands indicators.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window for moving average
        num_std: Number of standard deviations for bands
        
    Returns:
        DataFrame with BB indicators added
    """
    df = df.copy()
    df['bb_middle'] = df['close'].rolling(window=window).mean()
    bb_std = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * num_std)
    df['bb_lower'] = df['bb_middle'] - (bb_std * num_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    return df

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate RSI indicator.
    
    Args:
        df: DataFrame with OHLCV data
        window: RSI calculation window
        
    Returns:
        DataFrame with RSI indicator added
    """
    df = df.copy()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate On-Balance Volume (OBV) indicator.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with OBV indicator added
    """
    df = df.copy()
    df['obv'] = 0.0
    df['obv_sma'] = 0.0
    
    # Calculate OBV
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            df['obv'].iloc[i] = df['obv'].iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            df['obv'].iloc[i] = df['obv'].iloc[i-1] - df['volume'].iloc[i]
        else:
            df['obv'].iloc[i] = df['obv'].iloc[i-1]
    
    # Calculate OBV moving average
    df['obv_sma'] = df['obv'].rolling(window=20).mean()
    df['obv_ratio'] = df['obv'] / df['obv_sma']
    
    return df

def generate_signals(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Generate trading signals based on BB + RSI + OBV strategy, with stop loss, take profit, and position management.
    
    Args:
        df: DataFrame with technical indicators
        params: Strategy parameters (including stop_loss_pct, take_profit_pct, max_position)
        
    Returns:
        DataFrame with signals and positions added
    """
    df = df.copy()
    df['signal'] = 0
    df['position'] = 0
    df['entry_price'] = np.nan
    
    stop_loss_pct = params.get('stop_loss_pct', 0.02)  # 2% default
    take_profit_pct = params.get('take_profit_pct', 0.01)  # 1% default
    max_position = params.get('max_position', 1)
    
    position = 0
    entry_price = np.nan
    for i in range(len(df)):
        # 進場條件
        if position == 0:
            buy_condition = (
                (df['close'].iloc[i] <= df['bb_lower'].iloc[i]) &
                (df['rsi'].iloc[i] < params['rsi_oversold']) &
                (df['obv_ratio'].iloc[i] > params['obv_threshold'])
            )
            sell_condition = (
                (df['close'].iloc[i] >= df['bb_upper'].iloc[i]) &
                (df['rsi'].iloc[i] > params['rsi_overbought']) &
                (df['obv_ratio'].iloc[i] < 1/params['obv_threshold'])
            )
            if buy_condition and position < max_position:
                df['signal'].iloc[i] = 1
                position = 1
                entry_price = df['close'].iloc[i]
            elif sell_condition and position > -max_position:
                df['signal'].iloc[i] = -1
                position = -1
                entry_price = df['close'].iloc[i]
        else:
            # 持倉中，檢查停損/停利
            if position == 1:
                # 多單停損/停利
                if (df['close'].iloc[i] <= entry_price * (1 - stop_loss_pct)) or (df['close'].iloc[i] >= entry_price * (1 + take_profit_pct)):
                    df['signal'].iloc[i] = -1  # 平多
                    position = 0
                    entry_price = np.nan
            elif position == -1:
                # 空單停損/停利
                if (df['close'].iloc[i] >= entry_price * (1 + stop_loss_pct)) or (df['close'].iloc[i] <= entry_price * (1 - take_profit_pct)):
                    df['signal'].iloc[i] = 1  # 平空
                    position = 0
                    entry_price = np.nan
        df['position'].iloc[i] = position
        df['entry_price'].iloc[i] = entry_price
    return df

def calculate_returns(df: pd.DataFrame, params: Dict) -> Dict:
    """
    Calculate strategy returns and performance metrics.
    
    Args:
        df: DataFrame with signals and positions
        params: Strategy parameters
        
    Returns:
        Dictionary with performance metrics
    """
    df = df.copy()
    
    # Calculate strategy returns
    df['price_change'] = df['close'].pct_change()
    df['strategy_return'] = df['position'].shift(1) * df['price_change']
    
    # Apply transaction costs
    transaction_cost = params.get('transaction_cost', 0.0001)  # 0.01% per trade
    position_changes = df['position'].diff().abs()
    df['transaction_costs'] = position_changes * transaction_cost
    df['net_return'] = df['strategy_return'] - df['transaction_costs']
    
    # Calculate cumulative returns
    df['cumulative_return'] = (1 + df['net_return']).cumprod()
    
    # Calculate performance metrics
    total_return = df['cumulative_return'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(df)) - 1
    
    # Calculate volatility
    daily_returns = df['net_return'].resample('D').sum()
    volatility = daily_returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio
    risk_free_rate = 0.02  # 2% annual risk-free rate
    sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Calculate maximum drawdown
    cumulative = df['cumulative_return']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    trades = df[df['signal'] != 0]
    if len(trades) > 0:
        winning_trades = trades[trades['strategy_return'] > 0]
        win_rate = len(winning_trades) / len(trades)
    else:
        win_rate = 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'final_cumulative_return': df['cumulative_return'].iloc[-1]
    }

def optimize_parameters(df: pd.DataFrame, param_ranges: Dict) -> Tuple[Dict, Dict]:
    """
    Optimize strategy parameters using grid search.
    
    Args:
        df: DataFrame with OHLCV data
        param_ranges: Dictionary with parameter ranges to test
        
    Returns:
        Tuple of (best_params, best_metrics)
    """
    print("Starting parameter optimization...")
    
    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    param_combinations = list(itertools.product(*param_values))
    
    best_sharpe = -np.inf
    best_params = None
    best_metrics = None
    results = []
    
    total_combinations = len(param_combinations)
    print(f"Testing {total_combinations} parameter combinations...")
    
    # 新增停損/停利參數範圍（如未指定則預設）
    if 'stop_loss_pct' not in param_ranges:
        param_ranges['stop_loss_pct'] = [0.02]
    if 'take_profit_pct' not in param_ranges:
        param_ranges['