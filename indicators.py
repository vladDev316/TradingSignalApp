import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class TechnicalIndicators:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        import yaml
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI indicator."""
        period = self.config['indicators']['rsi']['period']
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        fast_period = self.config['indicators']['macd']['fast_period']
        slow_period = self.config['indicators']['macd']['slow_period']
        signal_period = self.config['indicators']['macd']['signal_period']
        
        # Calculate EMAs
        exp1 = df['close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd = exp1 - exp2
        
        # Calculate Signal line
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate Histogram
        hist = macd - signal
        
        return macd, signal, hist
    
    def calculate_ema(self, df: pd.DataFrame) -> Dict[int, pd.Series]:
        """Calculate EMA for multiple periods."""
        ema_dict = {}
        for period in self.config['indicators']['ema']['periods']:
            ema_dict[period] = df['close'].ewm(span=period, adjust=False).mean()
        return ema_dict
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        period = self.config['indicators']['bollinger_bands']['period']
        std_dev = self.config['indicators']['bollinger_bands']['std_dev']
        
        # Calculate middle band (SMA)
        middle = df['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = df['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        period = self.config['indicators']['atr']['period']
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all technical indicators for a given dataframe."""
        indicators = {}
        
        # Calculate RSI
        indicators['rsi'] = self.calculate_rsi(df)
        
        # Calculate MACD
        macd, signal, hist = self.calculate_macd(df)
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_hist'] = hist
        
        # Calculate EMAs
        ema_dict = self.calculate_ema(df)
        for period, ema in ema_dict.items():
            indicators[f'ema_{period}'] = ema
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        # Calculate ATR
        indicators['atr'] = self.calculate_atr(df)
        
        return indicators

if __name__ == "__main__":
    # Test the indicators
    import yaml
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    df = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Calculate indicators
    ti = TechnicalIndicators()
    indicators = ti.calculate_all_indicators(df)
    
    # Print sample results
    print("\nSample Technical Indicators:")
    for name, indicator in indicators.items():
        print(f"\n{name} - Last 5 values:")
        print(indicator.tail()) 