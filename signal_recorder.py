import pandas as pd
from datetime import datetime
from typing import Optional, Dict
import os
import yaml

class SignalRecorder:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.csv_path = self.config['signal_recording']['csv_path']
        self._initialize_csv()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _initialize_csv(self):
        """Initialize CSV file if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            columns = self.config['signal_recording']['columns']
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.csv_path, index=False)
    
    def record_signal(self, signal_data: Dict):
        """Record a new trading signal."""
        # Create a new row with default values
        new_row = {
            'timestamp': datetime.now(),
            'symbol': signal_data['symbol'],
            'signal_type': signal_data['signal_type'],
            'confidence': signal_data['confidence'],
            'stop_loss': signal_data['stop_loss'],
            'take_profit': signal_data['take_profit'],
            'entry_price': signal_data['entry_price'],
            'analysis': signal_data['analysis'],
            'status': 'OPEN',
            'exit_price': None,
            'exit_timestamp': None,
            'pnl': None,
            'pnl_percentage': None
        }
        
        # Append to CSV
        df = pd.read_csv(self.csv_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.csv_path, index=False)
    
    def update_signal(self, symbol: str, timestamp: datetime, update_data: Dict):
        """Update an existing signal with exit information."""
        df = pd.read_csv(self.csv_path)
        
        # Convert timestamp to datetime for comparison
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Find the matching signal
        mask = (df['symbol'] == symbol) & (df['timestamp'] == timestamp)
        if not any(mask):
            raise ValueError(f"No open signal found for {symbol} at {timestamp}")
        
        # Update the signal
        for key, value in update_data.items():
            df.loc[mask, key] = value
        
        # Calculate PnL if exit price is provided
        if 'exit_price' in update_data:
            entry_price = df.loc[mask, 'entry_price'].iloc[0]
            exit_price = update_data['exit_price']
            signal_type = df.loc[mask, 'signal_type'].iloc[0]
            
            if signal_type == 'LONG':
                pnl = exit_price - entry_price
            else:  # SHORT
                pnl = entry_price - exit_price
            
            pnl_percentage = (pnl / entry_price) * 100
            
            df.loc[mask, 'pnl'] = pnl
            df.loc[mask, 'pnl_percentage'] = pnl_percentage
        
        df.to_csv(self.csv_path, index=False)
    
    def get_open_signals(self) -> pd.DataFrame:
        """Get all open signals."""
        df = pd.read_csv(self.csv_path)
        return df[df['status'] == 'OPEN']
    
    def get_closed_signals(self) -> pd.DataFrame:
        """Get all closed signals."""
        df = pd.read_csv(self.csv_path)
        return df[df['status'] == 'CLOSED']
    
    def get_signal_history(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get signal history for a specific symbol or all symbols."""
        df = pd.read_csv(self.csv_path)
        if symbol:
            return df[df['symbol'] == symbol]
        return df
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics from signal history."""
        df = pd.read_csv(self.csv_path)
        closed_signals = df[df['status'] == 'CLOSED']
        
        if len(closed_signals) == 0:
            return {
                'total_signals': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0
            }
        
        winning_signals = closed_signals[closed_signals['pnl'] > 0]
        
        return {
            'total_signals': len(closed_signals),
            'win_rate': len(winning_signals) / len(closed_signals),
            'avg_pnl': closed_signals['pnl'].mean(),
            'total_pnl': closed_signals['pnl'].sum()
        }

if __name__ == "__main__":
    # Test the signal recorder
    recorder = SignalRecorder()
    
    # Record a test signal
    test_signal = {
        'symbol': 'BTCUSDT',
        'signal_type': 'LONG',
        'confidence': 0.85,
        'stop_loss': 40000,
        'take_profit': 45000,
        'entry_price': 42000,
        'analysis': 'Strong bullish momentum with volume confirmation'
    }
    
    recorder.record_signal(test_signal)
    
    # Update the signal
    update_data = {
        'status': 'CLOSED',
        'exit_price': 44500,
        'exit_timestamp': datetime.now()
    }
    
    recorder.update_signal('BTCUSDT', test_signal['timestamp'], update_data)
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    metrics = recorder.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}") 