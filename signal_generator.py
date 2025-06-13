import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import ta
from data_fetcher import DataFetcher
from logging_utils import SignalEngineLogger

class SignalGenerator:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.logger = SignalEngineLogger()
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the given price data."""
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # EMAs
        df['ema20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
        df['ema50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close']
        ).average_true_range()
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Generate trading signal based on technical indicators."""
        # Get the latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Initialize signal components
        signal_components = []
        confidence_components = []
        
        # RSI signals
        if latest['rsi'] < 30:
            signal_components.append('buy')
            confidence_components.append(0.8)
        elif latest['rsi'] > 70:
            signal_components.append('sell')
            confidence_components.append(0.8)
        
        # MACD signals
        if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            signal_components.append('buy')
            confidence_components.append(0.7)
        elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            signal_components.append('sell')
            confidence_components.append(0.7)
        
        # EMA trend signals
        if latest['ema20'] > latest['ema50']:
            signal_components.append('buy')
            confidence_components.append(0.6)
        elif latest['ema20'] < latest['ema50']:
            signal_components.append('sell')
            confidence_components.append(0.6)
        
        # Bollinger Band signals
        if latest['close'] < latest['bb_lower']:
            signal_components.append('buy')
            confidence_components.append(0.7)
        elif latest['close'] > latest['bb_upper']:
            signal_components.append('sell')
            confidence_components.append(0.7)
        
        # ATR volatility filter
        atr_threshold = df['atr'].mean() * 0.5
        if latest['atr'] < atr_threshold:
            # Low volatility - reduce confidence
            confidence_components = [c * 0.5 for c in confidence_components]
        
        # Determine final signal
        if not signal_components:
            return 'neutral', 0.0
        
        # Count buy and sell signals
        buy_count = signal_components.count('buy')
        sell_count = signal_components.count('sell')
        
        # Calculate average confidence
        avg_confidence = sum(confidence_components) / len(confidence_components)
        
        # Determine final signal
        if buy_count > sell_count:
            return 'buy', avg_confidence
        elif sell_count > buy_count:
            return 'sell', avg_confidence
        else:
            return 'neutral', avg_confidence
    
    def generate_all_signals(self) -> pd.DataFrame:
        """Generate signals for all assets and save to CSV."""
        signals = []
        errors = []
        processed_symbols = []
        
        # Get available symbols
        available_symbols = self.data_fetcher.get_available_symbols()
        
        # Process each asset type
        for asset_type, symbols in available_symbols.items():
            for symbol in symbols:
                try:
                    processed_symbols.append(symbol)
                    # Fetch data
                    if asset_type == 'crypto':
                        df = self.data_fetcher.fetch_crypto_data(symbol)
                    elif asset_type == 'stocks':
                        df = self.data_fetcher.fetch_stock_data(symbol)
                    else:
                        continue
                    
                    # Calculate indicators
                    df = self.calculate_indicators(df)
                    
                    # Generate signal
                    signal, confidence = self.generate_signal(df)
                    
                    # Add to signals list
                    signals.append({
                        'timestamp': df.index[-1],
                        'symbol': symbol,
                        'asset_type': asset_type,
                        'signal': signal,
                        'confidence_score': confidence
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing {symbol}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.log_error(error_msg)
        
        # Convert to DataFrame
        signals_df = pd.DataFrame(signals)
        
        # Save to CSV
        signals_df.to_csv('signals.csv', index=False)
        
        # Log the run summary
        self.logger.log_signal_engine_run(
            assets_processed=processed_symbols,
            signal_count=len(signals),
            macro_bias_score=self.data_fetcher.get_macro_bias_score(),
            errors=errors
        )
        
        return signals_df

if __name__ == "__main__":
    # Test the signal generator
    generator = SignalGenerator()
    signals = generator.generate_all_signals()
    print("\nGenerated Signals:")
    print(signals) 