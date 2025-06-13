import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from indicators import TechnicalIndicators
from openai import OpenAI
import os
from signal_recorder import SignalRecorder

@dataclass
class TradingSignal:
    timestamp: pd.Timestamp
    symbol: str
    signal_type: str  # 'LONG' or 'SHORT'
    confidence: float  # 0.0 to 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    indicators: Dict[str, float] = None
    analysis: str = None

class SignalGenerator:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.indicators = TechnicalIndicators(config_path)
        self.signal_recorder = SignalRecorder(config_path)
        self._setup_openai()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        import yaml
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _setup_openai(self):
        """Setup OpenAI client with API key."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
    
    def _get_ai_analysis(self, df: pd.DataFrame, indicators: Dict[str, pd.Series], signal_type: str) -> Tuple[float, float, float, str]:
        """Get AI analysis for signal confidence and SL/TP levels."""
        latest = df.iloc[-1]
        latest_indicators = {k: v.iloc[-1] for k, v in indicators.items()}
        
        prompt = f"""Analyze the following market data and provide trading signal analysis:

Symbol: {latest.name}
Signal Type: {signal_type}
Current Price: {latest['close']:.2f}
RSI: {latest_indicators['rsi']:.2f}
MACD: {latest_indicators['macd']:.2f}
EMA20: {latest_indicators['ema_20']:.2f}
EMA50: {latest_indicators['ema_50']:.2f}
BB Upper: {latest_indicators['bb_upper']:.2f}
BB Lower: {latest_indicators['bb_lower']:.2f}
Volume: {latest['volume']:.0f}

Please provide:
1. Signal confidence (0.0 to 1.0)
2. Stop Loss level
3. Take Profit level
4. Brief analysis of the signal

Format your response as:
CONFIDENCE: [confidence]
STOP_LOSS: [stop_loss]
TAKE_PROFIT: [take_profit]
ANALYSIS: [analysis]
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config['api']['openai']['model'],
                messages=[
                    {"role": "system", "content": "You are a professional trading analyst specializing in technical analysis and risk management."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['api']['openai']['temperature']
            )
            
            # Parse the response
            result = response.choices[0].message.content.strip()
            lines = result.split('\n')
            
            confidence = float(lines[0].split(': ')[1].strip())
            stop_loss = float(lines[1].split(': ')[1].strip())
            take_profit = float(lines[2].split(': ')[1].strip())
            analysis = lines[3].split(': ')[1].strip()
            
            return confidence, stop_loss, take_profit, analysis
            
        except Exception as e:
            print(f"Error in AI analysis: {str(e)}")
            return 0.5, latest['close'] * 0.95, latest['close'] * 1.05, "Error in AI analysis"
    
    def _check_long_conditions(self, df: pd.DataFrame, indicators: Dict[str, pd.Series]) -> bool:
        """Check conditions for LONG signal."""
        # Get latest values
        latest = df.iloc[-1]
        latest_indicators = {k: v.iloc[-1] for k, v in indicators.items()}
        
        # EMA20 > EMA50
        ema_condition = latest_indicators['ema_20'] > latest_indicators['ema_50']
        
        # RSI rising from oversold
        rsi_condition = (indicators['rsi'].iloc[-2] < 30 and 
                        indicators['rsi'].iloc[-1] > indicators['rsi'].iloc[-2])
        
        # BB breakout with volume
        bb_condition = (latest['close'] > latest_indicators['bb_upper'] and
                       latest['volume'] > df['volume'].rolling(20).mean().iloc[-1])
        
        return ema_condition and rsi_condition and bb_condition
    
    def _check_short_conditions(self, df: pd.DataFrame, indicators: Dict[str, pd.Series]) -> bool:
        """Check conditions for SHORT signal."""
        # Get latest values
        latest = df.iloc[-1]
        latest_indicators = {k: v.iloc[-1] for k, v in indicators.items()}
        
        # EMA20 < EMA50
        ema_condition = latest_indicators['ema_20'] < latest_indicators['ema_50']
        
        # RSI falling from overbought
        rsi_condition = (indicators['rsi'].iloc[-2] > 70 and 
                        indicators['rsi'].iloc[-1] < indicators['rsi'].iloc[-2])
        
        # Price rejection from upper BB
        bb_condition = (latest['high'] > latest_indicators['bb_upper'] and
                       latest['close'] < latest_indicators['bb_upper'])
        
        return ema_condition and rsi_condition and bb_condition
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """Generate trading signal based on technical indicators."""
        # Calculate all indicators
        indicators = self.indicators.calculate_all_indicators(df)
        
        # Get latest values
        latest = df.iloc[-1]
        latest_indicators = {k: v.iloc[-1] for k, v in indicators.items()}
        
        # Determine signal type
        signal_type = None
        
        # Check for LONG signal
        if self._check_long_conditions(df, indicators):
            signal_type = 'LONG'
        
        # Check for SHORT signal
        elif self._check_short_conditions(df, indicators):
            signal_type = 'SHORT'
        
        if signal_type:
            # Get AI analysis for confidence and SL/TP levels
            confidence, stop_loss, take_profit, analysis = self._get_ai_analysis(
                df, indicators, signal_type
            )
            
            # Create signal object
            signal = TradingSignal(
                timestamp=latest.name,
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=latest_indicators,
                analysis=analysis
            )
            
            # Record the signal
            signal_data = {
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': confidence,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_price': latest['close'],
                'analysis': analysis
            }
            self.signal_recorder.record_signal(signal_data)
            
            return signal
        
        return None
    
    def get_signal_history(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get signal history for a specific symbol or all symbols."""
        return self.signal_recorder.get_signal_history(symbol)
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for all signals."""
        return self.signal_recorder.get_performance_metrics()

if __name__ == "__main__":
    # Test the signal generator
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
    
    # Generate signal
    sg = SignalGenerator()
    signal = sg.generate_signal(df, 'BTCUSDT')
    
    if signal:
        print("\nGenerated Trading Signal:")
        print(f"Symbol: {signal.symbol}")
        print(f"Type: {signal.signal_type}")
        print(f"Confidence: {signal.confidence:.2%}")
        print(f"Stop Loss: {signal.stop_loss:.2f}")
        print(f"Take Profit: {signal.take_profit:.2f}")
        print(f"Analysis: {signal.analysis}")
        print("\nLatest Indicators:")
        for name, value in signal.indicators.items():
            print(f"{name}: {value:.2f}")
        
        # Print performance metrics
        print("\nPerformance Metrics:")
        metrics = sg.get_performance_metrics()
        for key, value in metrics.items():
            print(f"{key}: {value}")
    else:
        print("\nNo signal generated - conditions not met") 