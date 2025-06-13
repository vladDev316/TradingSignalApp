# NEXEDGE AI Trader

An AI-powered trading assistant that combines technical analysis with macro market sentiment to generate trading signals for crypto, stocks, and forex markets.

## Features

- Real-time market data from Binance (crypto) and Alpha Vantage (stocks/forex)
- Technical indicators: RSI, MACD, EMA(20/50/200), Bollinger Bands, ATR
- AI-powered macro market sentiment analysis using GPT-4
- Interactive Streamlit dashboard with real-time charts and signals
- Configurable trading parameters and asset selection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nexedge-ai-trader.git
cd nexedge-ai-trader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Start the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

2. Access the dashboard at `http://localhost:8501`

3. Select an asset type and specific asset from the sidebar

4. View technical analysis charts and trading signals

5. Enter market headlines for AI-powered sentiment analysis

## Project Structure

- `data_fetcher.py`: Fetches market data from various sources
- `indicators.py`: Implements technical indicators
- `signal_logic.py`: Generates trading signals based on technical analysis
- `ai_bias.py`: Analyzes market sentiment using GPT-4
- `dashboard.py`: Streamlit web interface
- `config.yaml`: Configuration file for assets and parameters

## Trading Logic

The system generates trading signals based on the following criteria:

### Long Signals
- EMA20 > EMA50
- RSI rising from oversold (< 30)
- Bollinger Band breakout with volume confirmation
- MACD bullish crossover

### Short Signals
- EMA20 < EMA50
- RSI falling from overbought (> 70)
- Price rejection from upper Bollinger Band
- MACD bearish crossover

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS. 