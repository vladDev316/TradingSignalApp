import os
from dotenv import load_dotenv
import yaml
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from alpha_vantage.timeseries import TimeSeries
import requests
from typing import Dict, List, Optional, Any
import yfinance as yf
import streamlit as st

# Load environment variables
load_dotenv()

class DataFetcher:
    def __init__(self):
        self.config = self._load_config()
        self.binance_client = None
        
        # Initialize Binance client (optional)
        binance_api_key = st.secrets.get("BINANCE_API_KEY", os.getenv('BINANCE_API_KEY'))
        binance_api_secret = st.secrets.get("BINANCE_API_SECRET", os.getenv('BINANCE_API_SECRET'))
        
        if binance_api_key and binance_api_secret:
            try:
                self.binance_client = Client(binance_api_key, binance_api_secret)
                # Test the connection
                self.binance_client.ping()
            except BinanceAPIException as e:
                st.warning(f"""
                ⚠️ Failed to connect to Binance API: {str(e)}
                
                This could be due to:
                1. Invalid API keys
                2. API keys not having the correct permissions
                3. IP restrictions on your API keys
                
                Crypto data fetching will be disabled. The app will continue to work with stocks and forex data.
                """)
                self.binance_client = None
            except Exception as e:
                st.warning(f"""
                ⚠️ Error initializing Binance client: {str(e)}
                
                Crypto data fetching will be disabled. The app will continue to work with stocks and forex data.
                """)
                self.binance_client = None
        else:
            st.info("""
            ℹ️ Binance API credentials not found. Crypto data fetching will be disabled.
            The app will continue to work with stocks and forex data.
            """)
        
        # Initialize Alpha Vantage client
        alpha_vantage_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", os.getenv('ALPHA_VANTAGE_API_KEY'))
        if not alpha_vantage_key:
            error_msg = """
            ⚠️ Alpha Vantage API Key Required
            
            Since you're running on Streamlit Cloud, please add your API key in the Streamlit Cloud dashboard:
            
            1. Go to https://share.streamlit.io/
            2. Click on your app
            3. Click "Manage app" in the lower right
            4. Go to the "Secrets" section
            5. Add your Alpha Vantage API key:
            
            ```toml
            ALPHA_VANTAGE_API_KEY = "your_key_here"
            ```
            
            You can get a free Alpha Vantage API key from: https://www.alphavantage.co/support/#api-key
            """
            st.error(error_msg)
            st.stop()
            
        self.alpha_vantage_client = TimeSeries(key=alpha_vantage_key, output_format='pandas')
        
        # Initialize Polygon.io client (optional)
        self.polygon_api_key = st.secrets.get("POLYGON_API_KEY", os.getenv('POLYGON_API_KEY'))
        if not self.polygon_api_key:
            st.info("""
            ℹ️ Polygon.io API key not found. Index data fetching will be disabled.
            
            If you want to enable index data fetching, add your Polygon.io API key in the Streamlit Cloud dashboard:
            
            1. Go to https://share.streamlit.io/
            2. Click on your app
            3. Click "Manage app" in the lower right
            4. Go to the "Secrets" section
            5. Add your Polygon.io API key:
            
            ```toml
            POLYGON_API_KEY = "your_key_here"
            ```
            
            You can get a Polygon.io API key from: https://polygon.io/
            """)
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open("config.yaml", 'r') as file:
            return yaml.safe_load(file)
    
    def fetch_crypto_data(self, symbol: str) -> pd.DataFrame:
        """Fetch cryptocurrency data from Binance."""
        if not self.binance_client:
            raise ValueError("""
            Crypto data fetching is disabled because Binance API is not available.
            
            This could be due to:
            1. Missing API keys
            2. Invalid API keys
            3. API keys not having the correct permissions
            4. IP restrictions on your API keys
            
            The app will continue to work with stocks and forex data.
            """)
        
        try:
            # Get historical klines/candlestick data
            klines = self.binance_client.get_historical_klines(
                symbol,
                Client.KLINE_INTERVAL_1DAY,
                str(datetime.now() - timedelta(days=365))
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df
            
        except BinanceAPIException as e:
            raise ValueError(f"Binance API error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error fetching crypto data: {str(e)}")
    
    def fetch_forex_data(self, symbol: str) -> pd.DataFrame:
        """Fetch forex data from Alpha Vantage."""
        try:
            # Get daily forex data
            data, _ = self.alpha_vantage_client.get_daily(
                symbol=symbol,
                outputsize='full'
            )
            
            if data is None or data.empty:
                raise ValueError(f"No data returned for forex pair {symbol}. Please check if the symbol is correct.")
            
            # Rename columns to match our standard format
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Sort index to ensure chronological order
            data.sort_index(inplace=True)
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove any rows with NaN values
            data.dropna(inplace=True)
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error fetching forex data for {symbol}: {str(e)}")
    
    def fetch_index_data(self, symbol: str) -> pd.DataFrame:
        """Fetch index data using yfinance."""
        try:
            # For SPX, use ^GSPC ticker in yfinance
            if symbol == "I:SPX":
                ticker = "^GSPC"
            else:
                ticker = symbol

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Fetch data using yfinance
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d"
            )
            
            if data.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Get the actual column names
            columns = data.columns.tolist()
            
            # Drop the adjusted close column if it exists
            # Convert column names to strings before checking
            adj_close_cols = [col for col in columns if isinstance(col, str) and ('adj' in col.lower() or 'adjusted' in col.lower())]
            if adj_close_cols:
                data = data.drop(adj_close_cols, axis=1)
            
            # Rename columns to match our standard format
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Sort index to ensure chronological order
            data.sort_index(inplace=True)
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error fetching index data for {symbol}: {str(e)}")
    
    def fetch_stock_data(self, symbol: str) -> pd.DataFrame:
        """Fetch stock data from Alpha Vantage."""
        try:
            # Special handling for SPX - use yfinance
            if symbol == "SPX":
                return self.fetch_index_data("I:SPX")
            
            # Get daily stock data from Alpha Vantage for other symbols
            data, _ = self.alpha_vantage_client.get_daily(
                symbol=symbol,
                outputsize='full'
            )
            
            if data is None or data.empty:
                raise ValueError(f"No data returned for symbol {symbol}. Please check if the symbol is correct.")
            
            # Rename columns to match our standard format
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Sort index to ensure chronological order
            data.sort_index(inplace=True)
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove any rows with NaN values
            data.dropna(inplace=True)
            
            return data
            
        except ValueError as ve:
            # Re-raise ValueError with more context
            raise ValueError(f"Error fetching stock data for {symbol}: {str(ve)}")
        except Exception as e:
            # Handle other exceptions
            raise ValueError(f"Unexpected error fetching stock data for {symbol}: {str(e)}")
    
    def get_available_symbols(self) -> Dict[str, list]:
        """Get list of available symbols for each asset type."""
        available_symbols = {
            'stocks': [asset['symbol'] for asset in self.config['assets']['stocks']],
            'forex': [asset['symbol'] for asset in self.config['assets']['forex']]
        }
        
        # Only include crypto if Binance client is available
        if self.binance_client:
            available_symbols['crypto'] = [asset['symbol'] for asset in self.config['assets']['crypto']]
        else:
            st.info("""
            ℹ️ Crypto assets are not available because Binance API credentials are not configured.
            The app will continue to work with stocks and forex data.
            """)
        
        return available_symbols

    def fetch_all_assets(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch data for all configured assets."""
        data = {
            'stocks': {},
            'forex': {}
        }
        
        # Only include crypto if Binance client is available
        if self.binance_client:
            data['crypto'] = {}
            # Fetch crypto data
            for asset in self.config['assets']['crypto']:
                try:
                    data['crypto'][asset['symbol']] = self.fetch_crypto_data(asset['symbol'])
                except Exception as e:
                    st.warning(f"Error fetching {asset['symbol']}: {str(e)}")
        
        # Fetch forex data
        for asset in self.config['assets']['forex']:
            try:
                data['forex'][asset['symbol']] = self.fetch_forex_data(asset['symbol'])
            except Exception as e:
                st.warning(f"Error fetching {asset['symbol']}: {str(e)}")
        
        # Fetch stock data
        for asset in self.config['assets']['stocks']:
            try:
                data['stocks'][asset['symbol']] = self.fetch_stock_data(asset['symbol'])
            except Exception as e:
                st.warning(f"Error fetching {asset['symbol']}: {str(e)}")
        
        return data

if __name__ == "__main__":
    # Test the data fetcher
    fetcher = DataFetcher()
    data = fetcher.fetch_all_assets()
    
    # Print sample data for each asset type
    for asset_type, assets in data.items():
        print(f"\n{asset_type.upper()} Data:")
        for symbol, df in assets.items():
            print(f"\n{symbol} - Last 5 days:")
            print(df.tail()) 