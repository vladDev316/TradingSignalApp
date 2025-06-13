import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yaml
from data_fetcher import DataFetcher
from indicators import TechnicalIndicators
from signal_generator import SignalGenerator
from ai_bias import MacroBiasAnalyzer, MacroBias
import json
import openai
from openai import OpenAI
import ta

# Load environment variables
load_dotenv()

class TradingDashboard:
    def __init__(self):
        """Initialize the dashboard."""
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components with error handling
        try:
            self.data_fetcher = DataFetcher()
        except Exception as e:
            st.error(str(e))
            st.stop()
        
        self.signal_generator = SignalGenerator()
        
        # Initialize OpenAI client
        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv('OPENAI_API_KEY'))
        if not api_key:
            st.error("""
            ⚠️ OpenAI API Key Required
            
            Since you're running on Streamlit Cloud, please add your API key in the Streamlit Cloud dashboard:
            
            1. Go to https://share.streamlit.io/
            2. Click on your app
            3. Click "Manage app" in the lower right
            4. Go to the "Secrets" section
            5. Add your OpenAI API key:
            
            ```toml
            OPENAI_API_KEY = "your_key_here"
            ```
            
            You can get an OpenAI API key from: https://platform.openai.com/api-keys
            """)
            st.stop()
        self.openai_client = OpenAI(api_key=api_key)
        
        # Initialize macro analyzer with API key
        self.macro_analyzer = MacroBiasAnalyzer(api_key=api_key)
        
        # Initialize session state
        if 'macro_bias' not in st.session_state:
            st.session_state.macro_bias = None
        if 'generated_signals' not in st.session_state:
            st.session_state.generated_signals = []
        if 'final_signals' not in st.session_state:
            st.session_state.final_signals = []
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open("config.yaml", 'r') as file:
            return yaml.safe_load(file)
    
    def _create_export_data(self, df: pd.DataFrame, indicators: dict) -> pd.DataFrame:
        """Create a DataFrame for export with all chart values."""
        # Create a copy of the original DataFrame
        export_df = df.copy()
        
        # Reset index to make timestamp a column
        export_df = export_df.reset_index()
        
        # Add technical indicators
        for name, values in indicators.items():
            export_df[name] = values.values
        
        # Ensure all columns are properly named
        export_df.columns = [col.lower() for col in export_df.columns]
        
        return export_df
    
    def _create_candlestick_chart(self, df: pd.DataFrame, indicators: dict, symbol: str, start_date: datetime, end_date: datetime) -> go.Figure:
        """Create an interactive candlestick chart with indicators."""
        # Filter data for the selected time period
        mask = (df.index >= start_date) & (df.index <= end_date)
        filtered_df = df[mask]
        filtered_indicators = {k: v[mask] for k, v in indicators.items()}
        
        fig = go.Figure()
        
        # Add Bollinger Bands first (so they appear behind the candlesticks)
        fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_indicators['bb_upper'],
            name='BB Upper',
            line=dict(width=1, color='rgba(255, 0, 0, 0.7)'),
            fill=None
        ))
        fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_indicators['bb_middle'],
            name='BB Middle',
            line=dict(width=1, color='rgba(255, 255, 0, 0.7)'),
            fill=None
        ))
        fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_indicators['bb_lower'],
            name='BB Lower',
            line=dict(width=1, color='rgba(255, 0, 0, 0.7)'),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ))
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=filtered_df.index,
            open=filtered_df['open'],
            high=filtered_df['high'],
            low=filtered_df['low'],
            close=filtered_df['close'],
            name='Price'
        ))
        
        # Add EMAs (20, 50, 200)
        ema_colors = {20: 'yellow', 50: 'blue', 200: 'red'}
        for period in [20, 50, 200]:
            if f'ema_{period}' in filtered_indicators:
                fig.add_trace(go.Scatter(
                    x=filtered_df.index,
                    y=filtered_indicators[f'ema_{period}'],
                    name=f'EMA {period}',
                    line=dict(width=1, color=ema_colors[period])
                ))
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Price Chart',
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_dark',
            height=800,
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date",
                range=[start_date, end_date]
            ),
            yaxis=dict(
                autorange=True,
                fixedrange=False
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def _create_indicator_charts(self, df: pd.DataFrame, indicators: dict, start_date: datetime, end_date: datetime) -> tuple:
        """Create charts for RSI, MACD, and ATR."""
        # Filter data for the selected time period
        mask = (df.index >= start_date) & (df.index <= end_date)
        filtered_df = df[mask]
        filtered_indicators = {k: v[mask] for k, v in indicators.items()}
        
        # RSI Chart
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_indicators['rsi'],
            name='RSI',
            line=dict(color='purple')
        ))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
        rsi_fig.update_layout(
            title='RSI',
            yaxis_title='RSI',
            template='plotly_dark',
            height=300,
            xaxis=dict(
                range=[start_date, end_date]
            ),
            yaxis=dict(
                range=[0, 100]  # Fixed range for RSI
            )
        )
        
        # MACD Chart
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_indicators['macd'],
            name='MACD',
            line=dict(color='blue')
        ))
        macd_fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_indicators['macd_signal'],
            name='Signal',
            line=dict(color='orange')
        ))
        macd_fig.add_trace(go.Bar(
            x=filtered_df.index,
            y=filtered_indicators['macd_hist'],
            name='Histogram'
        ))
        macd_fig.update_layout(
            title='MACD',
            yaxis_title='MACD',
            template='plotly_dark',
            height=300,
            xaxis=dict(
                range=[start_date, end_date]
            )
        )
        
        # ATR Chart
        atr_fig = go.Figure()
        atr_fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_indicators['atr'],
            name='ATR',
            line=dict(color='green')
        ))
        atr_fig.update_layout(
            title='Average True Range (ATR)',
            yaxis_title='ATR',
            template='plotly_dark',
            height=300,
            xaxis=dict(
                range=[start_date, end_date]
            )
        )
        
        return rsi_fig, macd_fig, atr_fig
    
    def _get_asset_type(self, symbol: str) -> str:
        """Determine asset type from symbol."""
        if symbol.startswith('C:'):
            return 'Crypto'
        elif symbol.startswith('I:'):
            return 'Index'
        else:
            return 'Stocks'
    
    def _adjust_signal_confidence(self, signal_dict: dict, asset_type: str) -> dict:
        """Adjust signal confidence based on macro bias."""
        try:
            if not signal_dict:  # Check if signal_dict is None
                return None
                
            if not st.session_state.macro_bias or asset_type not in st.session_state.macro_bias:
                return signal_dict
                
            bias = st.session_state.macro_bias[asset_type]
            
            # Get confidence adjustment from macro bias
            confidence_adjustment = bias.confidence_adjustment if hasattr(bias, 'confidence_adjustment') else 0.0
            
            # Apply confidence adjustment
            adjusted_confidence = signal_dict['confidence'] + confidence_adjustment
            
            # Ensure confidence stays within bounds
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            return {
                'symbol': signal_dict['symbol'],
                'signal': signal_dict['signal'],
                'confidence': adjusted_confidence,
                'technical_confidence': signal_dict['confidence'],
                'timestamp': signal_dict['timestamp'],
                'indicators': signal_dict['indicators'],
                'macro_bias': {
                    'sentiment': bias.sentiment,
                    'impact_score': bias.impact_score,
                    'confidence': bias.confidence,
                    'confidence_adjustment': confidence_adjustment
                }
            }
        except Exception as e:
            st.error(f"Error adjusting signal confidence: {str(e)}")
            return signal_dict
    
    def _adjust_final_signal(self, signal_dict: dict, macro_bias: dict) -> dict:
        """Adjust final signal confidence by combining technical and macro bias."""
        if not macro_bias:
            return signal_dict
        
        # Get the impact score from macro bias
        impact_score = macro_bias.get('impact_score', 0)
        
        # Adjust confidence based on macro bias impact
        # If macro bias is negative, reduce confidence; if positive, increase confidence
        adjusted_confidence = signal_dict['confidence'] * (1 + impact_score)
        
        # Ensure confidence stays within 0-1 range
        adjusted_confidence = max(0, min(1, adjusted_confidence))
        
        return {
            'signal': signal_dict['signal'],
            'confidence': adjusted_confidence,
            'technical_confidence': signal_dict['confidence'],
            'macro_bias_impact': impact_score
        }
    
    def _generate_final_signal_with_ai(self, signal_dict: dict, macro_bias: MacroBias, asset_type: str) -> dict:
        """Generate final signal using OpenAI for sophisticated analysis."""
        try:
            # Get confidence adjustment from macro bias
            confidence_adjustment = macro_bias.confidence_adjustment if macro_bias else 0.0
            
            # Prepare the prompt for OpenAI
            prompt = f"""As a professional trading analyst, analyze the following trading signal and provide a final recommendation:

Asset Type: {asset_type}
Technical Signal: {signal_dict['signal']}
Technical Confidence: {signal_dict['confidence']:.2f}

Macro Market Context:
- Sentiment: {macro_bias.sentiment if macro_bias else 'N/A'}
- Impact Score: {macro_bias.impact_score if macro_bias else 0:.2f}
- Confidence: {macro_bias.confidence if macro_bias else 0:.2f}
- Confidence Adjustment: {confidence_adjustment:.2f}

Please provide:
1. Final signal (buy/sell/neutral)
2. Final confidence score (0-1)
3. Brief explanation of your reasoning
4. Risk assessment (low/medium/high)

Format your response as a JSON object with these fields:
{{
    "signal": "buy/sell/neutral",
    "confidence": 0.XX,
    "explanation": "your explanation",
    "risk_level": "low/medium/high"
}}"""

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional trading analyst with expertise in technical analysis and macro market analysis. Provide clear, concise, and accurate trading signals."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent outputs
                max_tokens=500
            )

            # Parse the response
            ai_analysis = json.loads(response.choices[0].message.content)

            return {
                'signal': ai_analysis['signal'],
                'confidence': float(ai_analysis['confidence']),
                'technical_confidence': signal_dict['confidence'],
                'macro_bias_impact': macro_bias.impact_score if macro_bias else 0,
                'confidence_adjustment': confidence_adjustment,
                'explanation': ai_analysis['explanation'],
                'risk_level': ai_analysis['risk_level']
            }

        except Exception as e:
            st.error(f"Error in AI analysis: {str(e)}")
            # Fallback to basic adjustment if AI analysis fails
            return {
                'signal': signal_dict['signal'],
                'confidence': signal_dict['confidence'],
                'technical_confidence': signal_dict['confidence'],
                'macro_bias_impact': macro_bias.impact_score if macro_bias else 0,
                'confidence_adjustment': confidence_adjustment,
                'explanation': "Error in AI analysis, using technical signal only",
                'risk_level': "medium"
            }
    
    def _calculate_confidence_score(self, indicators: dict) -> float:
        """Calculate confidence score based on technical indicators."""
        confidence = 0.0
        total_factors = 0

        # RSI Analysis (0-1 score)
        if 'rsi' in indicators:
            rsi = indicators['rsi'].iloc[-1]  # Get the latest value
            if rsi < 30:  # Oversold
                confidence += 1.0
            elif rsi > 70:  # Overbought
                confidence += 1.0
            elif 40 <= rsi <= 60:  # Neutral
                confidence += 0.5
            total_factors += 1

        # MACD Analysis (0-1 score)
        if all(k in indicators for k in ['macd', 'macd_signal']):
            macd = indicators['macd'].iloc[-1]  # Get the latest value
            macd_signal = indicators['macd_signal'].iloc[-1]  # Get the latest value
            if (macd > macd_signal and macd > 0) or (macd < macd_signal and macd < 0):
                confidence += 1.0  # Strong trend
            elif macd > macd_signal or macd < macd_signal:
                confidence += 0.7  # Weak trend
            else:
                confidence += 0.3  # No clear trend
            total_factors += 1

        # EMA Trend Analysis (0-1 score)
        if all(k in indicators for k in ['ema20', 'ema50']):
            ema20 = indicators['ema20'].iloc[-1]  # Get the latest value
            ema50 = indicators['ema50'].iloc[-1]  # Get the latest value
            ema20_prev = indicators['ema20'].iloc[-2]  # Get the previous value
            if (ema20 > ema50 and ema20 > ema20_prev) or (ema20 < ema50 and ema20 < ema20_prev):
                confidence += 1.0  # Strong trend
            elif ema20 > ema50 or ema20 < ema50:
                confidence += 0.7  # Weak trend
            else:
                confidence += 0.3  # No clear trend
            total_factors += 1

        # Bollinger Bands Analysis (0-1 score)
        if all(k in indicators for k in ['bb_upper', 'bb_lower', 'bb_middle']):
            price = indicators['close'].iloc[-1]  # Get the latest value
            bb_upper = indicators['bb_upper'].iloc[-1]  # Get the latest value
            bb_lower = indicators['bb_lower'].iloc[-1]  # Get the latest value
            bb_middle = indicators['bb_middle'].iloc[-1]  # Get the latest value
            
            if price > bb_upper or price < bb_lower:
                confidence += 1.0  # Strong breakout
            elif price > bb_middle or price < bb_middle:
                confidence += 0.7  # Moderate breakout
            else:
                confidence += 0.3  # No breakout
            total_factors += 1

        # ATR Volatility Filter (0-1 score)
        if 'atr' in indicators:
            atr = indicators['atr'].iloc[-1]  # Get the latest value
            atr_ma = indicators['atr'].rolling(window=20).mean().iloc[-1]  # Get the latest MA value
            if atr > atr_ma * 1.5:
                confidence += 1.0  # High volatility
            elif atr > atr_ma:
                confidence += 0.7  # Moderate volatility
            else:
                confidence += 0.3  # Low volatility
            total_factors += 1

        # Calculate final confidence score
        if total_factors > 0:
            final_confidence = confidence / total_factors
            return min(1.0, max(0.0, final_confidence))  # Ensure between 0 and 1
        return 0.0

    def _generate_signal(self, data: pd.DataFrame, symbol: str) -> dict:
        """Generate trading signal with confidence score."""
        try:
            # Calculate indicators
            indicators = self.signal_generator.calculate_indicators(data)
            
            # Calculate Bollinger Bands
            bollinger = ta.volatility.BollingerBands(close=data['close'])
            indicators['bb_upper'] = bollinger.bollinger_hband()
            indicators['bb_lower'] = bollinger.bollinger_lband()
            indicators['bb_middle'] = bollinger.bollinger_mavg()
            
            # Calculate ATR and its 20-period moving average
            atr = ta.volatility.AverageTrueRange(high=data['high'], low=data['low'], close=data['close'])
            indicators['atr'] = atr.average_true_range()
            indicators['atr_20ma'] = indicators['atr'].rolling(window=20).mean()
            
            # Get the latest row of data
            latest = data.iloc[-1]
            latest_indicators = {k: v.iloc[-1] if isinstance(v, pd.Series) else v for k, v in indicators.items()}
            
            # Calculate confidence using the provided logic
            weights = {'rsi': 0.3, 'macd': 0.25, 'ema': 0.25, 'bollinger': 0.2}
            confidence = 0.5  # neutral baseline
            
            # RSI logic
            rsi = latest_indicators['rsi']
            if rsi < 30:
                confidence += ((30 - rsi)/30) * weights['rsi']
            elif rsi > 70:
                confidence -= ((rsi - 70)/30) * weights['rsi']
                
            # MACD logic
            macd = latest_indicators['macd']
            macd_signal = latest_indicators['macd_signal']
            if macd > macd_signal:  # bullish crossover
                confidence += (macd - macd_signal)/0.5 * weights['macd']
            elif macd < macd_signal:  # bearish crossover
                confidence -= (macd_signal - macd)/0.5 * weights['macd']
                
            # EMA logic
            ema20 = latest_indicators['ema20']
            ema50 = latest_indicators['ema50']
            if ema20 > ema50:
                confidence += ((ema20 - ema50)/ema50 * 100) * weights['ema']
            else:
                confidence -= ((ema50 - ema20)/ema50 * 100) * weights['ema']
                
            # Bollinger Bands logic
            close = latest['close']
            bb_upper = latest_indicators['bb_upper']
            bb_lower = latest_indicators['bb_lower']
            if close < bb_lower:
                confidence += ((bb_lower - close)/(bb_upper - bb_lower)) * weights['bollinger']
            elif close > bb_upper:
                confidence -= ((close - bb_upper)/(bb_upper - bb_lower)) * weights['bollinger']
                
            # ATR filter
            atr = latest_indicators['atr']
            atr_20ma = latest_indicators['atr_20ma']
            if pd.notna(atr) and pd.notna(atr_20ma):  # Check for valid values
                atr_multiplier = 1.0 if atr > (atr_20ma * 0.7) else 0.5
                confidence = min(max((confidence * atr_multiplier), 0), 1)
            
            # Determine final signal and confidence score
            if confidence > 0.55:
                signal = 'buy'
                confidence_score = round(confidence, 2)
            elif confidence < 0.45:
                signal = 'sell'
                confidence_score = round(1 - confidence, 2)
            else:
                signal = 'neutral'
                confidence_score = 0.5
            
            # Get current timestamp
            timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence_score,
                'timestamp': timestamp,
                'indicators': {
                    'rsi': latest_indicators['rsi'],
                    'macd': latest_indicators['macd'],
                    'ema20': latest_indicators['ema20'],
                    'ema50': latest_indicators['ema50'],
                    'bb_upper': latest_indicators['bb_upper'],
                    'bb_middle': latest_indicators['bb_middle'],
                    'bb_lower': latest_indicators['bb_lower'],
                    'atr': latest_indicators['atr'],
                    'atr_20ma': latest_indicators['atr_20ma']
                }
            }
        except Exception as e:
            st.error(f"Error generating signal for {symbol}: {str(e)}")
            return None
    
    def _get_signal_color(self, signal_type: str) -> str:
        """Get color for signal type."""
        colors = {
            'buy': 'green',
            'sell': 'red',
            'neutral': 'gray'
        }
        return colors.get(signal_type.lower(), 'gray')
    
    def run(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="NEXEDGE AI Trader",
            page_icon="📈",
            layout="wide"
        )
        
        st.title("NEXEDGE AI Trading Dashboard")
        
        # Sidebar
        st.sidebar.header("Asset Selection")
        asset_type = st.sidebar.selectbox(
            "Select Asset Type",
            ["Crypto", "Forex", "Stocks"]
        )
        
        # Get assets for selected type
        assets = self.config['assets'][asset_type.lower()]
        selected_asset = st.sidebar.selectbox(
            "Select Asset",
            [asset['symbol'] for asset in assets]
        )
        
        # Time period selection
        st.sidebar.header("Time Period")
        time_period = st.sidebar.selectbox(
            "Select Time Period",
            ["1D", "1W", "1M", "3M", "6M", "1Y", "All"]
        )
        
        # Add refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        try:
            # Fetch data
            if asset_type.lower() == 'crypto':
                df = self.data_fetcher.fetch_crypto_data(selected_asset)
            elif asset_type.lower() == 'forex':
                df = self.data_fetcher.fetch_forex_data(selected_asset)
            else:
                df = self.data_fetcher.fetch_stock_data(selected_asset)
            
            if df is None or df.empty:
                st.error(f"No data available for {selected_asset}. Please try a different asset or check your API keys.")
                return
            
            # Calculate time period
            end_date = df.index[-1]
            if time_period == "1D":
                start_date = end_date - timedelta(days=1)
            elif time_period == "1W":
                start_date = end_date - timedelta(weeks=1)
            elif time_period == "1M":
                start_date = end_date - timedelta(days=30)
            elif time_period == "3M":
                start_date = end_date - timedelta(days=90)
            elif time_period == "6M":
                start_date = end_date - timedelta(days=180)
            elif time_period == "1Y":
                start_date = end_date - timedelta(days=365)
            else:  # "All"
                start_date = df.index[0]
            
            # Filter data for the selected time period
            mask = (df.index >= start_date) & (df.index <= end_date)
            filtered_df = df[mask]
            
            # Calculate indicators
            indicators = TechnicalIndicators().calculate_all_indicators(filtered_df)
            
            # Create price chart
            price_chart = self._create_candlestick_chart(filtered_df, indicators, selected_asset, start_date, end_date)
            st.plotly_chart(price_chart, use_container_width=True)
            
            # Display current values
            st.subheader("Current Values")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Price", f"${filtered_df['close'].iloc[-1]:.2f}")
                st.metric("RSI", f"{indicators['rsi'].iloc[-1]:.2f}")
                st.metric("MACD", f"{indicators['macd'].iloc[-1]:.2f}")
            
            with col2:
                st.metric("EMA 20", f"${indicators['ema_20'].iloc[-1]:.2f}")
                st.metric("EMA 50", f"${indicators['ema_50'].iloc[-1]:.2f}")
                st.metric("EMA 200", f"${indicators['ema_200'].iloc[-1]:.2f}")
            
            with col3:
                st.metric("BB Upper", f"${indicators['bb_upper'].iloc[-1]:.2f}")
                st.metric("BB Middle", f"${indicators['bb_middle'].iloc[-1]:.2f}")
                st.metric("BB Lower", f"${indicators['bb_lower'].iloc[-1]:.2f}")
            
            with col4:
                st.metric("ATR", f"${indicators['atr'].iloc[-1]:.2f}")
                st.metric("Volume", f"{filtered_df['volume'].iloc[-1]:,.0f}")
                st.metric("MACD Signal", f"{indicators['macd_signal'].iloc[-1]:.2f}")
            
            # Create tabs for technical indicators
            tab1, tab2, tab3 = st.tabs(["RSI", "MACD", "ATR"])
            
            # Create indicator charts
            rsi_fig, macd_fig, atr_fig = self._create_indicator_charts(filtered_df, indicators, start_date, end_date)
            
            # Display charts in respective tabs
            with tab1:
                st.plotly_chart(rsi_fig, use_container_width=True)
            with tab2:
                st.plotly_chart(macd_fig, use_container_width=True)
            with tab3:
                st.plotly_chart(atr_fig, use_container_width=True)
            
            # Add export button
            st.subheader("Export Data")
            if st.button("📥 Export Chart Data to CSV"):
                # Create export data
                export_df = self._create_export_data(filtered_df, indicators)
                
                # Generate filename
                filename = f"{selected_asset}_{time_period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                # Convert to CSV
                csv = export_df.to_csv(index=False)
                
                # Create download button
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )
            
            # Add a horizontal line to separate sections
            st.markdown("---")
            
            # Macro Market Analysis Section (Full Width)
            st.header("Macro Market Analysis")
            
            # Create two columns for the analysis button and latest analysis
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if st.button("Analyze Macro Headlines"):
                    with st.spinner("Analyzing macro headlines..."):
                        try:
                            # Analyze macro bias
                            biases = self.macro_analyzer.analyze_macro_bias()
                            st.session_state.macro_bias = biases
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
            
            with col2:
                if st.session_state.macro_bias:
                    st.write("Latest Analysis:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                else:
                    st.info("No macro bias analysis available. Click 'Analyze Macro Headlines' to generate one.")
            
            # Display macro bias analysis in a grid
            if st.session_state.macro_bias:
                cols = st.columns(len(st.session_state.macro_bias))
                for idx, (asset_type, bias) in enumerate(st.session_state.macro_bias.items()):
                    with cols[idx]:
                        st.subheader(f"{asset_type} Market")
                        st.metric("Sentiment", bias.sentiment)
                        st.metric("Impact", f"{bias.impact_score:.2f}")
                        st.metric("Confidence", f"{bias.confidence:.0%}")
                        if hasattr(bias, 'explanation'):
                            st.write("Explanation:", bias.explanation)
            
            # Add a horizontal line to separate sections
            st.markdown("---")
            
            # Trading Signals Section (Full Width)
            st.header("Trading Signals")
            
            # Asset selection for signal generation
            st.subheader("Asset Selection")
            signal_asset_types = st.multiselect(
                "Select Asset Types for Signal Generation",
                options=['crypto', 'forex', 'stocks'],
                default=['crypto', 'stocks']
            )
            
            if signal_asset_types:
                # Create a dictionary to store selected symbols for each asset type
                selected_symbols = {}
                
                # Display symbol selection in columns
                cols = st.columns(len(signal_asset_types))
                for idx, asset_type in enumerate(signal_asset_types):
                    with cols[idx]:
                        symbols = [asset['symbol'] for asset in self.config['assets'][asset_type]]
                        selected_symbols[asset_type] = st.multiselect(
                            f"{asset_type.title()} Symbols",
                            options=symbols,
                            default=symbols[:2]  # Default to first two symbols
                        )
            
            # Signal generation button
            if st.button("Generate Daily Signals"):
                if not signal_asset_types:
                    st.warning("Please select at least one asset type")
                else:
                    with st.spinner("Generating signals..."):
                        signals = []
                        final_signals = []
                        
                        # Generate signals only for selected assets
                        for asset_type in signal_asset_types:
                            if asset_type in selected_symbols and selected_symbols[asset_type]:
                                for symbol in selected_symbols[asset_type]:
                                    try:
                                        # Fetch data
                                        if asset_type == 'crypto':
                                            df = self.data_fetcher.fetch_crypto_data(symbol)
                                        elif asset_type == 'forex':
                                            df = self.data_fetcher.fetch_forex_data(symbol)
                                        else:
                                            df = self.data_fetcher.fetch_stock_data(symbol)
                                        
                                        # Calculate indicators
                                        df_with_indicators = self.signal_generator.calculate_indicators(df)
                                        
                                        # Generate signal
                                        signal = self._generate_signal(df_with_indicators, symbol)
                                        
                                        if signal:  # Only proceed if signal generation was successful
                                            # Get macro bias for the asset type
                                            asset_type_key = self._get_asset_type(symbol)
                                            macro_bias = st.session_state.macro_bias.get(asset_type_key) if st.session_state.macro_bias else None
                                            
                                            # Adjust confidence based on macro bias
                                            if macro_bias:
                                                signal = self._adjust_signal_confidence(signal, asset_type_key)
                                            
                                            if signal:  # Check if signal adjustment was successful
                                                # Add to signals list
                                                signals.append(signal)
                                                
                                                # Generate final signal using AI
                                                final_signal = self._generate_final_signal_with_ai(
                                                    signal,
                                                    macro_bias,
                                                    asset_type_key
                                                )
                                                
                                                if final_signal:  # Check if final signal generation was successful
                                                    final_signals.append({
                                                        'symbol': symbol,
                                                        'date': datetime.now().strftime('%Y-%m-%d'),
                                                        'signal': final_signal['signal'],
                                                        'confidence': final_signal['confidence'],
                                                        'technical_confidence': final_signal['technical_confidence'],
                                                        'macro_bias_impact': final_signal['macro_bias_impact'],
                                                        'confidence_adjustment': final_signal['confidence_adjustment'],
                                                        'explanation': final_signal['explanation'],
                                                        'risk_level': final_signal['risk_level']
                                                    })
                                        
                                    except Exception as e:
                                        st.error(f"Error generating signal for {symbol}: {str(e)}")
                        
                        # Update session state with new signals
                        st.session_state.generated_signals = signals
                        st.session_state.final_signals = final_signals
                        
                        # Display signals in a grid
                        if signals:
                            st.subheader("Generated Signals")
                            # Save signals to CSV
                            self._save_current_signals(signals)
                            
                            for signal in signals:
                                col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 2, 2, 2])
                                with col1:
                                    st.write(f"**{signal['symbol']}**")
                                    st.write(f"Time: {signal['timestamp']}")
                                with col2:
                                    signal_type = signal['signal']
                                    signal_color = self._get_signal_color(signal_type)
                                    st.markdown(f"<span style='color: {signal_color}; font-weight: bold;'>Signal: {signal_type.upper()}</span>", unsafe_allow_html=True)
                                with col3:
                                    # Display confidence value
                                    confidence = float(signal.get('confidence', 0.0))
                                    st.write("Confidence Score:")
                                    st.write(f"{confidence:.2f}")
                                with col4:
                                    if 'indicators' in signal:
                                        st.write("Technical Indicators:")
                                        st.write(f"RSI: {signal['indicators']['rsi']:.2f}")
                                        st.write(f"MACD: {signal['indicators']['macd']:.2f}")
                                        st.write(f"EMA20: {signal['indicators']['ema20']:.2f}")
                                        st.write(f"EMA50: {signal['indicators']['ema50']:.2f}")
                                with col5:
                                    if 'indicators' in signal:
                                        st.write("Bollinger Bands:")
                                        bb_upper = signal['indicators'].get('bb_upper')
                                        bb_middle = signal['indicators'].get('bb_middle')
                                        bb_lower = signal['indicators'].get('bb_lower')
                                        atr = signal['indicators'].get('atr')
                                        
                                        if bb_upper is not None:
                                            st.write(f"Upper: {bb_upper:.2f}")
                                        if bb_middle is not None:
                                            st.write(f"Middle: {bb_middle:.2f}")
                                        if bb_lower is not None:
                                            st.write(f"Lower: {bb_lower:.2f}")
                                        if atr is not None:
                                            st.write(f"ATR: {atr:.2f}")
                                with col6:
                                    if 'macro_bias' in signal:
                                        macro = signal['macro_bias']
                                        st.write("Macro Analysis:")
                                        st.write(f"Bias: {macro['sentiment']}")
                                        st.write(f"Impact: {macro['impact_score']:.2f}")
                                    else:
                                        st.write("No macro bias data")
                                st.divider()
                        else:
                            st.info("No signals generated yet. Select asset types and click 'Generate Daily Signals'.")
            
            # Add a horizontal line to separate sections
            st.markdown("---")
            
            # Final Signal Adjustment Section
            st.header("Final Signal Adjustment")
            
            if st.session_state.final_signals:
                # Display final signals in a grid
                st.subheader("AI-Generated Final Signals")
                
                # Group final signals by asset type
                final_signals_by_type = {}
                for signal in st.session_state.final_signals:
                    asset_type = self._get_asset_type(signal['symbol'])
                    if asset_type not in final_signals_by_type:
                        final_signals_by_type[asset_type] = []
                    final_signals_by_type[asset_type].append(signal)
                
                # Display final signals for each asset type
                for asset_type, type_signals in final_signals_by_type.items():
                    st.write(f"**{asset_type} Market Final Signals**")
                    for signal in type_signals:
                        with st.expander(f"{signal['symbol']} - {signal['signal'].upper()} (Confidence: {signal['confidence']:.2f})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Technical Confidence", f"{signal['technical_confidence']:.2f}")
                                st.metric("Macro Bias Impact", f"{signal['macro_bias_impact']:.2f}")
                                st.metric("Confidence Adjustment", f"{signal['confidence_adjustment']:.2f}")
                                st.metric("Final Confidence", f"{signal['confidence']:.2f}")
                            with col2:
                                st.metric("Risk Level", signal['risk_level'])
                                st.write("**Analysis:**")
                                st.write(signal['explanation'])
                    st.markdown("---")
                
                # Add export button for final signals
                if st.button("Save Final Signals to CSV"):
                    final_signals_df = pd.DataFrame(st.session_state.final_signals)
                    final_signals_df.to_csv('final_signals.csv', index=False)
                    st.success("Final signals saved to final_signals.csv")
            else:
                st.info("Generate signals first to see final adjusted signals")
            
            # Display historical signals if available
            if os.path.exists('signals.csv'):
                st.subheader("Historical Signals")
                historical_signals = pd.read_csv('signals.csv')
                st.dataframe(historical_signals, use_container_width=True)
            
            # Display historical final signals if available
            if os.path.exists('final_signals.csv'):
                st.subheader("Historical Final Signals")
                historical_final_signals = pd.read_csv('final_signals.csv')
                st.dataframe(historical_final_signals, use_container_width=True)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    def _save_macro_bias(self, macro_bias: dict):
        """Save macro bias analysis results to JSON file."""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_bias = {}
            for asset_type, bias in macro_bias.items():
                serializable_bias[asset_type] = {
                    'sentiment': bias['sentiment'],
                    'impact_score': float(bias['impact_score']),
                    'confidence': float(bias['confidence']),
                    'explanation': bias['explanation'],
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            with open('macro_bias.json', 'w') as f:
                json.dump(serializable_bias, f, indent=4)
        except Exception as e:
            st.error(f"Error saving macro bias: {str(e)}")

    def _save_current_signals(self, signals: list):
        """Save current generated signals to CSV file."""
        try:
            # Convert signals to DataFrame
            signals_data = []
            for signal in signals:
                signal_data = {
                    'symbol': signal['symbol'],
                    'signal': signal['signal'],
                    'confidence': signal['confidence'],
                    'timestamp': signal['timestamp'],
                    'rsi': signal['indicators']['rsi'],
                    'macd': signal['indicators']['macd'],
                    'ema20': signal['indicators']['ema20'],
                    'ema50': signal['indicators']['ema50'],
                    'bb_upper': signal['indicators']['bb_upper'],
                    'bb_middle': signal['indicators']['bb_middle'],
                    'bb_lower': signal['indicators']['bb_lower'],
                    'atr': signal['indicators']['atr'],
                    'atr_20ma': signal['indicators']['atr_20ma']
                }
                if 'macro_bias' in signal:
                    signal_data.update({
                        'macro_sentiment': signal['macro_bias']['sentiment'],
                        'macro_impact': signal['macro_bias']['impact_score']
                    })
                signals_data.append(signal_data)
            
            df = pd.DataFrame(signals_data)
            df.to_csv('signals.csv', index=False)
            st.success("Signals saved to signals.csv")
        except Exception as e:
            st.error(f"Error saving signals: {str(e)}")

    def analyze_macro_headlines(self):
        """Analyze macro headlines and update market bias."""
        try:
            with st.spinner("Analyzing macro headlines..."):
                # Get macro headlines
                headlines = self.macro_analyzer.get_macro_headlines()
                
                # Analyze headlines
                self.macro_bias = self.macro_analyzer.analyze_headlines(headlines)
                
                # Save macro bias results
                self._save_macro_bias(self.macro_bias)
                
                st.success("Macro analysis completed!")
        except Exception as e:
            st.error(f"Error analyzing macro headlines: {str(e)}")

    def generate_daily_signals(self):
        """Generate daily trading signals for selected asset types."""
        try:
            with st.spinner("Generating signals..."):
                signals = []
                for asset_type in self.selected_asset_types:
                    # Get data for each symbol in the asset type
                    for symbol in self.asset_symbols[asset_type]:
                        data = self.data_fetcher.get_data(symbol, asset_type)
                        if data is not None and not data.empty:
                            signal = self._generate_signal(data, symbol)
                            if signal is not None:
                                # Add macro bias if available
                                if self.macro_bias and asset_type in self.macro_bias:
                                    signal['macro_bias'] = {
                                        'sentiment': self.macro_bias[asset_type]['sentiment'],
                                        'impact_score': self.macro_bias[asset_type]['impact_score']
                                    }
                                signals.append(signal)
                
                # Save generated signals
                if signals:
                    self._save_current_signals(signals)
                
                return signals
        except Exception as e:
            st.error(f"Error generating signals: {str(e)}")
            return None

if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run() 