import streamlit as st
from dotenv import load_dotenv
from utils.stock_analyzer import get_stock_insight
from utils.llm_engine import load_llama_model
from utils.technical_indicators import compute_macd, compute_rsi, compute_sma, compute_bollinger_bands
import plotly.graph_objects as go

from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf

# Load environment variables 

load_dotenv()

def get_model():
    import os
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        raise EnvironmentError("MODEL_PATH is not set in the .env file.")
    return load_llama_model(model_path)

def create_technical_charts(ticker):
    """Create interactive technical analysis charts"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo", interval="1d")
        
        if df.empty:
            return None
            
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Bollinger Bands', 'MACD', 'RSI'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Price and Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Calculate Bollinger Bands using utility function
        sma_20, upper_band, lower_band = compute_bollinger_bands(df)
        
        fig.add_trace(
            go.Scatter(x=df.index, y=upper_band, name='Upper BB', line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=lower_band, name='Lower BB', line=dict(color='gray', dash='dash'), fill='tonexty'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=sma_20, name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
        
        # MACD using utility function
        macd, signal, hist = compute_macd(df)
        
        fig.add_trace(
            go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=signal, name='Signal', line=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=df.index, y=hist, name='Histogram', marker_color='gray'),
            row=2, col=1
        )
        
        # RSI using utility function
        rsi = compute_rsi(df)
        
        fig.add_trace(
            go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        # Add horizontal lines for RSI overbought/oversold levels
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
        
        fig.update_layout(height=800, title_text=f"Technical Analysis for {ticker.upper()}")
        return fig
        
    except Exception as e:
        st.error(f"Error creating charts: {e}")
        st.error("This might be due to insufficient data or network issues. Try a different ticker or check your internet connection.")
        return None

def display_news_sentiment(news_data):
    """Display news headlines and sentiment analysis"""
    if not news_data or not news_data.get('headlines'):
        st.warning("No recent news found for this stock.")
        return
    
    st.subheader("📰 Recent News & Sentiment")
    
    # Display headlines
    st.write("**Recent Headlines:**")
    for i, headline in enumerate(news_data['headlines'], 1):
        st.write(f"{i}. {headline}")
    
    # Display news analysis
    if news_data.get('news_analysis'):
        st.write("**News Sentiment Analysis:**")
        st.info(news_data['news_analysis'])

def display_technical_indicators(technicals):
    """Display technical indicators in a clean format"""
    if not technicals or technicals.get('error'):
        st.error(f"Technical analysis error: {technicals.get('error', 'Unknown error')}")
        return
    
    st.subheader("📊 Technical Indicators")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("MACD", f"{technicals['macd']:.2f}")
        st.metric("Signal Line", f"{technicals['signal_line']:.2f}")
    
    with col2:
        st.metric("RSI", f"{technicals['rsi']:.2f}")
        st.metric("SMA (20)", f"${technicals['sma_20']:.2f}")
    
    with col3:
        st.metric("BB Upper", f"${technicals['bb_upper']:.2f}")
        st.metric("BB Lower", f"${technicals['bb_lower']:.2f}")
    
    # Trend indicator
    trend = technicals.get('trend', 'neutral')
    if trend == 'bullish':
        st.success(f"🎯 Technical Trend: {trend.upper()}")
    elif trend == 'bearish':
        st.error(f"🎯 Technical Trend: {trend.upper()}")
    else:
        st.info(f"🎯 Technical Trend: {trend.upper()}")

# Try loading model once for reuse
try:
    llm = get_model()
except Exception as e:
    st.error(f"❌ Failed to load LLaMA model: {e}")
    st.stop()

# Streamlit app
st.set_page_config(page_title="LLaMA Stock Agent", layout="wide")
st.title("🤖 Advanced Stock Analyst Agent (LLaMA-powered)")

# Sidebar for additional options
st.sidebar.header("Analysis Options")
show_charts = st.sidebar.checkbox("Show Technical Charts", value=True)
show_news = st.sidebar.checkbox("Show News Analysis", value=True)

ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA):")

if ticker:
    with st.spinner("Agent analyzing stock data, technical indicators, and news sentiment..."):
        result = get_stock_insight(ticker, model=llm)

        if result.get("error"):
            st.error(result["error"])
        else:
            # Stock price data
            stock = result["stock"]
            st.subheader(f"📈 {result['ticker']} Price Data")
            
            # Fix linter errors by ensuring stock is a dictionary
            if isinstance(stock, dict):
                price = stock.get('price', 'N/A')
                change = stock.get('change', 'N/A')
                percent_change = stock.get('percent_change', 'N/A')
                
                st.metric(
                    label="Current Price",
                    value=f"${price}" if price != 'N/A' else price,
                    delta=f"{change} ({percent_change}%)" if change != 'N/A' else None
                )
            else:
                st.error("Invalid stock data format")

            # LLaMA Analysis
            st.subheader("🧠 LLaMA Analysis")
            st.write(result["comment"])

            # Sentiment Analysis
            st.subheader("🔍 Sentiment Analysis")
            sentiment = result["sentiment"]
            if sentiment == "positive":
                st.success(f"Sentiment: {sentiment.capitalize()}")
            elif sentiment == "negative":
                st.error(f"Sentiment: {sentiment.capitalize()}")
            else:
                st.info(f"Sentiment: {sentiment.capitalize()}")

            # Technical Analysis
            if result.get("technicals"):
                display_technical_indicators(result["technicals"])

            # Technical Charts
            if show_charts:
                st.subheader("📈 Technical Analysis Charts")
                chart = create_technical_charts(ticker)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.warning("Unable to generate technical charts.")

            # News Analysis
            if show_news and result.get("news"):
                display_news_sentiment(result["news"])

            # Additional insights section
            st.subheader("💡 Investment Insights")
            
            # Combine all analysis for final insights
            insights = []
            
            # Price movement insight
            if isinstance(stock, dict) and stock.get('percent_change'):
                try:
                    pct_change = float(stock['percent_change'].replace('%', ''))
                    if pct_change > 2:
                        insights.append("📈 Strong positive price movement today")
                    elif pct_change < -2:
                        insights.append("📉 Significant price decline today")
                    else:
                        insights.append("➡️ Moderate price movement today")
                except:
                    pass
            
            # Technical trend insight
            technicals = result.get("technicals")
            if technicals and isinstance(technicals, dict) and technicals.get("trend"):
                trend = technicals["trend"]
                if trend == "bullish":
                    insights.append("🚀 Technical indicators suggest bullish momentum")
                elif trend == "bearish":
                    insights.append("⚠️ Technical indicators suggest bearish pressure")
                else:
                    insights.append("⚖️ Technical indicators are neutral")
            
            # RSI insight
            if technicals and isinstance(technicals, dict) and technicals.get("rsi"):
                rsi = technicals["rsi"]
                try:
                    rsi_value = float(rsi)
                    if rsi_value > 70:
                        insights.append("🔥 RSI indicates overbought conditions")
                    elif rsi_value < 30:
                        insights.append("❄️ RSI indicates oversold conditions")
                    else:
                        insights.append("✅ RSI is in normal range")
                except (ValueError, TypeError):
                    pass  # Skip if RSI is not a valid number
            
            # Display insights
            for insight in insights:
                st.write(f"• {insight}")

# Footer
st.markdown("---")
st.markdown("*Powered by LLaMA AI and real-time market data*")
