import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.technical_agent import TechnicalAnalysisAgent
from src.agents.research_agent import ResearchAnalysisAgent
from src.agents.advisor_agent import InvestmentAdvisorAgent
from src.agents.social_sentiment_agent import SocialSentimentAgent
from src.data.market_data import get_market_data
from src.data.news_data import get_news_data
from src.data.social_sentiment_data import get_social_sentiment_data
from src.utils.technical_indicators import compute_macd, compute_rsi, compute_sma, compute_bollinger_bands

# Load environment variables
load_dotenv()

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
        
        # Calculate Bollinger Bands
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
        
        # MACD
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
        
        # RSI
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
        return None

def display_stock_info(ticker):
    """Display basic stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
            st.metric("Market Cap", f"${info.get('marketCap', 0):,}" if info.get('marketCap') else "N/A")
        
        with col2:
            st.metric("Volume", f"{info.get('volume', 0):,}" if info.get('volume') else "N/A")
            st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
        
        with col3:
            st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            st.metric("52 Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
            
    except Exception as e:
        st.error(f"Error fetching stock info: {e}")

def display_analysis_results(technical_result, research_result, social_result, advice):
    """Display all analysis results in organized sections"""
    
    # Technical Analysis
    if technical_result and not technical_result.get('error'):
        st.subheader("📊 Technical Analysis")
        st.write(technical_result.get("gemini_analysis") or technical_result)
    elif technical_result and technical_result.get('error'):
        st.error(f"Technical Analysis Error: {technical_result['error']}")
    
    # Research Analysis
    if research_result and not research_result.get('error'):
        st.subheader("📰 Research & News Analysis")
        st.write(research_result.get("gemini_news_analysis") or research_result)
    elif research_result and research_result.get('error'):
        st.error(f"Research Analysis Error: {research_result['error']}")
    
    # Social Sentiment Analysis
    if social_result and not social_result.get('error'):
        st.subheader("💬 Social Sentiment Analysis")
        st.write(social_result.get("gemini_social_sentiment") or social_result)
    elif social_result and social_result.get('error'):
        st.error(f"Social Sentiment Error: {social_result['error']}")
    
    # Investment Advice
    if advice and not advice.get('error'):
        st.subheader("💡 Investment Advice")
        st.info(advice.get("gemini_advice") or advice)
    elif advice and advice.get('error'):
        st.error(f"Investment Advice Error: {advice['error']}")

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Multi-Agent Stock Analyzer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🤖 Multi-Agent Stock Analyzer")
    st.markdown("Powered by Gemini LLM - Technical Analysis, Research, Social Sentiment & Investment Advice")
    
    # Initialize session state
    if 'ticker' not in st.session_state:
        st.session_state.ticker = ""
    
    # Sidebar
    st.sidebar.header("⚙️ Analysis Options")
    
    # Ticker input
    ticker = st.sidebar.text_input(
        "Enter Stock Ticker",
        value=st.session_state.ticker,
        placeholder="e.g., AAPL, TSLA, GOOGL",
        help="Enter a valid stock ticker symbol"
    ).upper().strip()
    
    # Update session state when ticker changes
    if ticker != st.session_state.ticker:
        st.session_state.ticker = ticker
    
    # Analysis options
    show_charts = st.sidebar.checkbox("Show Technical Charts", value=True)
    show_stock_info = st.sidebar.checkbox("Show Stock Information", value=True)
    show_analysis = st.sidebar.checkbox("Show AI Analysis", value=True)
    
    # Popular tickers for quick access
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Popular Stocks")
    popular_tickers = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "NVDA", "META", "NFLX"]
    
    cols = st.sidebar.columns(2)
    for i, pop_ticker in enumerate(popular_tickers):
        if cols[i % 2].button(pop_ticker, key=f"btn_{pop_ticker}"):
            st.session_state.ticker = pop_ticker
            st.rerun()
    
    # Main content
    if ticker:
        st.header(f"📈 Analysis for {ticker}")
        
        # Stock information
        if show_stock_info:
            with st.expander("📊 Stock Information", expanded=True):
                display_stock_info(ticker)
        
        # Analysis button
        analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("🤖 Multi-agent analysis in progress..."):
                try:
                    # Initialize agents
                    technical_agent = TechnicalAnalysisAgent()
                    research_agent = ResearchAnalysisAgent()
                    advisor_agent = InvestmentAdvisorAgent()
                    social_agent = SocialSentimentAgent()
                    
                    # Get data
                    market_data = get_market_data(ticker)
                    news_data = get_news_data(ticker)
                    sentiment_data = get_social_sentiment_data(ticker)
                    
                    if not market_data:
                        st.error(f"❌ Unable to fetch data for {ticker}. Please check the ticker symbol.")
                        return
                    
                    # Run analysis
                    technical_result = technical_agent.analyze(market_data)
                    research_result = research_agent.analyze(news_data)
                    social_result = social_agent.analyze(sentiment_data)
                    advice = advisor_agent.analyze(technical_result, research_result)
                    
                    # Display results
                    if show_analysis:
                        display_analysis_results(technical_result, research_result, social_result, advice)
                    
                    # Technical charts
                    if show_charts:
                        st.subheader("📊 Technical Analysis Charts")
                        chart = create_technical_charts(ticker)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        else:
                            st.warning("Unable to generate technical charts.")
                    
                    st.success("✅ Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("Please check your API keys and internet connection.")
    
    else:
        # Welcome message
        st.markdown("""
        ### 🎯 How to use this analyzer:
        
        1. **Enter a stock ticker** in the sidebar (e.g., AAPL, TSLA, GOOGL)
        2. **Configure analysis options** in the sidebar
        3. **Click 'Start Analysis'** to run the multi-agent analysis
        4. **Review results** including technical analysis, news sentiment, and investment advice
        
        ### 🔧 Features:
        - **Technical Analysis**: MACD, RSI, Bollinger Bands, and more
        - **News Analysis**: AI-powered news sentiment analysis
        - **Social Sentiment**: Social media sentiment analysis
        - **Investment Advice**: Comprehensive investment recommendations
        - **Interactive Charts**: Real-time technical analysis charts
        """)
        
        # Quick start section
        st.markdown("### 🚀 Quick Start")
        st.markdown("Try analyzing one of these popular stocks:")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("AAPL", use_container_width=True):
                st.session_state.ticker = "AAPL"
                st.rerun()
        with col2:
            if st.button("TSLA", use_container_width=True):
                st.session_state.ticker = "TSLA"
                st.rerun()
        with col3:
            if st.button("GOOGL", use_container_width=True):
                st.session_state.ticker = "GOOGL"
                st.rerun()
        with col4:
            if st.button("MSFT", use_container_width=True):
                st.session_state.ticker = "MSFT"
                st.rerun()

if __name__ == "__main__":
    main() 