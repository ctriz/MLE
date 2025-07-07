# 🤖 Multi-Agent Stock Analyzer

A comprehensive stock analysis application powered by Google Gemini LLM that provides technical analysis, sentiment analysis, news insights, and investment advice for any stock ticker.

## ✨ Features

### 📈 **Real-time Stock Data**
- Current price, change, and percentage change
- Historical price data visualization
- Market information and statistics

### 📊 **Technical Analysis**
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
- **RSI (Relative Strength Index)**: Momentum oscillator measuring speed and magnitude of price changes
- **Bollinger Bands**: Volatility indicator with upper and lower bands
- **SMA (Simple Moving Average)**: 20-day moving average
- **Interactive Charts**: Plotly-powered charts with multiple technical indicators
- **Trend Signals**: Bullish, bearish, or neutral trend analysis

### 🧠 **AI-Powered Analysis**
- **Multi-Agent System**: Technical, Research, Social Sentiment, and Investment Advisor agents
- **Gemini LLM Integration**: Advanced stock analysis using Google Gemini models
- **Sentiment Analysis**: AI analysis of news and social media sentiment
- **Investment Insights**: AI-generated investment outlook and recommendations

### 📰 **News & Sentiment Analysis**
- **Recent News Headlines**: Latest news related to the stock
- **News Sentiment**: AI analysis of news impact on stock sentiment
- **Social Media Sentiment**: Analysis of X (Twitter) posts and social sentiment
- **Market Impact Assessment**: How news affects stock performance

### 🎯 **Investment Insights**
- **Combined Analysis**: Integration of technical, sentiment, and news data
- **Smart Recommendations**: AI-powered investment insights
- **Risk Assessment**: Technical indicator-based risk evaluation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key (set in .env)
- Optional: News API key for enhanced news analysis

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file with:
   ```env
   GOOGLE_API_KEY=your-gemini-api-key
   GEMINI_MODEL=gemini-1.5-flash
   NEWS_API_KEY=your-news-api-key
   ```

## 🖥️ Usage

### GUI Mode (Recommended)
Start the web-based GUI application:

```bash
# Start GUI (default)
python main.py

# Or explicitly start GUI
python main.py --gui
python main.py --streamlit
```

The GUI will open in your browser at `http://localhost:8501`

### CLI Mode
Run analysis directly from command line:

```bash
# Analyze a specific stock
python main.py AAPL
python main.py TSLA
python main.py GOOGL

# Get help
python main.py --help
```

## 🎨 GUI Features

### Interactive Interface
- **Stock Ticker Input**: Enter any stock symbol
- **Quick Access Buttons**: Popular stocks (AAPL, TSLA, GOOGL, etc.)
- **Analysis Options**: Toggle different analysis components
- **Real-time Charts**: Interactive technical analysis charts

### Analysis Sections
1. **Stock Information**: Current price, market cap, volume, P/E ratio
2. **Technical Analysis**: MACD, RSI, Bollinger Bands with AI insights
3. **Research Analysis**: News sentiment and market impact analysis
4. **Social Sentiment**: Social media sentiment analysis
5. **Investment Advice**: Comprehensive investment recommendations

### Technical Charts
- **Price & Bollinger Bands**: Main price chart with volatility bands
- **MACD Chart**: MACD line, signal line, and histogram
- **RSI Chart**: RSI with overbought/oversold levels

## 📊 Technical Indicators Explained

### MACD
- **What it measures**: Trend-following momentum indicator
- **Interpretation**: 
  - MACD > Signal Line = Bullish
  - MACD < Signal Line = Bearish
  - Histogram shows momentum strength

### RSI (Relative Strength Index)
- **What it measures**: Momentum oscillator (0-100 scale)
- **Interpretation**:
  - RSI > 70 = Overbought (potential sell signal)
  - RSI < 30 = Oversold (potential buy signal)
  - 30-70 = Normal range

### Bollinger Bands
- **What it measures**: Volatility and price levels
- **Interpretation**:
  - Price near upper band = Potentially overbought
  - Price near lower band = Potentially oversold
  - Band width indicates volatility

### SMA (Simple Moving Average)
- **What it measures**: Average price over 20 days
- **Interpretation**:
  - Price > SMA = Uptrend
  - Price < SMA = Downtrend

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Google Gemini API key (required)
- `GEMINI_MODEL`: Gemini model to use (default: gemini-1.5-flash)
- `NEWS_API_KEY`: News API key for enhanced news analysis (optional)

### Customization
- Modify technical analysis parameters in `src/utils/technical_indicators.py`
- Adjust sentiment thresholds in `src/utils/sentiment_analyzer.py`
- Customize news analysis prompts in `src/agents/research_agent.py`

## 📁 Project Structure

```
stock-analyzer/
├── main.py                      # Main entry point (GUI/CLI)
├── src/
│   ├── gui.py                   # Streamlit GUI application
│   ├── agents/                  # Multi-agent system
│   │   ├── technical_agent.py   # Technical analysis agent
│   │   ├── research_agent.py    # Research/news analysis agent
│   │   ├── advisor_agent.py     # Investment advisor agent
│   │   └── social_sentiment_agent.py # Social sentiment agent
│   ├── data/                    # Data utilities
│   │   ├── market_data.py       # Market data fetching
│   │   ├── news_data.py         # News data processing
│   │   └── social_sentiment_data.py # Social sentiment data
│   ├── utils/                   # Shared utilities
│   │   ├── technical_indicators.py # Technical analysis functions
│   │   ├── embedding.py         # Vector embeddings
│   │   └── helpers.py           # Helper functions
│   └── workflows/               # Workflow orchestration
│       └── investment_workflow.py # Main analysis workflow
├── tests/                       # Test files
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This application is for educational and research purposes only. It does not constitute financial advice. Always do your own research and consult with financial professionals before making investment decisions.

---

*Powered by Google Gemini AI and real-time market data* 🚀 