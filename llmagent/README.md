# 🤖 Advanced Stock Analyst Agent (LLaMA-powered)

A comprehensive stock analysis application powered by LLaMA AI that provides technical analysis, sentiment analysis, and news insights for any stock ticker.

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
- **LLaMA AI Integration**: Advanced stock analysis using local LLaMA models
- **Sentiment Analysis**: TextBlob-powered sentiment scoring
- **Investment Insights**: AI-generated investment outlook and recommendations

### 📰 **News & Sentiment Analysis**
- **Recent News Headlines**: Latest news related to the stock
- **News Sentiment**: AI analysis of news impact on stock sentiment
- **Market Impact Assessment**: How news affects stock performance

### 🎯 **Investment Insights**
- **Combined Analysis**: Integration of technical, sentiment, and news data
- **Smart Recommendations**: AI-powered investment insights
- **Risk Assessment**: Technical indicator-based risk evaluation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- LLaMA model file (set in .env)
- Optional: News API key for enhanced news analysis

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd llm-stock-insights
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file with:
   ```env
   MODEL_PATH=/path/to/your/llama/model.gguf
   LLAMA_NEWS_API=https://your-llama-api-endpoint
   GEMINI_API_KEY=your-gemini-api-key
   ```

4. **Run the application**
   ```bash
   streamlit run stock_dashboard.py
   ```

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

## 🎨 UI Features

### Interactive Charts
- **Price & Bollinger Bands**: Main price chart with volatility bands
- **MACD Chart**: MACD line, signal line, and histogram
- **RSI Chart**: RSI with overbought/oversold levels

### Analysis Options
- **Sidebar Controls**: Toggle technical charts and news analysis
- **Responsive Layout**: Wide layout for better data visualization
- **Color-coded Indicators**: Visual feedback for different sentiment levels

### Data Presentation
- **Metrics Display**: Clean presentation of technical indicators
- **Sentiment Visualization**: Color-coded sentiment analysis
- **News Headlines**: Numbered list of recent news
- **Investment Insights**: Bullet-point summary of key findings

## 🔧 Configuration

### Environment Variables
- `MODEL_PATH`: Path to your LLaMA model file
- `LLAMA_NEWS_API`: API endpoint for news analysis
- `GEMINI_API_KEY`: Google Gemini API key for enhanced analysis

### Customization
- Modify technical analysis parameters in `utils/technical_indicators.py`
- Adjust sentiment thresholds in `utils/sentiment_analyzer.py`
- Customize news analysis prompts in `utils/news_analyzer.py`
- Use `utils/news_analyzer.py` for CLI-based news analysis

## 📁 Project Structure

```
llm-stock-insights/
├── stock_dashboard.py              # Main Streamlit dashboard
├── requirements.txt                # Python dependencies
├── utils/
│   ├── stock_analyzer.py          # Main orchestration logic
│   ├── market_data.py             # Market data fetching
│   ├── technical_indicators.py    # Technical indicators
│   ├── sentiment_analyzer.py      # Sentiment analysis
│   ├── llm_engine.py              # LLM model management
│   ├── news_analyzer.py           # News analysis with NewsAPI & Gemini (includes CLI)
│   └── ...
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

*Powered by LLaMA AI and real-time market data* 🚀 