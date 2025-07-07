# 🤖 Multi-Agent Stock Analyzer

A comprehensive stock analysis application powered by Google Gemini LLM that provides technical analysis, sentiment analysis, news insights, and investment advice for any stock ticker.

## ✨ Features

### 📈 **Real-time Stock Data**
- Current price, change, and percentage change
- Historical price data visualization
- Market information and statistics

### 📊 **Technical Analysis**
- **MACD**: Trend-following momentum indicator
- **RSI**: Momentum oscillator (0-100 scale)
- **Bollinger Bands**: Volatility indicator
- **SMA**: 20-day moving average
- **Interactive Charts**: Plotly-powered technical analysis charts

### 🧠 **AI-Powered Analysis**
- **Multi-Agent System**: Technical, Research, Social Sentiment, and Investment Advisor agents
- **Gemini LLM Integration**: Advanced stock analysis using Google Gemini models
- **Sentiment Analysis**: AI analysis of news and social media sentiment
- **Investment Insights**: AI-generated investment recommendations

### 🔍 **Vectorization & Semantic Search**
- **Text Embeddings**: Vector embeddings using sentence-transformers
- **Semantic Search**: Context-aware news search using vector similarity
- **ChromaDB Integration**: Vector database for news embeddings
- **Enhanced Context**: Provide AI agents with relevant historical context

### 📰 **News & Sentiment Analysis**
- **Recent News Headlines**: Latest news related to the stock
- **News Sentiment**: AI analysis of news impact on stock sentiment
- **Social Media Sentiment**: Analysis of X (Twitter) posts
- **Semantic News Search**: Find similar historical news using embeddings

## 🖥️ Usage

### GUI Mode (Recommended)
```bash
python main.py
```
The GUI will open in your browser at `http://localhost:8501`

### CLI Mode
```bash
python main.py AAPL
python main.py TSLA
python main.py --help
```

## 📁 Project Structure

```
stock-analyzer/
├── main.py                      # Main entry point
├── src/
│   ├── gui.py                   # Streamlit GUI
│   ├── agents/                  # Multi-agent system
│   ├── data/                    # Data utilities
│   ├── utils/                   # Shared utilities
│   └── workflows/               # Workflow orchestration
├── tests/                       # Test files
├── chroma_db/                   # Vector database storage
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

*Powered by Google Gemini AI, Vector Embeddings, and real-time market data* 🚀 