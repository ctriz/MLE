# 🤖 Multi-Agent Stock Analyzer

🚀 A weekend experiment using CrewAI, Gemini LLM, and vector embeddings to orchestrate agent-based analysis of a stock ticker — combining technical indicators, news sentiment, and social media signals. The pipeline is modular and can be adapted to other research and monitoring use cases across industries.


## What It Does

- Retrieves and processes stock-related data
- Analyzes technical indicators
- Fetches news and social media content
- Applies sentiment analysis using LLM
- Summarizes insights via agent-based orchestration

## Tools Used

- CrewAI for multi-agent task orchestration
- Gemini (LLM) for summarization and sentiment classification
- Vector embeddings for semantic search and clustering

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