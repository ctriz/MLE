from utils.llm_engine import load_llama_model, generate_stock_comment
from utils.market_data import get_stock_data
from utils.sentiment_analyzer import analyze_sentiment
from utils.technical_indicators import get_technical_analysis
from utils.news_analyzer import analyze_news_with_gemini_agent
from dotenv import load_dotenv
import os


def get_stock_insight(ticker: str, model=None):
    """
    Orchestrates stock analysis using LLaMA, stock data, sentiment, and technical indicators.

    Returns:
        dict with stock data, LLaMA comment, sentiment, and technical analysis.
    """
    # Load config
    load_dotenv()

    stock_info = get_stock_data(ticker)
    if not stock_info:
        return {"error": f"No stock data found for {ticker.upper()}"}

    # Load LLaMA model if not provided
    if model is None:
        load_dotenv()
        model_path = os.getenv("MODEL_PATH")
        if not model_path:
            return {"error": "MODEL_PATH is not set in the environment."}
        model = load_llama_model(model_path)

    # Generate LLaMA comment
    prompt = f"Analyze the stock performance of {ticker} today. Give an investment outlook."
    comment = generate_stock_comment(model, prompt)

    # Analyze sentiment
    sentiment = analyze_sentiment(comment)

    # Perform technical analysis
    technicals = get_technical_analysis(ticker)

    # News analysis using Gemini
    news = analyze_news_with_gemini_agent(ticker)
    return {
        "ticker": ticker.upper(),
        "stock": stock_info,
        "comment": comment,
        "sentiment": sentiment,
        "technicals": technicals,
        "news": news
    }
