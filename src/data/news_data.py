from src.utils.embedding import EmbeddingManager
import datetime

def get_news_data(ticker):
    """
    Fetch news data and store embeddings for semantic search.
    Args:
        ticker (str): Stock ticker symbol (required).
    Returns:
        dict: News data with headlines and semantic search results.
    """
    # Placeholder for real news/research data fetching logic
    headlines = [
        f"{ticker} reports record quarterly earnings.",
        f"Analysts upgrade {ticker} stock outlook.",
        f"New product launch excites {ticker} investors.",
        f"{ticker} announces strategic partnership.",
        f"Market reacts positively to {ticker} earnings call."
    ]
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    
    # Add news to vector database with metadata
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "source": "news_api"
    }
    embedding_manager.add_news(ticker, headlines, metadata)
    
    # Get similar news for context (example search)
    similar_news = embedding_manager.search_similar_news(
        f"{ticker} earnings", 
        ticker=ticker, 
        n_results=3
    )
    
    return {
        "ticker": ticker,
        "headlines": headlines,
        "similar_news": similar_news,
        "embedding_manager": embedding_manager  # Pass for further use
    } 