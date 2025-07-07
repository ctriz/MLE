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
    # Initialize embedding manager to get company relationships
    embedding_manager = EmbeddingManager()
    
    # Get related terms using vector similarity
    related_terms = embedding_manager.get_company_relationships(ticker)
    
    # Placeholder for real news/research data fetching logic
    # In a real implementation, you would search news APIs for these terms
    headlines = [
        f"{ticker} reports record quarterly earnings.",
        f"Analysts upgrade {ticker} stock outlook.",
        f"New product launch excites {ticker} investors.",
        f"{ticker} announces strategic partnership.",
        f"Market reacts positively to {ticker} earnings call."
    ]
    
    # Add headlines mentioning related brands/companies based on vector relationships
    if len(related_terms) > 1:
        # Generate headlines mentioning related terms
        for term in related_terms[:3]:  # Use top 3 related terms
            if term.lower() != ticker.lower():
                headlines.append(f"{term} contributes to {ticker}'s strong performance.")
                headlines.append(f"Analysts focus on {term}'s role in {ticker} growth strategy.")
    
    # Add news to vector database with metadata
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "source": "news_api",
        "related_terms": related_terms
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
        "related_terms": related_terms,
        "embedding_manager": embedding_manager  # Pass for further use
    } 