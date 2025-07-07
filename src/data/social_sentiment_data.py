def get_social_sentiment_data(ticker="AAPL"):
    """
    Fetch social sentiment data with support for parent company relationships.
    Args:
        ticker (str): Stock ticker symbol.
    Returns:
        dict: Social sentiment data with posts and related terms.
    """
    from src.utils.embedding import EmbeddingManager
    
    # Initialize embedding manager to get company relationships
    embedding_manager = EmbeddingManager()
    
    # Get related terms using vector similarity
    related_terms = embedding_manager.get_company_relationships(ticker)
    
    # Placeholder for real X (Twitter) data fetching logic
    # In a real implementation, you would search for posts containing these terms
    posts = [
        f"${ticker} is trending! Lots of bullish talk today.",
        f"Some investors are worried about supply chain issues for {ticker}.",
        f"{ticker}'s new product launch is getting a lot of positive buzz."
    ]
    
    # Add posts mentioning related brands/companies based on vector relationships
    if len(related_terms) > 1:
        # Generate posts mentioning related terms
        for term in related_terms[:3]:  # Use top 3 related terms
            if term.lower() != ticker.lower():
                posts.append(f"{term} is getting attention in social media discussions.")
                posts.append(f"Investors are discussing {term}'s impact on {ticker} stock.")
    
    return {
        "ticker": ticker,
        "posts": posts,
        "related_terms": related_terms,
        "search_terms": [ticker] + related_terms
    } 