from src.utils.embedding import EmbeddingManager

def format_output(data):
    # Placeholder for formatting or shared utilities
    return str(data)

def add_company_relationship(ticker: str, new_terms: list):
    """
    Dynamically add new relationship terms to a company.
    Args:
        ticker (str): Stock ticker symbol.
        new_terms (list): List of new related terms to add.
    """
    embedding_manager = EmbeddingManager()
    
    for term in new_terms:
        embedding_manager.add_dynamic_relationship(ticker, term)
    
    print(f"✅ Added {len(new_terms)} new terms to {ticker} relationships")

def search_companies_by_term(search_term: str, n_results: int = 3):
    """
    Search for companies related to a specific term.
    Args:
        search_term (str): Term to search for (e.g., "Facebook", "iPhone").
        n_results (int): Number of results to return.
    Returns:
        List of related companies with metadata.
    """
    embedding_manager = EmbeddingManager()
    return embedding_manager.search_companies_by_term(search_term, n_results)

def get_company_relationships(ticker: str):
    """
    Get all related terms for a company.
    Args:
        ticker (str): Stock ticker symbol.
    Returns:
        List of related terms.
    """
    embedding_manager = EmbeddingManager()
    return embedding_manager.get_company_relationships(ticker) 