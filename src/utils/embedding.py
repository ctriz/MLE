import chromadb
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any
import uuid

class EmbeddingManager:
    def __init__(self, collection_name="news_embeddings"):
        """
        Initialize embedding manager with sentence-transformers and ChromaDB.
        Args:
            collection_name (str): Name of the ChromaDB collection.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize company relationships collection
        self.company_collection = self.client.get_or_create_collection(
            name="company_relationships",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize with default company relationships
        self._initialize_company_relationships()
    
    def _initialize_company_relationships(self):
        """Initialize default company relationships in the vector database."""
        default_relationships = {
            "META": ["Facebook", "Instagram", "WhatsApp", "Meta", "Oculus", "Threads", "Reels"],
            "GOOGL": ["Google", "Alphabet", "YouTube", "Android", "Chrome", "Gmail", "Maps"],
            "AMZN": ["Amazon", "AWS", "Prime", "Alexa", "Kindle", "Whole Foods"],
            "MSFT": ["Microsoft", "Windows", "Office", "Azure", "Xbox", "Teams", "LinkedIn"],
            "AAPL": ["Apple", "iPhone", "iPad", "Mac", "iOS", "Siri", "iCloud"],
            "TSLA": ["Tesla", "Elon Musk", "Model S", "Model 3", "Model X", "Model Y", "Cybertruck"],
            "NVDA": ["NVIDIA", "Nvidia", "GPU", "AI chips", "CUDA", "GeForce"],
            "NFLX": ["Netflix", "streaming", "original content", "subscription"]
        }
        
        for ticker, related_terms in default_relationships.items():
            self.add_company_relationship(ticker, related_terms)
    
    def add_company_relationship(self, ticker: str, related_terms: List[str]):
        """
        Add company relationship to the vector database.
        Args:
            ticker (str): Stock ticker symbol.
            related_terms (List[str]): List of related company names, brands, or products.
        """
        # Create a combined text for embedding
        combined_text = f"{ticker} {' '.join(related_terms)}"
        embedding = self.get_embedding(combined_text)
        
        # Create metadata
        metadata = {
            "ticker": ticker,
            "related_terms": related_terms,
            "term_count": len(related_terms)
        }
        
        # Add to company relationships collection
        self.company_collection.add(
            embeddings=[embedding],
            documents=[combined_text],
            metadatas=[metadata],
            ids=[f"company_{ticker}"]
        )
    
    def get_company_relationships(self, ticker: str) -> List[str]:
        """
        Get related terms for a company using vector similarity.
        Args:
            ticker (str): Stock ticker symbol.
        Returns:
            List[str]: List of related terms.
        """
        # Search for company relationships
        query_embedding = self.get_embedding(ticker)
        
        results = self.company_collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            where={"ticker": ticker}
        )
        
        if results['metadatas'] and results['metadatas'][0]:
            return results['metadatas'][0][0].get("related_terms", [ticker])
        
        return [ticker]
    
    def search_companies_by_term(self, search_term: str, n_results: int = 3) -> List[Dict]:
        """
        Search for companies related to a specific term.
        Args:
            search_term (str): Term to search for (e.g., "Facebook", "iPhone").
            n_results (int): Number of results to return.
        Returns:
            List[Dict]: Related companies with metadata.
        """
        query_embedding = self.get_embedding(search_term)
        
        results = self.company_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                result = {
                    "ticker": results['metadatas'][0][i].get("ticker"),
                    "related_terms": results['metadatas'][0][i].get("related_terms", []),
                    "similarity": 1 - results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def add_dynamic_relationship(self, ticker: str, new_term: str):
        """
        Dynamically add a new relationship term to an existing company.
        Args:
            ticker (str): Stock ticker symbol.
            new_term (str): New related term to add.
        """
        current_terms = self.get_company_relationships(ticker)
        if new_term not in current_terms:
            current_terms.append(new_term)
            self.add_company_relationship(ticker, current_terms)
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a given text.
        Args:
            text (str): Text to embed.
        Returns:
            List[float]: Embedding vector.
        """
        return self.model.encode(text).tolist()
    
    def add_news(self, ticker: str, headlines: List[str], metadata: Dict[str, Any] = None):
        """
        Add news headlines to the vector database.
        Args:
            ticker (str): Stock ticker.
            headlines (List[str]): List of news headlines.
            metadata (Dict[str, Any]): Additional metadata.
        """
        if not headlines:
            return
        
        # Generate embeddings for all headlines
        embeddings = [self.get_embedding(headline) for headline in headlines]
        
        # Create unique IDs for each headline
        ids = [str(uuid.uuid4()) for _ in headlines]
        
        # Prepare metadata
        metadatas = []
        for i, headline in enumerate(headlines):
            meta = {
                "ticker": ticker,
                "headline": headline,
                "timestamp": metadata.get("timestamp", "") if metadata else "",
                "source": metadata.get("source", "news") if metadata else "news"
            }
            metadatas.append(meta)
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=headlines,
            metadatas=metadatas,
            ids=ids
        )
    
    def search_similar_news(self, query: str, ticker: str = None, n_results: int = 5) -> List[Dict]:
        """
        Search for similar news headlines.
        Args:
            query (str): Search query.
            ticker (str): Optional ticker filter.
            n_results (int): Number of results to return.
        Returns:
            List[Dict]: Similar news with metadata.
        """
        query_embedding = self.get_embedding(query)
        
        # Build where clause if ticker is specified
        where_clause = {"ticker": ticker} if ticker else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                result = {
                    "headline": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def get_news_summary(self, ticker: str, n_results: int = 10) -> List[Dict]:
        """
        Get recent news summary for a ticker.
        Args:
            ticker (str): Stock ticker.
            n_results (int): Number of results to return.
        Returns:
            List[Dict]: Recent news with metadata.
        """
        results = self.collection.get(
            where={"ticker": ticker},
            limit=n_results
        )
        
        formatted_results = []
        if results['documents']:
            for i in range(len(results['documents'])):
                result = {
                    "headline": results['documents'][i],
                    "metadata": results['metadatas'][i]
                }
                formatted_results.append(result)
        
        return formatted_results 