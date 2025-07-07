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