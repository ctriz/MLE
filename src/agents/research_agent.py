import os
import google.generativeai as genai
from .base_agent import BaseAgent
from src.data.news_data import get_news_data
from dotenv import load_dotenv

load_dotenv()

class ResearchAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Research Analysis Agent")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini API.")

    def analyze(self, news_data):
        """
        Analyze news headlines for a stock using Google Gemini LLM with semantic search context.
        Args:
            news_data (dict): Dictionary with news data including headlines and similar news.
        Returns:
            dict: Research analysis result with enhanced context.
        """
        ticker = news_data.get("ticker")
        headlines = news_data.get("headlines", [])
        similar_news = news_data.get("similar_news", [])
        
        if not ticker or not headlines:
            return {"error": "Both 'ticker' and 'headlines' are required in news_data."}

        # Build context from similar news
        similar_context = ""
        if similar_news:
            similar_context = "\n\nSimilar historical news:\n"
            for i, news in enumerate(similar_news[:3], 1):
                similar_context += f"{i}. {news.get('headline', 'N/A')}\n"

        prompt = (
            f"You are a financial research analyst. "
            f"Analyze the following recent news headlines for {ticker} and provide insights on sentiment, key themes, and potential market impact. "
            f"Current headlines:\n{chr(10).join([f'- {headline}' for headline in headlines])}"
            f"{similar_context}\n\n"
            f"Provide a comprehensive analysis including:\n"
            f"1. Overall sentiment assessment\n"
            f"2. Key themes and trends\n"
            f"3. Potential market impact\n"
            f"4. Historical context (if similar news found)\n"
            f"5. Risk factors to consider"
        )

        # Use the latest Gemini LLM API pattern
        try:
            if hasattr(genai, "GenerativeModel"):
                model = genai.GenerativeModel(self.gemini_model)
                response = model.generate_content(prompt)
                return {
                    "gemini_news_analysis": response.text, 
                    "prompt": prompt,
                    "similar_news_count": len(similar_news)
                }
            else:
                return {"error": "No compatible GenerativeModel class found in google-generativeai SDK."}
        except Exception as e:
            return {"error": str(e), "prompt": prompt} 