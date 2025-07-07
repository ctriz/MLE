import os
import google.generativeai as genai
from .base_agent import BaseAgent
from dotenv import load_dotenv

load_dotenv()

class SocialSentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Social Sentiment Agent")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini API.")

    def analyze(self, sentiment_data):
        """
        Analyze social sentiment for a stock using Gemini LLM.
        Args:
            sentiment_data (dict): Should include 'ticker', 'posts' (list of strings), and 'related_terms'.
        Returns:
            dict: Sentiment analysis result.
        """
        ticker = sentiment_data.get("ticker")
        posts = sentiment_data.get("posts", [])
        related_terms = sentiment_data.get("related_terms", [ticker])
        
        if not ticker or not posts:
            return {"error": "Both 'ticker' and 'posts' are required in sentiment_data."}

        # Build context about related companies/brands
        related_context = ""
        if len(related_terms) > 1:
            related_context = f"\n\nNote: Analysis includes mentions of related brands/companies: {', '.join(related_terms)}"

        prompt = (
            f"You are a financial social sentiment analyst. "
            f"Analyze the following recent X (Twitter) posts about {ticker} and summarize the overall sentiment, key themes, and any notable opinions. "
            f"Posts: {posts}"
            f"{related_context}\n\n"
            f"Provide analysis covering:\n"
            f"1. Overall sentiment (positive/negative/neutral)\n"
            f"2. Key themes and topics discussed\n"
            f"3. Notable opinions or trends\n"
            f"4. Impact on brand perception (if applicable)\n"
            f"5. Potential market implications"
        )

        # Use the latest Gemini LLM API pattern
        try:
            if hasattr(genai, "GenerativeModel"):
                model = genai.GenerativeModel(self.gemini_model)
                response = model.generate_content(prompt)
                return {"gemini_social_sentiment": response.text, "prompt": prompt}
            else:
                return {"error": "No compatible GenerativeModel class found in google-generativeai SDK."}
        except Exception as e:
            return {"error": str(e), "prompt": prompt} 