import os
import google.generativeai as genai
from .base_agent import BaseAgent

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
            sentiment_data (dict): Should include 'ticker' and 'posts' (list of strings).
        Returns:
            dict: Sentiment analysis result.
        """
        ticker = sentiment_data.get("ticker")
        posts = sentiment_data.get("posts", [])
        if not ticker or not posts:
            return {"error": "Both 'ticker' and 'posts' are required in sentiment_data."}

        prompt = (
            f"You are a financial social sentiment analyst. "
            f"Analyze the following recent X (Twitter) posts about {ticker} and summarize the overall sentiment, key themes, and any notable opinions. "
            f"Posts: {posts}"
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