import os
import google.generativeai as genai
from .base_agent import BaseAgent
from src.data.market_data import get_market_data
from src.data.news_data import get_news_data
from dotenv import load_dotenv

load_dotenv()

class TechnicalAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Technical Analysis Agent")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        # Use GOOGLE_API_KEY for the latest Gemini API
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini API.")

    def analyze(self, market_data):
        """
        Analyze a stock using technical indicators and Google Gemini LLM.
        Args:
            market_data (dict): Dictionary with stock data including ticker, price, volume, and technical indicators.
        Returns:
            dict: Analysis result with technical signal and rationale.
        """
        # Check if market_data has an error
        if market_data.get("error"):
            return {"error": f"Market data error: {market_data['error']}"}
        
        ticker = market_data.get("ticker")
        if not ticker:
            return {"error": "Ticker symbol is required in market_data."}

        price = market_data.get("price")
        volume = market_data.get("volume")
        technicals = market_data.get("technicals", {})

        prompt = (
            f"You are a technical analysis expert. Analyze the following real-time data for {ticker}:\n"
            f"Current Price: ${price}\n"
            f"Volume: {volume:,}\n"
            f"Technical Indicators:\n"
            f"- RSI (14): {technicals.get('rsi', 'N/A')}\n"
            f"- MACD: {technicals.get('macd', 'N/A')}\n"
            f"- Signal Line: {technicals.get('signal_line', 'N/A')}\n"
            f"- Bollinger Bands Upper: ${technicals.get('bb_upper', 'N/A')}\n"
            f"- Bollinger Bands Lower: ${technicals.get('bb_lower', 'N/A')}\n"
            f"- SMA (20): ${technicals.get('sma_20', 'N/A')}\n"
            f"- SMA (50): ${technicals.get('sma_50', 'N/A')}\n\n"
            f"Provide a comprehensive technical analysis including:\n"
            f"1. Overall technical signal (Buy/Hold/Sell)\n"
            f"2. Key technical insights\n"
            f"3. Support and resistance levels\n"
            f"4. Risk assessment\n"
            f"5. Short-term price outlook"
        )

        # Use the latest Gemini LLM API pattern
        try:
            model = genai.GenerativeModel(self.gemini_model)
            response = model.generate_content(prompt)
            return {"gemini_analysis": response.text, "prompt": prompt}
        except Exception as e:
            return {"error": str(e), "prompt": prompt} 