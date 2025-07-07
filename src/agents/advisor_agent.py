import os
import google.generativeai as genai
from .base_agent import BaseAgent
from dotenv import load_dotenv

load_dotenv()

class InvestmentAdvisorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Investment Advisor Agent")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini API.")

    def analyze(self, technical_result, research_result):
        """
        Synthesize technical and research analysis using Google Gemini LLM to provide investment advice.
        Args:
            technical_result (dict): Output from technical analysis agent.
            research_result (dict): Output from research analysis agent.
        Returns:
            dict: Investment advice from Gemini LLM.
        """
        # Extract relevant details
        technical_summary = technical_result.get("gemini_analysis") or str(technical_result)
        research_summary = research_result.get("gemini_news_analysis") or str(research_result)
        ticker = technical_result.get("ticker") or research_result.get("ticker") or "the stock"

        prompt = (
            f"You are an expert investment advisor. Given the following technical analysis and research/news analysis for {ticker}, "
            f"provide a clear investment recommendation (buy/sell/hold) and a concise rationale.\n"
            f"Technical Analysis Summary:\n{technical_summary}\n"
            f"Research/News Analysis Summary:\n{research_summary}\n"
        )

        try:
            if hasattr(genai, "Gemini"):
                model = genai.Gemini(self.gemini_model)
            elif hasattr(genai, "GenerativeModel"):
                model = genai.GenerativeModel(self.gemini_model)
            else:
                return {"error": "No Gemini model class found in google-generativeai. Please update your SDK."}
            response = model.generate_content(prompt)
            return {
                "gemini_advice": response.text,
                "prompt": prompt,
                "technical_summary": technical_summary,
                "research_summary": research_summary
            }
        except Exception as e:
            return {"error": f"Gemini API error: {str(e)}"} 