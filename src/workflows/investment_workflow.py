from src.agents.technical_agent import TechnicalAnalysisAgent
from src.agents.research_agent import ResearchAnalysisAgent
from src.agents.advisor_agent import InvestmentAdvisorAgent
from src.agents.social_sentiment_agent import SocialSentimentAgent
from src.data.market_data import get_market_data
from src.data.news_data import get_news_data
from src.data.social_sentiment_data import get_social_sentiment_data

def run_investment_workflow(ticker):
    technical_agent = TechnicalAnalysisAgent()
    research_agent = ResearchAnalysisAgent()
    advisor_agent = InvestmentAdvisorAgent()
    social_agent = SocialSentimentAgent()

    market_data = get_market_data(ticker)
    news_data = get_news_data(ticker)
    sentiment_data = get_social_sentiment_data(ticker)

    technical_result = technical_agent.analyze(market_data)
    print("Technical Analysis Result:", technical_result)
    research_result = research_agent.analyze(news_data)
    print("Research Analysis Result:", research_result)
    social_result = social_agent.analyze(sentiment_data)
    print("Social Sentiment Result:", social_result)
    advice = advisor_agent.analyze(technical_result, research_result)
    print("Investment Advice:", advice)

# Optionally, add a default runner for CLI usage
if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    run_investment_workflow(ticker) 