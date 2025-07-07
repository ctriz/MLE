from crewai import Agent
from textwrap import dedent

from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools
from tools.sec_tools import SECTools


class StockAnalysisAgents:
    def research_analyst(self):
        return Agent(
            role='Research Analyst',
            goal='Conduct thorough research on companies and market conditions',
            backstory=dedent("""
                You are an expert research analyst with over 15 years of experience 
                in financial markets. You specialize in gathering comprehensive 
                information about companies, their business models, market position, 
                and industry trends. You have access to various data sources and 
                can provide detailed insights about company fundamentals.
            """),
            verbose=True,
            allow_delegation=False,
            tools=[
                SearchTools.search_internet,
                SearchTools.search_news,
                BrowserTools.scrape_and_summarize_website
            ]
        )

    def financial_analyst(self):
        return Agent(
            role='Financial Analyst',
            goal='Analyze financial statements, ratios, and SEC filings',
            backstory=dedent("""
                You are a senior financial analyst with expertise in financial 
                statement analysis, ratio analysis, and SEC filings interpretation. 
                You have a deep understanding of accounting principles, financial 
                metrics, and regulatory requirements. You can identify trends, 
                risks, and opportunities from financial data.
            """),
            verbose=True,
            allow_delegation=False,
            tools=[
                CalculatorTools.calculate,
                SECTools.search_10k,
                SECTools.search_10q,
                SECTools.get_filing_summary
            ]
        )

    def investment_advisor(self):
        return Agent(
            role='Investment Advisor',
            goal='Provide investment recommendations based on comprehensive analysis',
            backstory=dedent("""
                You are a seasoned investment advisor with 20+ years of experience 
                in portfolio management and investment strategy. You have helped 
                numerous clients make informed investment decisions. You excel at 
                synthesizing complex information into clear, actionable investment 
                advice. You consider risk tolerance, market conditions, and 
                individual investment goals in your recommendations.
            """),
            verbose=True,
            allow_delegation=False,
            tools=[
                CalculatorTools.calculate,
                SearchTools.search_news
            ]
        )
