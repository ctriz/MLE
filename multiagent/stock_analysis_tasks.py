from crewai import Task
from textwrap import dedent


class StockAnalysisTasks:
    def research(self, agent, company):
        return Task(
            description=dedent(f"""
                Conduct comprehensive research on {company}. Your research should include:
                
                1. Company Overview:
                   - Business model and core products/services
                   - Market position and competitive landscape
                   - Industry trends and market dynamics
                   - Recent news and developments
                
                2. Market Analysis:
                   - Stock price performance and trends
                   - Market capitalization and trading volume
                   - Analyst ratings and price targets
                   - Sector performance comparison
                
                3. Key Metrics:
                   - Revenue growth and profitability trends
                   - Market share and competitive advantages
                   - Management team and corporate governance
                   - Risk factors and challenges
                
                Provide detailed, well-structured information that will serve as the foundation 
                for financial analysis and investment recommendations.
            """),
            agent=agent,
            expected_output=dedent("""
                A comprehensive research report covering company overview, market analysis, 
                and key metrics with detailed insights and supporting data.
            """)
        )

    def financial_analysis(self, agent):
        return Task(
            description=dedent("""
                Based on the research provided, perform detailed financial analysis including:
                
                1. Financial Ratios:
                   - P/E ratio, P/B ratio, EV/EBITDA
                   - ROE, ROA, profit margins
                   - Debt-to-equity, current ratio
                   - Dividend yield and payout ratio
                
                2. Financial Performance:
                   - Revenue and earnings growth trends
                   - Cash flow analysis
                   - Balance sheet strength
                   - Working capital management
                
                3. Valuation Analysis:
                   - DCF valuation if applicable
                   - Comparable company analysis
                   - Asset-based valuation
                   - Growth prospects assessment
                
                4. Financial Health Indicators:
                   - Liquidity and solvency ratios
                   - Efficiency metrics
                   - Quality of earnings
                   - Financial risk assessment
                
                Provide quantitative analysis with clear interpretations and implications.
            """),
            agent=agent,
            expected_output=dedent("""
                A detailed financial analysis report with ratios, performance metrics, 
                valuation analysis, and financial health indicators with clear interpretations.
            """)
        )

    def filings_analysis(self, agent):
        return Task(
            description=dedent("""
                Analyze recent SEC filings and regulatory documents to identify:
                
                1. 10-K and 10-Q Analysis:
                   - Management discussion and analysis
                   - Risk factors and uncertainties
                   - Business segment performance
                   - Regulatory compliance issues
                
                2. Material Events:
                   - Recent 8-K filings
                   - Significant corporate events
                   - Management changes
                   - Strategic initiatives
                
                3. Regulatory Compliance:
                   - Accounting policies and changes
                   - Internal control assessments
                   - Legal proceedings
                   - Environmental and social responsibility
                
                4. Forward-Looking Statements:
                   - Management guidance
                   - Strategic plans and objectives
                   - Market outlook and projections
                   - Investment plans and capital allocation
                
                Focus on material information that could impact investment decisions.
            """),
            agent=agent,
            expected_output=dedent("""
                A comprehensive analysis of SEC filings highlighting key insights, 
                risks, opportunities, and material information for investors.
            """)
        )

    def recommend(self, agent):
        return Task(
            description=dedent("""
                Based on all the research, financial analysis, and filings review, 
                provide comprehensive investment recommendations:
                
                1. Investment Thesis:
                   - Key investment drivers
                   - Bull and bear case scenarios
                   - Risk-reward assessment
                   - Investment timeline
                
                2. Recommendation:
                   - Buy, Hold, or Sell recommendation
                   - Target price and rationale
                   - Position sizing guidance
                   - Entry and exit strategies
                
                3. Risk Assessment:
                   - Key risks and mitigants
                   - Market and company-specific risks
                   - Regulatory and competitive risks
                   - Risk management strategies
                
                4. Portfolio Considerations:
                   - Diversification benefits
                   - Sector allocation impact
                   - Correlation with existing holdings
                   - Alternative investment options
                
                Provide clear, actionable advice suitable for different investor profiles.
            """),
            agent=agent,
            expected_output=dedent("""
                A comprehensive investment recommendation with clear buy/hold/sell advice, 
                target price, risk assessment, and portfolio considerations.
            """)
        )

    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"
