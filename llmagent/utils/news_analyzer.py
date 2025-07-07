from dotenv import load_dotenv
import os
import google.generativeai as genai
import requests
from datetime import datetime, timedelta
import argparse

load_dotenv()


# Initialize the Gemini client

def fetch_news_headlines(ticker, limit=5):
    """
    Fetch credible financial news using NewsAPI.
    Requires NEWS_API_KEY environment variable.
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise ValueError("NEWS_API_KEY environment variable is required. Please set it in your .env file.")
    
    # Calculate date range (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # NewsAPI endpoint
    url = "https://newsapi.org/v2/everything"
    
    # Search for company news
    params = {
        'q': f'"{ticker}" OR "{ticker.upper()}"',
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': limit,
        'apiKey': api_key,
        'domains': 'reuters.com,bloomberg.com,cnbc.com,marketwatch.com,wsj.com,ft.com,yahoo.com,seekingalpha.com'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'ok' and data['articles']:
            headlines = []
            for article in data['articles'][:limit]:
                title = article.get('title', 'No title')
                source = article.get('source', {}).get('name', 'Unknown')
                published_at = article.get('publishedAt', '')
                
                # Format: "Title - Source (Date)"
                if published_at:
                    try:
                        date_obj = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        date_str = date_obj.strftime('%Y-%m-%d')
                    except:
                        date_str = published_at[:10]
                else:
                    date_str = 'Unknown date'
                
                headline = f"{title} - {source} ({date_str})"
                headlines.append(headline)
            
            return headlines
        else:
            return [f"No recent news found for {ticker}"]
            
    except Exception as e:
        raise Exception(f"Failed to fetch news from NewsAPI: {str(e)}")

def analyze_news_with_gemini(headlines, model_name=None):
    if model_name is None:
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    if not headlines:
        return "No recent news found."
    
    # Configure the API
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Create the model
    model = genai.GenerativeModel(model_name)
    
    prompt = (
        "Analyze the following news headlines for market impact "
        "and summarize sentiment related to the stock:\n\n"
        + "\n".join(f"- {headline}" for headline in headlines)
    )
    
    response = model.generate_content(prompt)
    return response.text

def analyze_news_with_gemini_agent(ticker, model_name=None, limit=5):
    if model_name is None:
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    headlines = fetch_news_headlines(ticker, limit)
    analysis = analyze_news_with_gemini(headlines, model_name=model_name)
    return {
        "headlines": headlines,
        "news_analysis": analysis
    }

def main():
    """
    CLI interface for news analysis.
    Usage: python agent_news.py AAPL --limit 5 --headlines-only
    """
    parser = argparse.ArgumentParser(description="Analyze news sentiment for a stock ticker using Google Gemini AI.")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), help="Gemini model to use (default: from GEMINI_MODEL env var)")
    parser.add_argument("--limit", type=int, default=5, help="Number of news headlines to fetch (default: 5)")
    parser.add_argument("--headlines-only", action="store_true", help="Only fetch and display headlines without analysis")
    args = parser.parse_args()

    print(f"Fetching news for {args.ticker}...")
    headlines = fetch_news_headlines(args.ticker, args.limit)
    print(f"\nNews headlines for {args.ticker}:")
    for i, headline in enumerate(headlines, 1):
        print(f"{i}. {headline}")

    if not args.headlines_only:
        print(f"\nNews analysis (using {args.model}):")
        analysis = analyze_news_with_gemini(headlines, model_name=args.model)
        print(analysis)

if __name__ == "__main__":
    main()
