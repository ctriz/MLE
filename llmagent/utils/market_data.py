import yfinance as yf


def get_stock_data(ticker: str):
    """
    Fetches current stock data for the given ticker.

    Returns a dictionary with price, change, percent_change, and more.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty:
            return None

        last_quote = data.iloc[-1]
        previous_close = stock.info.get("previousClose", None)
        price = last_quote["Close"]
        change = price - previous_close if previous_close else 0
        percent_change = (change / previous_close * 100) if previous_close else 0

        return {
            "ticker": ticker.upper(),
            "price": round(price, 2),
            "change": round(change, 2),
            "percent_change": round(percent_change, 2),
            "info": stock.info
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
