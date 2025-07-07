import yfinance as yf
import pandas_ta as ta
import pandas as pd

def get_market_data(ticker="AAPL"):
    """
    Fetch real-time stock data and compute technical indicators using yfinance and pandas_ta.
    Args:
        ticker (str): Stock ticker symbol.
    Returns:
        dict: Market data with price, volume, and technical indicators.
    """
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")  # Get 1 month of data for indicators
        
        if hist.empty:
            return {"error": f"No data found for ticker {ticker}"}
        
        # Get current price and volume
        current_price = hist['Close'].iloc[-1]
        current_volume = hist['Volume'].iloc[-1]
        
        # Compute technical indicators
        # RSI
        rsi = ta.rsi(hist['Close'], length=14).iloc[-1]
        
        # MACD
        macd = ta.macd(hist['Close'])
        macd_line = macd['MACD_12_26_9'].iloc[-1]
        signal_line = macd['MACDs_12_26_9'].iloc[-1]
        
        # Bollinger Bands
        bb = ta.bbands(hist['Close'], length=20)
        bb_upper = bb['BBU_20_2.0'].iloc[-1]
        bb_lower = bb['BBL_20_2.0'].iloc[-1]
        
        # Simple Moving Averages
        sma_20 = ta.sma(hist['Close'], length=20).iloc[-1]
        sma_50 = ta.sma(hist['Close'], length=50).iloc[-1]
        
        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "volume": current_volume,
            "technicals": {
                "rsi": round(rsi, 2) if not pd.isna(rsi) else None,
                "macd": round(macd_line, 4) if not pd.isna(macd_line) else None,
                "signal_line": round(signal_line, 4) if not pd.isna(signal_line) else None,
                "bb_upper": round(bb_upper, 2) if not pd.isna(bb_upper) else None,
                "bb_lower": round(bb_lower, 2) if not pd.isna(bb_lower) else None,
                "sma_20": round(sma_20, 2) if not pd.isna(sma_20) else None,
                "sma_50": round(sma_50, 2) if not pd.isna(sma_50) else None
            }
        }
    except Exception as e:
        return {"error": f"Error fetching data for {ticker}: {str(e)}"} 