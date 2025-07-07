import yfinance as yf
import pandas as pd


def compute_macd(data, fast=12, slow=26, signal=9):
    """
    Compute MACD, Signal Line, and Histogram
    """
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def compute_rsi(data, period=14):
    """
    Compute Relative Strength Index (RSI)
    """
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def compute_sma(data, period=20):
    return data['Close'].rolling(window=period).mean()

def compute_bollinger_bands(data, period=20, num_std=2):
    sma = compute_sma(data, period)
    std = data['Close'].rolling(window=period).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    return sma, upper_band, lower_band

def get_technical_analysis(ticker):
    """
    Perform technical analysis on a ticker using MACD, RSI, SMA, and Bollinger Bands

    Returns:
        dict with analysis results and trend signal.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo", interval="1d")

        if df.empty or len(df) < 30:
            return {"error": "Not enough data to compute indicators."}

        macd, signal_line, hist = compute_macd(df)
        rsi = compute_rsi(df)
        sma = compute_sma(df)
        bb_sma, bb_upper, bb_lower = compute_bollinger_bands(df)

        latest_macd = macd.iloc[-1] if isinstance(macd, pd.Series) else macd
        latest_signal = signal_line.iloc[-1] if isinstance(signal_line, pd.Series) else signal_line
        latest_rsi = rsi.iloc[-1] if isinstance(rsi, pd.Series) else rsi
        latest_sma = sma.iloc[-1] if isinstance(sma, pd.Series) else sma
        latest_bb_upper = bb_upper.iloc[-1] if isinstance(bb_upper, pd.Series) else bb_upper
        latest_bb_lower = bb_lower.iloc[-1] if isinstance(bb_lower, pd.Series) else bb_lower
        # Ensure values are float for rounding
        latest_macd = float(latest_macd)
        latest_signal = float(latest_signal)
        latest_rsi = float(latest_rsi)
        latest_sma = float(latest_sma)
        latest_bb_upper = float(latest_bb_upper)
        latest_bb_lower = float(latest_bb_lower)

        # Trend signal logic
        trend = "neutral"
        if latest_macd > latest_signal and latest_rsi < 70:
            trend = "bullish"
        elif latest_macd < latest_signal and latest_rsi > 30:
            trend = "bearish"

        return {
            "macd": round(latest_macd, 2),
            "signal_line": round(latest_signal, 2),
            "rsi": round(latest_rsi, 2),
            "sma_20": round(latest_sma, 2),
            "bb_upper": round(latest_bb_upper, 2),
            "bb_lower": round(latest_bb_lower, 2),
            "trend": trend
        }

    except Exception as e:
        return {"error": f"Technical analysis failed: {e}"}
