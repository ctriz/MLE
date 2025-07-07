from textblob import TextBlob


def analyze_sentiment(text: str) -> str:
    """
    Analyzes sentiment of the input text using TextBlob.

    Returns:
        "positive", "neutral", or "negative"
    """
    if not text or not text.strip():
        return "neutral"
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"
