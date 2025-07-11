import pandas as pd
from models import create_model
from stock_store import stock_store

class StockPredictor:
    def __init__(self, symbol='GOOGL', model_type='linear_regression'):
        self.symbol = symbol
        self.model_type = model_type
        self.data = None
        self.model_instance = None
        self.features = None
        self.target = None

    def fetch_data(self):
        try:
            self.data = stock_store.get_stock_data(self.symbol)
            # Ensure self.data is a DataFrame or None
            if self.data is not None and not isinstance(self.data, pd.DataFrame):
                self.data = pd.DataFrame(self.data)
            if self.data is None or (isinstance(self.data, pd.DataFrame) and self.data.empty):
                self._fetch_individual_stock_data()
                if self.data is not None and not isinstance(self.data, pd.DataFrame):
                    self.data = pd.DataFrame(self.data)
                if self.data is None or (isinstance(self.data, pd.DataFrame) and self.data.empty):
                    return False
            return True
        except Exception:
            return False

    def _fetch_individual_stock_data(self):
        try:
            import requests
            import os
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            used_source = None
            if api_key:
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': 'TIME_SERIES_DAILY',
                    'symbol': self.symbol,
                    'apikey': api_key
                }
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                if data and 'Time Series (Daily)' in data:
                    time_series = data['Time Series (Daily)']
                    df_data = []
                    for date, values in time_series.items():
                        df_data.append({
                            'Date': pd.to_datetime(date),
                            'Open': float(values['1. open']),
                            'High': float(values['2. high']),
                            'Low': float(values['3. low']),
                            'Close': float(values['4. close']),
                            'Volume': int(values['5. volume'])
                        })
                    self.data = pd.DataFrame(df_data)
                    self.data.set_index('Date', inplace=True)
                    used_source = 'AlphaVantage'
            if self.data is None or (isinstance(self.data, pd.DataFrame) and self.data.empty):
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(self.symbol)
                    df = ticker.history(period='1y')
                    if not df.empty:
                        self.data = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                        used_source = 'yfinance'
                except Exception:
                    pass
        except Exception:
            pass

    def create_features(self):
        if self.data is None or not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data not loaded. Call fetch_data() first.")
        df = self.data.copy()
        df['Days'] = range(len(df))
        df['Price_Change'] = df['Close'].diff()
        df['Volume_Norm'] = df['Volume'] / df['Volume'].max()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['Price_MA5_Ratio'] = df['Close'] / df['MA_5']
        df['Price_MA10_Ratio'] = df['Close'] / df['MA_10']
        df = df.dropna()
        feature_columns = [
            'Days', 'Volume_Norm', 'High_Low_Ratio', 'Open_Close_Ratio',
            'Price_MA5_Ratio', 'Price_MA10_Ratio'
        ]
        self.features = df[feature_columns]
        self.target = df['Close']
        return df

    def train_model(self):
        if self.features is None or self.target is None or self.features.empty:
            return "Error: No data available for training. Please try another stock or refresh."
        self.model_instance = create_model(self.model_type)
        metrics = self.model_instance.train_model_with_data(self.features, self.target)
        return metrics

    def predict_next_day(self):
        if self.model_instance is None:
            raise ValueError("Model not trained. Call train_model() first.")
        return self.model_instance.predict_next_day()

    def get_model_info(self):
        if self.model_instance is None:
            temp_model = create_model(self.model_type)
            return temp_model.get_model_info()
        return self.model_instance.get_model_info()

    def get_feature_importance(self):
        if self.model_instance is None:
            raise ValueError("Model not trained. Call train_model() first.")
        if hasattr(self.model_instance, 'get_feature_importance'):
            return self.model_instance.get_feature_importance()
        else:
            return None

    def get_stock_info(self):
        return stock_store.get_stock_info(self.symbol)

    def get_technical_indicators(self):
        return stock_store.get_technical_indicators(self.symbol) 