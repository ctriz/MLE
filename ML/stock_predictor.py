import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from models import create_model, get_available_models
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
        """Fetch stock data from the stock store"""
        try:
            # Get data from stock store
            self.data = stock_store.get_stock_data(self.symbol)
            print(f"Data loaded for {self.symbol}: {self.data.shape if self.data is not None else None}")
            
            if self.data is None or self.data.empty:
                # If not in store, try to fetch directly from AlphaVantage
                print(f"Stock {self.symbol} not in cache, fetching individually...")
                self._fetch_individual_stock_data()
                print(f"Data loaded for {self.symbol} after individual fetch: {self.data.shape if self.data is not None else None}")
                if self.data is None or self.data.empty:
                    print(f"No data available for {self.symbol}")
                    return False
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def _fetch_individual_stock_data(self):
        """Fetch data for a single stock that's not in the prefetched data. Try AlphaVantage first, then yfinance as fallback."""
        try:
            import requests
            import os
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            used_source = None
            # Try AlphaVantage first
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
                if 'Error Message' in data:
                    print(f"AlphaVantage API error: {data['Error Message']}")
                elif 'Note' in data:
                    print(f"AlphaVantage API rate limit: {data['Note']}")
                elif data and 'Time Series (Daily)' in data:
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
            # If AlphaVantage failed, try yfinance
            if self.data is None or self.data.empty:
                print(f"Falling back to yfinance for {self.symbol}")
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(self.symbol)
                    df = ticker.history(period='1y')
                    if not df.empty:
                        df = df.rename(columns={
                            'Open': 'Open',
                            'High': 'High',
                            'Low': 'Low',
                            'Close': 'Close',
                            'Volume': 'Volume'
                        })
                        self.data = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                        used_source = 'yfinance'
                        print(f"Loaded {len(self.data)} rows from yfinance for {self.symbol}")
                    else:
                        print(f"yfinance returned empty data for {self.symbol}")
                except Exception as e:
                    print(f"Error fetching data from yfinance for {self.symbol}: {e}")
            if used_source:
                print(f"Data for {self.symbol} loaded from {used_source}")
            else:
                print(f"No data found for {self.symbol} from AlphaVantage or yfinance.")
        except Exception as e:
            print(f"Error fetching individual stock data: {e}")
    
    def create_features(self):
        """Create simple features for prediction models"""
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        df = self.data.copy()
        # Create basic features
        df['Days'] = range(len(df))
        df['Price_Change'] = df['Close'].diff()
        df['Volume_Norm'] = df['Volume'] / df['Volume'].max()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        # Simple moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        # Price relative to moving averages
        df['Price_MA5_Ratio'] = df['Close'] / df['MA_5']
        df['Price_MA10_Ratio'] = df['Close'] / df['MA_10']
        # Drop rows with NaN values
        df = df.dropna()
        print(f"Features after dropna for {self.symbol}: {df.shape}")
        # Select features for the model
        feature_columns = [
            'Days', 'Volume_Norm', 'High_Low_Ratio', 'Open_Close_Ratio',
            'Price_MA5_Ratio', 'Price_MA10_Ratio'
        ]
        self.features = df[feature_columns]
        self.target = df['Close']
        print(f"Final features shape: {self.features.shape}, target shape: {self.target.shape}")
        return df

    def train_model(self):
        """Train the specified model"""
        if self.features is None or self.target is None or self.features.empty:
            print(f"Error: No data available for training for {self.symbol}. Features shape: {None if self.features is None else self.features.shape}")
            return "Error: No data available for training. Please try another stock or refresh."
        # Create model instance
        self.model_instance = create_model(self.model_type)
        # Train the model
        metrics = self.model_instance.train_model_with_data(self.features, self.target)
        return metrics

    def predict_next_day(self):
        """Predict next day's stock price"""
        if self.model_instance is None:
            raise ValueError("Model not trained. Call train_model() first.")
        return self.model_instance.predict_next_day()

    def get_model_info(self):
        """Get information about the current model"""
        if self.model_instance is None:
            temp_model = create_model(self.model_type)
            return temp_model.get_model_info()
        return self.model_instance.get_model_info()

    def get_feature_importance(self):
        """Get feature importance if available"""
        if self.model_instance is None:
            raise ValueError("Model not trained. Call train_model() first.")
        if hasattr(self.model_instance, 'get_feature_importance'):
            return self.model_instance.get_feature_importance()
        else:
            return None

    def get_available_models(self):
        """Get list of available model types"""
        return list(get_available_models().keys())

    def get_stock_info(self):
        """Get comprehensive stock information"""
        return stock_store.get_stock_info(self.symbol)

    def get_technical_indicators(self):
        """Get technical indicators for the stock"""
        return stock_store.get_technical_indicators(self.symbol) 