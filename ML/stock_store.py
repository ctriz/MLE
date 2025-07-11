"""
Minimal Stock Data Store for NASDAQ Stocks

This module provides a simple stock data management system for the top 10 NASDAQ stocks.
"""

import threading
from typing import Dict, List, Optional, Any, Union
import pandas as pd

class StockStore:
    # Top 10 NASDAQ stocks by market cap
    NASDAQ_TOP_10 = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO',
        'PEP', 'COST'
    ]
    # Company name mapping for top 10 NASDAQ stocks
    COMPANY_NAMES = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc.',
        'AMZN': 'Amazon.com Inc.',
        'NVDA': 'NVIDIA Corporation',
        'META': 'Meta Platforms Inc.',
        'TSLA': 'Tesla Inc.',
        'AVGO': 'Broadcom Inc.',
        'PEP': 'PepsiCo Inc.',
        'COST': 'Costco Wholesale Corporation'
    }

    def __init__(self):
        self.stocks_data: Dict[str, Dict[str, Any]] = {}
        self.stocks_info: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        for ticker in self.NASDAQ_TOP_10:
            self.stocks_data[ticker] = {
                'price_data': None,
                'technical_indicators': None
            }
            self.stocks_info[ticker] = {
                'company_name': self.COMPANY_NAMES.get(ticker, ticker),
                'sector': 'Unknown',
                'market_cap': 0,
                'current_price': 0
            }

    def _calculate_technical_indicators(self, price_data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if price_data is None or price_data.empty:
            return pd.DataFrame()
        if isinstance(price_data, pd.Series):
            df = price_data.to_frame()
        else:
            df = price_data.copy()
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        return df

    def get_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        with self.lock:
            if ticker in self.stocks_data:
                return self.stocks_data[ticker]['price_data']
        return None

    def get_stock_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            if ticker in self.stocks_info:
                return self.stocks_info[ticker]
        return None

    def get_technical_indicators(self, ticker: str) -> Optional[pd.DataFrame]:
        with self.lock:
            if ticker in self.stocks_data:
                return self.stocks_data[ticker]['technical_indicators']
        return None

    def get_stocks_list(self) -> List[Dict[str, Any]]:
        stocks_list = []
        with self.lock:
            for ticker, info in self.stocks_info.items():
                company_name = info.get('company_name', ticker)
                stocks_list.append({
                    'ticker': ticker,
                    'display_name': f"{ticker} - {company_name}",
                    'company_name': company_name,
                    'sector': info.get('sector', 'Unknown'),
                    'current_price': info.get('current_price', 0)
                })
        stocks_list.sort(key=lambda x: x['company_name'])
        return stocks_list

# Global instance
stock_store = StockStore() 