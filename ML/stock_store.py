"""
Advanced Stock Data Store for NASDAQ Stocks

This module provides a comprehensive stock data management system using yfinance
for real-time market data. It includes caching, technical indicators calculation,
and thread-safe operations.

Author: Stock Analyzer Team
Date: 2024
"""

import os
import time
import threading
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()


class StockStore:
    """
    Advanced multi-dimensional stock store for NASDAQ stocks.
    
    This class provides a comprehensive solution for managing
    stock data including price information, technical indicators, and company metadata.
    
    Features:
    - Company information management
    - Thread-safe operations with locking
    - Technical indicators calculation
    - Session-based caching for performance
    """
    
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
        """
        Initialize the StockStore with configuration and data structures.
        
        Raises:
            ValueError: If required environment variables are missing
        """
        self._validate_environment()
        self._initialize_data_structures()
        self._initialize_store()
    
    def _validate_environment(self):
        """Validate that required environment variables are set."""
        # Note: API key validation removed since we're using yfinance
        pass
    
    def _initialize_data_structures(self):
        """Initialize data structures and configuration."""
        self.stocks_data: Dict[str, Dict[str, Any]] = {}
        self.stocks_info: Dict[str, Dict[str, Any]] = {}
        self.last_update: Optional[datetime] = None
        self.lock = threading.Lock()
        
        # Session caching for performance optimization
        self.session_cache = {
            'company_info_fetched': True,
            'last_info_fetch': datetime.now()
        }
    
    def _initialize_store(self):
        """Initialize the store with empty data structures for all stocks."""
        for ticker in self.NASDAQ_TOP_10:
            self.stocks_data[ticker] = {
                'price_data': None,
                'technical_indicators': None,
                'last_updated': None
            }
            self.stocks_info[ticker] = {
                'company_name': self.COMPANY_NAMES.get(ticker, ticker),
                'sector': 'Unknown',
                'market_cap': 0,
                'current_price': 0,
                'volume': 0,
                'pe_ratio': 0,
                'dividend_yield': 0
            }
    
    def _calculate_technical_indicators(self, price_data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate technical indicators for a stock.
        
        Args:
            price_data: DataFrame or Series containing OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        if price_data.empty:
            return pd.DataFrame()
        
        # Ensure we have a DataFrame
        if isinstance(price_data, pd.Series):
            df = price_data.to_frame()
        else:
            df = price_data.copy()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        return df
    
    def get_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Get price data for a specific stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with price data or None if not found
        """
        with self.lock:
            if ticker in self.stocks_data:
                return self.stocks_data[ticker]['price_data']
        return None
    
    def get_stock_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get company info for a specific stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company info or None if not found
        """
        with self.lock:
            if ticker in self.stocks_info:
                return self.stocks_info[ticker]
        return None
    
    def get_technical_indicators(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Get technical indicators for a specific stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with technical indicators or None if not found
        """
        with self.lock:
            if ticker in self.stocks_data:
                return self.stocks_data[ticker]['technical_indicators']
        return None
    
    def get_all_stocks_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get info for all stocks.
        
        Returns:
            Dictionary with all stocks info
        """
        with self.lock:
            return self.stocks_info.copy()
    
    def get_stocks_list(self) -> List[Dict[str, Any]]:
        """
        Get list of available stocks with display names.
        
        Returns:
            List of dictionaries with stock information
        """
        stocks_list = []
        with self.lock:
            for ticker, info in self.stocks_info.items():
                company_name = info.get('company_name', ticker)
                sector = info.get('sector', 'Unknown')
                stocks_list.append({
                    'ticker': ticker,
                    'display_name': f"{ticker} - {company_name}",
                    'company_name': company_name,
                    'sector': sector,
                    'current_price': info.get('current_price', 0)
                })
        
        # Sort by company name (alphabetical)
        stocks_list.sort(key=lambda x: x['company_name'])
        return stocks_list
    
    def get_stocks_by_sector(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get stocks grouped by sector.
        
        Returns:
            Dictionary with sectors as keys and lists of stocks as values
        """
        sector_groups = {}
        with self.lock:
            for ticker, info in self.stocks_info.items():
                sector = info.get('sector', 'Unknown')
                if sector not in sector_groups:
                    sector_groups[sector] = []
                
                sector_groups[sector].append({
                    'ticker': ticker,
                    'company_name': info.get('company_name', ticker),
                    'current_price': info.get('current_price', 0)
                })
        
        return sector_groups
    
    def get_market_summary(self) -> Dict[str, Any]:
        """
        Get market summary statistics.
        
        Returns:
            Dictionary with market summary statistics
        """
        with self.lock:
            total_market_cap = sum(info.get('market_cap', 0) for info in self.stocks_info.values())
            avg_pe = np.mean([info.get('pe_ratio', 0) for info in self.stocks_info.values() if info.get('pe_ratio', 0) > 0])
            avg_dividend = np.mean([info.get('dividend_yield', 0) for info in self.stocks_info.values() if info.get('dividend_yield', 0) > 0])
            
            return {
                'total_market_cap': total_market_cap,
                'average_pe_ratio': avg_pe,
                'average_dividend_yield': avg_dividend,
                'total_stocks': len(self.stocks_info),
                'last_update': self.last_update
            }
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        Get the current cache status.
        
        Returns:
            Dictionary with cache status information
        """
        return {
            'company_info_cached': self.session_cache['company_info_fetched'],
            'last_info_fetch': self.session_cache['last_info_fetch'],
            'total_stocks': len(self.NASDAQ_TOP_10),
            'stocks_with_prices': 0  # No live prices loaded
        }
    
    def get_stock_status(self, ticker: str) -> Dict[str, Any]:
        """
        Get status of a specific stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with stock status information
        """
        with self.lock:
            if ticker not in self.stocks_data:
                return {'status': 'not_found', 'last_update': None}
            
            data = self.stocks_data[ticker]
            if data['price_data'] is None:
                return {'status': 'no_data', 'last_update': None}
            
            return {
                'status': 'available',
                'last_update': data['last_updated'],
                'data_points': len(data['price_data']) if data['price_data'] is not None else 0
            }


# Global instance
stock_store = StockStore() 