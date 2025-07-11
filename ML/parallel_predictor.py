"""
Parallel Stock Price Prediction using Dask

This module provides parallel execution of machine learning models for stock
price prediction using Dask for distributed computing and orchestration.

Author: Stock Analyzer Team
Date: 2024
"""

import dask
from dask import delayed
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings

from stock_predictor import StockPredictor
from stock_store import stock_store

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ParallelStockPredictor:
    """
    Parallel stock predictor using Dask for orchestration.
    
    This class provides parallel execution of multiple machine learning models
    for stock price prediction, utilizing Dask for distributed computing.
    
    Features:
    - Parallel model training and prediction
    - Dask orchestration for distributed computing
    - Comprehensive result formatting
    - Error handling and recovery
    - Performance metrics comparison
    """
    
    def __init__(self):
        """Initialize the parallel predictor with available models."""
        self.available_models = {
            'linear_regression': 'Linear Regression',
            'random_forest': 'Random Forest'
        }
    
    @delayed
    def _train_model_parallel(self, ticker: str, model_type: str) -> Dict[str, Any]:
        """
        Train a single model in parallel using Dask delayed execution.
        
        Args:
            ticker: Stock ticker symbol
            model_type: Type of model to train ('linear_regression' or 'random_forest')
            
        Returns:
            Dictionary containing model results and metadata
        """
        try:
            # Create predictor instance
            predictor = StockPredictor(symbol=ticker, model_type=model_type)
            
            # Fetch and validate data
            if not predictor.fetch_data():
                return {
                    'model_type': model_type,
                    'model_name': self.available_models[model_type],
                    'error': f"Failed to fetch data for {ticker}"
                }
            
            # Create features for model training
            predictor.create_features()
            
            # Train model and get metrics
            metrics = predictor.train_model()
            
            # Validate data availability
            if predictor.data is None or predictor.data.empty:
                return {
                    'model_type': model_type,
                    'model_name': self.available_models[model_type],
                    'error': f"No data available for {ticker}"
                }
            
            # Get current price and make prediction
            current_price = predictor.data['Close'].iloc[-1]
            next_day_prediction = predictor.predict_next_day()
            
            # Calculate price change metrics
            price_change = next_day_prediction - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # Gather additional information
            model_info = predictor.get_model_info()
            stock_info = predictor.get_stock_info() or {}
            feature_importance = predictor.get_feature_importance()
            
            return {
                'model_type': model_type,
                'model_name': self.available_models[model_type],
                'current_price': current_price,
                'prediction': next_day_prediction,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'metrics': metrics,
                'model_info': model_info,
                'stock_info': stock_info,
                'feature_importance': feature_importance,
                'error': None
            }
            
        except Exception as e:
            return {
                'model_type': model_type,
                'model_name': self.available_models[model_type],
                'error': f"Error with {self.available_models[model_type]}: {str(e)}"
            }
    
    def predict_stock_parallel(self, ticker: str) -> str:
        """
        Run predictions for both models in parallel using Dask.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Formatted string with parallel prediction results
        """
        if not ticker:
            return "Please provide a valid stock ticker."
        
        ticker = ticker.strip().upper()
        
        # Create delayed tasks for both models
        lr_task = self._train_model_parallel(ticker, 'linear_regression')
        rf_task = self._train_model_parallel(ticker, 'random_forest')
        
        # Compute both tasks in parallel
        print(f"Running parallel predictions for {ticker}...")
        results = dask.compute(lr_task, rf_task)
        
        # Organize results by model type
        results_dict = {}
        for result in results:
            model_type = result['model_type']
            results_dict[model_type] = result
        
        # Format and return results
        return self._format_parallel_results(ticker, results_dict)
    
    def _format_parallel_results(self, ticker: str, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Format results from parallel model execution into readable text.
        
        Args:
            ticker: Stock ticker symbol
            results: Dictionary containing results for each model
            
        Returns:
            Formatted string with comprehensive results
        """
        result_text = f"=== PARALLEL PREDICTION RESULTS FOR {ticker} ===\n"
        result_text += f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Get stock info from first successful model
        stock_info = None
        for model_key, result in results.items():
            if result.get('error') is None:
                stock_info = result['stock_info']
                break
        
        if stock_info:
            result_text += f"Company: {stock_info.get('company_name', ticker)}\n"
            result_text += f"Sector: {stock_info.get('sector', 'Unknown')}\n"
            result_text += f"Market Cap: ${stock_info.get('market_cap', 0):,}\n\n"
        
        # Format results for each model
        for model_key, result in results.items():
            model_name = result['model_name']
            result_text += f"=== {model_name.upper()} MODEL ===\n"
            
            if result.get('error') is None:
                # Price predictions
                result_text += f"Current Price: ${result['current_price']:.2f}\n"
                result_text += f"Predicted Next Day: ${result['prediction']:.2f}\n"
                result_text += f"Expected Change: ${result['price_change']:.2f} ({result['price_change_pct']:+.2f}%)\n\n"
                
                # Model performance metrics
                metrics = result['metrics']
                result_text += f"Model Performance:\n"
                result_text += f"  Training R²: {metrics['train_r2']:.4f}\n"
                result_text += f"  Testing R²: {metrics['test_r2']:.4f}\n"
                result_text += f"  Training RMSE: ${metrics['train_rmse']:.2f}\n"
                result_text += f"  Testing RMSE: ${metrics['test_rmse']:.2f}\n"
                result_text += f"  Training MAE: ${metrics['train_mae']:.2f}\n"
                result_text += f"  Testing MAE: ${metrics['test_mae']:.2f}\n\n"
                
                # Feature importance for Random Forest
                if model_key == 'random_forest' and result['feature_importance'] is not None:
                    result_text += f"Top 5 Feature Importance:\n"
                    for _, row in result['feature_importance'].head(5).iterrows():
                        result_text += f"  {row['feature']}: {row['importance']:.4f}\n"
                    result_text += "\n"
                
            else:
                error_msg = result.get('error', f'Unknown error with {model_name}')
                result_text += f"Error: {error_msg}\n\n"
        
        # Add comparison summary if both models succeeded
        successful_models = [k for k, v in results.items() if v.get('error') is None]
        if len(successful_models) == 2:
            result_text += self._format_comparison_summary(results)
        
        return result_text
    
    def _format_comparison_summary(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Format comparison summary between models.
        
        Args:
            results: Dictionary containing results for each model
            
        Returns:
            Formatted comparison summary
        """
        summary = "=== PARALLEL MODEL COMPARISON ===\n"
        
        lr_result = results['linear_regression']
        rf_result = results['random_forest']
        
        summary += f"Linear Regression Prediction: ${lr_result['prediction']:.2f}\n"
        summary += f"Random Forest Prediction: ${rf_result['prediction']:.2f}\n"
        
        prediction_diff = abs(lr_result['prediction'] - rf_result['prediction'])
        summary += f"Prediction Difference: ${prediction_diff:.2f}\n\n"
        
        # Compare accuracy metrics
        lr_test_r2 = lr_result['metrics']['test_r2']
        rf_test_r2 = rf_result['metrics']['test_r2']
        
        summary += f"Accuracy Comparison:\n"
        summary += f"  Linear Regression Test R²: {lr_test_r2:.4f}\n"
        summary += f"  Random Forest Test R²: {rf_test_r2:.4f}\n"
        
        if lr_test_r2 > rf_test_r2:
            summary += f"  Linear Regression shows better accuracy\n"
        elif rf_test_r2 > lr_test_r2:
            summary += f"  Random Forest shows better accuracy\n"
        else:
            summary += f"  Both models show similar accuracy\n"
        
        summary += f"\n=== PARALLEL EXECUTION BENEFITS ===\n"
        summary += f"✓ Both models trained simultaneously\n"
        summary += f"✓ Reduced total execution time\n"
        summary += f"✓ Better resource utilization\n"
        
        return summary
    
    def get_parallel_stats(self) -> Dict[str, Any]:
        """
        Get information about parallel execution capabilities.
        
        Returns:
            Dictionary with parallel execution statistics
        """
        return {
            'framework': 'Dask',
            'parallel_models': list(self.available_models.keys()),
            'execution_type': 'Delayed parallel computation',
            'benefits': [
                'Faster execution through parallelization',
                'Better resource utilization',
                'Scalable architecture',
                'Fault tolerance'
            ]
        }


# Global instance for convenience
parallel_predictor = ParallelStockPredictor() 