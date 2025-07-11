"""
Stock Price Prediction GUI using Gradio

This module provides a professional web interface for stock price prediction
using parallel machine learning models with Dask orchestration.

Author: Stock Analyzer Team
Date: 2024
"""

import time
from typing import List, Dict, Any, Optional

import gradio as gr
import pandas as pd

from stock_store import stock_store
from parallel_predictor import ParallelStockPredictor


class StockPredictionGradioGUI:
    """
    Professional Gradio GUI for stock price prediction.
    
    This class provides a comprehensive web interface for stock analysis
    with parallel model execution and comparison tables.
    
    Features:
    - Stock data display with company names
    - Parallel model execution with Dask
    - Professional comparison tables
    - Interactive stock selection
    """
    
    def __init__(self):
        """Initialize the GUI with stock store and predictor."""
        self.stocks_data: List[Dict[str, Any]] = []
        self.parallel_predictor = ParallelStockPredictor()
    
    def get_stocks_table(self) -> pd.DataFrame:
        """
        Get stocks data for display in table with company names.
        
        Returns:
            DataFrame with stock information for display
        """
        # Get stock list from store
        self.stocks_data = stock_store.get_stocks_list()
        
        if not self.stocks_data:
            return pd.DataFrame({
                'Company Name': [],
                'Stock Code': [],
                'Status': []
            })
        
        table_data = []
        for stock in self.stocks_data:
            ticker = stock['ticker']
            company_name = stock.get('company_name', ticker)
            
            table_data.append({
                'Company Name': company_name,
                'Stock Code': ticker,
                'Status': 'Available'
            })
        
        return pd.DataFrame(table_data)
    
    def on_stock_selected(self, evt: gr.SelectData) -> str:
        """
        Handle stock selection and auto-populate the input.
        
        Args:
            evt: Gradio selection event
            
        Returns:
            Selected stock ticker or empty string
        """
        if evt.index[0] is not None:
            selected_row = evt.index[0]
            if selected_row < len(self.stocks_data):
                selected_ticker = self.stocks_data[selected_row]['ticker']
                return selected_ticker
        return ""
    
    def predict_stock(self, selected_stock: str) -> pd.DataFrame:
        """
        Run predictions for both models in parallel using Dask and return formatted table.
        
        Args:
            selected_stock: Stock ticker symbol
            
        Returns:
            DataFrame with comparison table of model results
        """
        if not selected_stock:
            return pd.DataFrame({
                'Metric': ['Status'],
                'Linear Regression': ['Please select a stock'],
                'Random Forest': ['Please select a stock']
            })

        ticker = selected_stock.strip().upper()

        # Use parallel predictor with Dask
        results_text = self.parallel_predictor.predict_stock_parallel(ticker)
        
        # Format the results into a table
        return self.format_results_table(results_text)
    
    def format_results_table(self, results_text: str) -> pd.DataFrame:
        """
        Convert results text to a structured table format.
        
        Args:
            results_text: Raw results text from parallel predictor
            
        Returns:
            DataFrame with formatted comparison table
        """
        if "Please select" in results_text or "Error" in results_text:
            return pd.DataFrame({
                'Metric': ['Status'],
                'Linear Regression': ['No Data'],
                'Random Forest': ['No Data']
            })

        # Parse the results text to extract key metrics
        lines = results_text.split('\n')
        
        # Extract key metrics
        metrics = {
            'Current Price': ['N/A', 'N/A'],
            'Predicted Next Day': ['N/A', 'N/A'],
            'Expected Change': ['N/A', 'N/A'],
            'Training R²': ['N/A', 'N/A'],
            'Testing R²': ['N/A', 'N/A'],
            'Training RMSE': ['N/A', 'N/A'],
            'Testing RMSE': ['N/A', 'N/A'],
            'Training MAE': ['N/A', 'N/A'],
            'Testing MAE': ['N/A', 'N/A']
        }

        current_model = None
        for line in lines:
            line = line.strip()

            # Detect model sections
            if 'LINEAR REGRESSION MODEL' in line:
                current_model = 'lr'
            elif 'RANDOM FOREST MODEL' in line:
                current_model = 'rf'

            # Extract metrics based on current model
            if current_model == 'lr':
                if 'Current Price: $' in line:
                    metrics['Current Price'][0] = line.split('$')[1]
                elif 'Predicted Next Day: $' in line:
                    metrics['Predicted Next Day'][0] = line.split('$')[1]
                elif 'Expected Change: $' in line:
                    change_part = line.split('Expected Change: ')[1]
                    metrics['Expected Change'][0] = change_part.split(' (')[0]
                elif 'Training R²:' in line:
                    metrics['Training R²'][0] = line.split(': ')[1]
                elif 'Testing R²:' in line:
                    metrics['Testing R²'][0] = line.split(': ')[1]
                elif 'Training RMSE: $' in line:
                    metrics['Training RMSE'][0] = line.split('$')[1]
                elif 'Testing RMSE: $' in line:
                    metrics['Testing RMSE'][0] = line.split('$')[1]
                elif 'Training MAE: $' in line:
                    metrics['Training MAE'][0] = line.split('$')[1]
                elif 'Testing MAE: $' in line:
                    metrics['Testing MAE'][0] = line.split('$')[1]

            elif current_model == 'rf':
                if 'Current Price: $' in line:
                    metrics['Current Price'][1] = line.split('$')[1]
                elif 'Predicted Next Day: $' in line:
                    metrics['Predicted Next Day'][1] = line.split('$')[1]
                elif 'Expected Change: $' in line:
                    change_part = line.split('Expected Change: ')[1]
                    metrics['Expected Change'][1] = change_part.split(' (')[0]
                elif 'Training R²:' in line:
                    metrics['Training R²'][1] = line.split(': ')[1]
                elif 'Testing R²:' in line:
                    metrics['Testing R²'][1] = line.split(': ')[1]
                elif 'Training RMSE: $' in line:
                    metrics['Training RMSE'][1] = line.split('$')[1]
                elif 'Testing RMSE: $' in line:
                    metrics['Testing RMSE'][1] = line.split('$')[1]
                elif 'Training MAE: $' in line:
                    metrics['Training MAE'][1] = line.split('$')[1]
                elif 'Testing MAE: $' in line:
                    metrics['Testing MAE'][1] = line.split('$')[1]

        # Create comparison table
        table_data = []
        for metric, values in metrics.items():
            table_data.append({
                'Metric': metric,
                'Linear Regression': values[0],
                'Random Forest': values[1]
            })

        return pd.DataFrame(table_data)
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the professional Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="Top 10 NASDAQ Stock Price Predictor (Parallel)"
        ) as interface:
            
            gr.Markdown("# Top 10 NASDAQ Stock Price Predictor")
            gr.Markdown(
                "Predict next day stock prices using parallel Linear Regression and "
                "Random Forest models with Dask orchestration"
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Available Stocks")
                    stocks_table = gr.Dataframe(
                        value=self.get_stocks_table(),
                        headers=['Company Name', 'Stock Code', 'Status'],
                        datatype=['str', 'str', 'str'],
                        interactive=True,
                        wrap=True
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Stock Selection")
                    selected_stock = gr.Textbox(
                        label="Selected Stock Code",
                        placeholder="Click on a stock row to auto-populate",
                        info="Stock code will be auto-populated when you click on a row"
                    )
                    predict_btn = gr.Button(
                        "Predict Next Day Price (Parallel Models)",
                        variant="primary",
                        size="lg"
                    )
            
            gr.Markdown("### Prediction Results")
            
            # Comparison Table only
            results_table = gr.Dataframe(
                headers=['Metric', 'Linear Regression', 'Random Forest'],
                datatype=['str', 'str', 'str'],
                interactive=False,
                wrap=True
            )
            
            with gr.Row():
                with gr.Accordion("📖 Instructions", open=False):
                    gr.Markdown("""
                    ## How to use:
                    1. **View the table** of top 10 NASDAQ stocks
                    2. **Click on any stock row** to auto-populate the stock code
                    3. **Click "Predict Next Day Price"** to run both models simultaneously using Dask
                    4. **Compare results** in the comparison table showing side-by-side metrics
                    
                    ## Parallel Processing Benefits:
                    - **Faster Execution**: Both models train simultaneously
                    - **Better Resource Utilization**: Uses all available CPU cores
                    - **Scalable**: Easy to add more models
                    - **Orchestrated**: Dask manages the parallel workflow
                    
                    ## Available Stocks:
                    - **AAPL** - Apple Inc.
                    - **MSFT** - Microsoft Corporation
                    - **GOOGL** - Alphabet Inc.
                    - **AMZN** - Amazon.com Inc.
                    - **NVDA** - NVIDIA Corporation
                    - **META** - Meta Platforms Inc.
                    - **TSLA** - Tesla Inc.
                    - **AVGO** - Broadcom Inc.
                    - **PEP** - PepsiCo Inc.
                    - **COST** - Costco Wholesale Corporation
                    """)
            
            # Event handlers
            stocks_table.select(
                fn=self.on_stock_selected,
                outputs=[selected_stock]
            )
            
            predict_btn.click(
                fn=self.predict_stock,
                inputs=[selected_stock],
                outputs=[results_table]
            )
        
        return interface


def main():
    """Main function to launch the professional Gradio interface."""
    gui = StockPredictionGradioGUI()
    interface = gui.create_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=True
    )


if __name__ == "__main__":
    main() 