# NASDAQ Stock Price Predictor GUI

A professional GUI for stock price prediction using parallel ML models.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements_stock_prediction.txt
   ```

2. **Run the GUI**:
   ```bash
   python stock-prediction-gui.py
   ```

## Features

- **Stock Selection**: Choose from top 10 NASDAQ stocks
- **Parallel Models**: Linear Regression and Random Forest with Dask
- **Professional GUI**: Gradio web interface
- **Company Names**: Proper company names instead of ticker codes
- **Comparison Table**: Side-by-side model performance metrics

## Usage

1. **View Stock Table**: See available top 10 NASDAQ stocks
2. **Select Stock**: Click on any stock row to auto-populate
3. **Predict**: Click "Predict Next Day Price" button
4. **Compare**: View results in the comparison table

## Supported Stocks

| Ticker | Company |
|--------|---------|
| AAPL   | Apple Inc. |
| MSFT   | Microsoft Corporation |
| GOOGL  | Alphabet Inc. |
| AMZN   | Amazon.com Inc. |
| NVDA   | NVIDIA Corporation |
| META   | Meta Platforms Inc. |
| TSLA   | Tesla Inc. |
| AVGO   | Broadcom Inc. |
| PEP    | PepsiCo Inc. |
| COST   | Costco Wholesale Corporation |

## Example Output

```
Current Price: $150.25
Predicted Next Day: $151.80
Expected Change: $1.55 (+1.03%)

Model Performance:
Training R²: 0.8542
Testing R²: 0.8234
Training RMSE: $2.45
Testing RMSE: $3.12
```

## Technical Details

- **Data Source**: Stock predictor fetches data as needed
- **Parallel Processing**: Dask orchestration for model training
- **Thread-safe**: Background updates with locking
- **Error Handling**: Graceful degradation on failures

## Files

- `stock-prediction-gui.py`: Main GUI application
- `stock_store.py`: Data management and company info
- `parallel_predictor.py`: Parallel ML execution
- `stock_predictor.py`: Core prediction logic
- `requirements_stock_prediction.txt`: Dependencies 