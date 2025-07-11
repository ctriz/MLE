# NASDAQ Stock Price Predictor

## Description
A simple and efficient tool for predicting next-day prices of the top 10 NASDAQ stocks using parallel machine learning models (Linear Regression and Random Forest) with Dask. Includes a minimal web interface for user interaction.

## Dependencies
- pandas
- numpy
- scikit-learn
- yfinance
- requests
- python-dotenv
- gradio
- dask

Install all dependencies with:
```
pip install -r requirements.txt
```

## File Structure
- `main.py` — Main GUI application (Gradio web interface)
- `parallel_predictor.py` — Parallel ML execution logic
- `stock_predictor.py` — Core prediction logic and feature engineering
- `models.py` — Machine learning model definitions
- `stock_store.py` — Data management and company info
- `requirements.txt` — Python dependencies 