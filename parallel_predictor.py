import dask
from dask import delayed
from datetime import datetime
from stock_predictor import StockPredictor

class ParallelStockPredictor:
    def __init__(self):
        self.available_models = {
            'linear_regression': 'Linear Regression',
            'random_forest': 'Random Forest'
        }

    @delayed
    def _train_model_parallel(self, ticker, model_type):
        try:
            predictor = StockPredictor(symbol=ticker, model_type=model_type)
            if not predictor.fetch_data():
                return {
                    'model_type': model_type,
                    'model_name': self.available_models[model_type],
                    'error': f"Failed to fetch data for {ticker}"
                }
            predictor.create_features()
            metrics = predictor.train_model()
            if predictor.data is None or predictor.data.empty:
                return {
                    'model_type': model_type,
                    'model_name': self.available_models[model_type],
                    'error': f"No data available for {ticker}"
                }
            current_price = predictor.data['Close'].iloc[-1]
            next_day_prediction = predictor.predict_next_day()
            price_change = next_day_prediction - current_price
            price_change_pct = (price_change / current_price) * 100
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

    def predict_stock_parallel(self, ticker):
        if not ticker:
            return "Please provide a valid stock ticker."
        ticker = ticker.strip().upper()
        lr_task = self._train_model_parallel(ticker, 'linear_regression')
        rf_task = self._train_model_parallel(ticker, 'random_forest')
        results = dask.compute(lr_task, rf_task)
        results_dict = {result['model_type']: result for result in results}
        return self._format_parallel_results(ticker, results_dict)

    def _format_parallel_results(self, ticker, results):
        result_text = f"=== PARALLEL PREDICTION RESULTS FOR {ticker} ===\n"
        result_text += f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        stock_info = None
        for result in results.values():
            if result.get('error') is None:
                stock_info = result['stock_info']
                break
        if stock_info:
            result_text += f"Company: {stock_info.get('company_name', ticker)}\n"
            result_text += f"Sector: {stock_info.get('sector', 'Unknown')}\n"
            result_text += f"Market Cap: ${stock_info.get('market_cap', 0):,}\n\n"
        for model_key, result in results.items():
            model_name = result['model_name']
            result_text += f"=== {model_name.upper()} MODEL ===\n"
            if result.get('error') is None:
                result_text += f"Current Price: ${result['current_price']:.2f}\n"
                result_text += f"Predicted Next Day: ${result['prediction']:.2f}\n"
                result_text += f"Expected Change: ${result['price_change']:.2f} ({result['price_change_pct']:+.2f}%)\n\n"
                metrics = result['metrics']
                result_text += f"Model Performance:\n"
                result_text += f"  Training R²: {metrics['train_r2']:.4f}\n"
                result_text += f"  Testing R²: {metrics['test_r2']:.4f}\n"
                result_text += f"  Training RMSE: ${metrics['train_rmse']:.2f}\n"
                result_text += f"  Testing RMSE: ${metrics['test_rmse']:.2f}\n"
                result_text += f"  Training MAE: ${metrics['train_mae']:.2f}\n"
                result_text += f"  Testing MAE: ${metrics['test_mae']:.2f}\n\n"
                if model_key == 'random_forest' and result['feature_importance'] is not None:
                    result_text += f"Top 5 Feature Importance:\n"
                    for _, row in result['feature_importance'].head(5).iterrows():
                        result_text += f"  {row['feature']}: {row['importance']:.4f}\n"
                    result_text += "\n"
            else:
                error_msg = result.get('error', f'Unknown error with {model_name}')
                result_text += f"Error: {error_msg}\n\n"
        successful_models = [k for k, v in results.items() if v.get('error') is None]
        if len(successful_models) == 2:
            result_text += self._format_comparison_summary(results)
        return result_text

    def _format_comparison_summary(self, results):
        summary = "=== PARALLEL MODEL COMPARISON ===\n"
        lr_result = results['linear_regression']
        rf_result = results['random_forest']
        summary += f"Linear Regression Prediction: ${lr_result['prediction']:.2f}\n"
        summary += f"Random Forest Prediction: ${rf_result['prediction']:.2f}\n"
        prediction_diff = abs(lr_result['prediction'] - rf_result['prediction'])
        summary += f"Prediction Difference: ${prediction_diff:.2f}\n\n"
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

parallel_predictor = ParallelStockPredictor() 