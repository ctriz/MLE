import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class BaseModel:
    def __init__(self, model_type='linear_regression'):
        self.model_type = model_type
        self.model = None
        self.features = None
        self.target = None
    def create_model(self):
        raise NotImplementedError()
    def train_model_with_data(self, features, target):
        if features is None or target is None:
            raise ValueError("Features and target must be provided")
        self.features = features
        self.target = target
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42, shuffle=False
        )
        self.model = self.create_model()
        self.model.fit(X_train, y_train)
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        return {
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae
        }
    def predict_next_day(self):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        if self.features is None or self.features.empty:
            raise ValueError("Features not available. Call train_model() first.")
        last_features = self.features.iloc[-1].copy()
        last_day = last_features['Days']
        next_day_features = last_features.copy()
        next_day_features['Days'] = last_day + 1
        prediction = self.model.predict([next_day_features])[0]
        return prediction
    def get_model_info(self):
        raise NotImplementedError()

class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__('linear_regression')
    def create_model(self):
        return LinearRegression()
    def get_model_info(self):
        return {
            'name': 'Linear Regression',
            'description': 'Simple linear regression model using basic technical indicators',
            'features': ['Days', 'Volume_Norm', 'High_Low_Ratio', 'Open_Close_Ratio', 'Price_MA5_Ratio', 'Price_MA10_Ratio'],
            'advantages': ['Fast training', 'Easy to interpret', 'Good for linear relationships'],
            'disadvantages': ['May miss complex patterns', 'Assumes linear relationships']
        }

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        super().__init__('random_forest')
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
    def create_model(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
    def get_feature_importance(self):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        if self.features is None or self.features.empty:
            raise ValueError("Features not available. Call train_model() first.")
        feature_importance = pd.DataFrame({
            'feature': self.features.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return feature_importance
    def get_model_info(self):
        return {
            'name': 'Random Forest',
            'description': 'Ensemble learning method using multiple decision trees',
            'features': ['Days', 'Volume_Norm', 'High_Low_Ratio', 'Open_Close_Ratio', 'Price_MA5_Ratio', 'Price_MA10_Ratio'],
            'advantages': ['Handles non-linear relationships', 'Feature importance', 'Robust to overfitting'],
            'disadvantages': ['Slower training', 'Less interpretable', 'More complex']
        }

def get_available_models():
    return {
        'linear_regression': LinearRegressionModel,
        'random_forest': RandomForestModel
    }

def create_model(model_type):
    models = get_available_models()
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    return models[model_type]() 