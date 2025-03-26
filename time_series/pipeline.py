# forecasting_pipeline.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import timedelta
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, mean_absolute_percentage_error
import mlflow
import dagshub
dagshub.init(repo_owner='Darshan922000', repo_name='Capstone-2', mlflow=True)

def load_and_preprocess_data(filepath):
    """Load and preprocess the sales data"""
    df = pd.read_csv(filepath)
    
    # Calculate total sales
    df['total_sales'] = df['transaction_qty'] * df['unit_price']
    
    # Convert to datetime
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%d-%m-%Y')
    
    # Aggregate sales by day
    daily_sales = df.groupby('transaction_date')['total_sales'].sum().reset_index()
    daily_sales = daily_sales.set_index('transaction_date')
    
    return daily_sales

def create_features(df):
    """Create time-based features"""
    df = df.copy()
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week
    return df

def train_forecast_model(data, forecast_horizon=7):
    """Train XGBoost model and make forecasts"""
    # Create features
    data = create_features(data)
    
    # Split into features and target
    X = data.drop(columns=['total_sales'])
    y = data['total_sales']
    
    # Split into train and test
    train = data.iloc[:-forecast_horizon]
    test = data.iloc[-forecast_horizon:]
    
    # Define features and target
    FEATURES = ['day', 'month', 'quarter', 'day_of_week', 'day_of_year', 'week_of_year']
    TARGET = 'total_sales'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    
    # Initialize and train the model
    model = XGBRegressor(n_estimators=8000, 
                        early_stopping_rounds=50,
                        learning_rate=0.03)

    model.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    with mlflow.start_run():
        # train_mae = mean_absolute_error(y_train, train_pred)
        # train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        
        # test_mae = mean_absolute_error(y_test, test_pred)
        # test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        r2 = r2_score(y_test, test_pred)
        mape = mean_absolute_percentage_error(y_test, test_pred)
        mlflow.log_metric("Accuracy", r2)
        mlflow.log_metric("MAPE", mape)
    
    # print(f"Train MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}")
    # print(f"Test MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}")
    # print(f"r2: {r2:.2f}")
    # print(f"mape: {mape:.2f}")
  
    
    # Generate future dates for forecasting
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_horizon+1)]
    future_df = pd.DataFrame(index=future_dates)
    future_df = create_features(future_df)
    
    
    # Make forecasts
    forecast = model.predict(future_df)
    
    return model, train_pred, test_pred, forecast, future_dates

def plot_results(data, train_pred, test_pred, forecast, future_dates):
    """Plot training, test and forecast results"""
    plt.figure(figsize=(15, 6))
    
    # Plot historical data
    plt.plot(data.index, data['total_sales'], label='Actual Sales', color='blue')
    
    # Plot training predictions
    train_dates = data.index[:-len(test_pred)]
    plt.plot(train_dates, train_pred, label='Training Predictions', color='green', linestyle='--')
    
    # Plot test predictions
    test_dates = data.index[-len(test_pred):]
    plt.plot(test_dates, test_pred, label='Test Predictions', color='red', linestyle='--')
    
    # Plot forecast
    plt.plot(future_dates, forecast, label='Forecast', color='purple', marker='o')
    
    plt.title('Coffee Shop Sales Forecasting')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_model(model, filename='./model/forecast_model.pkl'):
    """Save the trained model and scaler"""
    with open(filename, 'wb') as f:
        pickle.dumps(model)

# def main():
#     # Load and preprocess data
#     data = load_and_preprocess_data('./data/Coffee Shop Sales.csv')
    
#     # Train model and make forecasts
#     model, train_pred, test_pred, forecast, future_dates = train_forecast_model(data)
    
#     # Plot results
#     plot_results(data, train_pred, test_pred, forecast, future_dates)
    
#     # Save model
#     save_model(model)
    
#     # Print forecast
#     print("\nForecasted Sales:")
#     for date, amount in zip(future_dates, forecast):
#         print(f"{date.date()}: ${amount:.2f}")

# if __name__ == "__main__":
#     main()