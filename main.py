# forecasting_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from time_series.pipeline import load_and_preprocess_data, train_forecast_model

# Your existing functions (load_and_preprocess_data, create_features, train_forecast_model)
# Keep them exactly the same as in your pipeline.py

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
    return plt

def save_model(model, filename='./model/forecast_model.pkl'):
    """Save the trained model"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def main():
    st.title('☕︎ Coffee Shop Sales Forecasting ☕︎')
    st.write('Forecasts sales for the next 7 days.')
    
    # Load and preprocess data
    data = load_and_preprocess_data('./data/Coffee Shop Sales.csv')
    
    if st.button('Generate Forecast'):
        with st.spinner('Training model and generating forecast...'):
            # Train model and make forecasts
            model, train_pred, test_pred, forecast, future_dates = train_forecast_model(data)
            
            # Save model
            save_model(model)
            
            # Plot results
            st.subheader('Sales Forecast')
        
            # Display forecast
            st.subheader('Forecasted Sales for Next 7 Days')
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecasted Sales': forecast
            })
            forecast_df['Date'] = forecast_df['Date'].dt.date
            st.dataframe(forecast_df.style.format({'Forecasted Sales': '${:.2f}'}))
            
            # # Option to download forecast
            # csv = forecast_df.to_csv(index=False).encode('utf-8')
            # st.download_button(
            #     "Download Forecast as CSV",
            #     csv,
            #     "sales_forecast.csv",
            #     "text/csv",
            #     key='download-csv'
            # )

if __name__ == "__main__":
    main()