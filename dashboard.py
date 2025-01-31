import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Define Banking Stocks
companies = {
    'HDFC Bank': 'HDFCBANK.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'State Bank of India': 'SBIN.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Bank of Baroda': 'BANKBARODA.NS'
}

# Streamlit Sidebar
st.sidebar.title("Banking Sector Stock Analysis")
selected_stock = st.sidebar.selectbox("Select a Bank", list(companies.keys()))

# Fetch Stock Data
@st.cache_data
def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, period="10y", interval="1d")
    
    if stock_data.empty:
        st.error(f"Error: No data found for {ticker}.")
        return pd.DataFrame()
    
    st.write("Columns in fetched data:", stock_data.columns)
    
    required_columns = ['Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in stock_data.columns:
            st.error(f"Error: Column '{col}' missing in {ticker} data.")
            return pd.DataFrame()
    
    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['Price_Change'] = stock_data['Close'].pct_change()
    return stock_data.dropna()

# Load Data
stock_data = fetch_stock_data(companies[selected_stock])

if not stock_data.empty:
    # Display Data & Chart
    st.subheader(f"Stock Price Data for {selected_stock}")
    st.dataframe(stock_data.tail())
    
    st.subheader(f"Stock Price Trend: {selected_stock}")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data.index, stock_data['Close'], label="Close Price", color='blue')
    ax.plot(stock_data.index, stock_data['MA_20'], label="20-Day MA", linestyle='dashed', color='orange')
    ax.plot(stock_data.index, stock_data['MA_50'], label="50-Day MA", linestyle='dashed', color='green')
    ax.legend()
    st.pyplot(fig)
    
    # Train Machine Learning Models
    def train_model(data):
        X = data[['Open', 'High', 'Low', 'MA_20', 'MA_50', 'Price_Change']]
        y = data['Close']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        lr_model = LinearRegression()
        svr_model = SVR(kernel='rbf', C=10, epsilon=0.1)
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10)

        lr_model.fit(X_train, y_train)
        svr_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)

        voting_model = VotingRegressor([('lr', lr_model), ('svr', svr_model), ('rf', rf_model)])
        voting_model.fit(X_train, y_train)

        y_pred = voting_model.predict(X_test)
        return y_test, y_pred, voting_model

    y_test, y_pred, model = train_model(stock_data)

    # Display Model Evaluation
    st.subheader(f"Model Performance for {selected_stock}")
    st.write(f"Mean Squared Error: {np.round(mean_squared_error(y_test, y_pred), 2)}")
    st.write(f"R² Score: {np.round(r2_score(y_test, y_pred), 2)}")
    
    # Plot Predictions
    st.subheader(f"Actual vs Predicted Prices for {selected_stock}")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(y_test.index, y_test, label="Actual Price", color='blue')
    ax2.plot(y_test.index, y_pred, label="Predicted Price", linestyle='dashed', color='red')
    ax2.legend()
    st.pyplot(fig2)
    
    # ARIMA Forecasting
    def predict_future(data):
        arima_model = ARIMA(data['Close'], order=(5, 1, 0))
        arima_result = arima_model.fit()
        future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 31)]
        future_predictions = arima_result.forecast(steps=30)
        return pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})

    future_predictions = predict_future(stock_data)
    
    # Display Future Forecast
    st.subheader(f"Future Price Predictions for {selected_stock}")
    st.dataframe(future_predictions)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(future_predictions['Date'], future_predictions['Predicted Price'], label="Future Price", color='purple')
    ax3.legend()
    st.pyplot(fig3)
    
    st.success("Analysis Completed!")

