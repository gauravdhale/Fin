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
from sklearn.metrics import mean_squared_error, r2_score

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

def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, period="10y", interval="1d")
    if stock_data.empty:
        st.error(f"Error: No data found for {ticker}.")
        return pd.DataFrame()
    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['Price_Change'] = stock_data['Close'].pct_change()
    return stock_data.dropna()

def fetch_live_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Current Price": info.get("currentPrice", "N/A"),
        "Open": info.get("open", "N/A"),
        "Close": info.get("previousClose", "N/A"),
        "P/E Ratio": info.get("trailingPE", "N/A"),
        "Volume": info.get("volume", "N/A"),
        "IPO Price": info.get("regularMarketPreviousClose", "N/A")
    }

# Fetch Data
stock_data = fetch_stock_data(companies[selected_stock])
live_data = fetch_live_data(companies[selected_stock])

if not stock_data.empty:
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Open Price", live_data["Open"])
    col2.metric("Close Price", live_data["Close"])
    col3.metric("Current Price", live_data["Current Price"])
    col4.metric("P/E Ratio", live_data["P/E Ratio"])
    col5.metric("Volume", live_data["Volume"])
    col6.metric("IPO Price", live_data["IPO Price"])
    
    st.subheader(f"Stock Price Trend: {selected_stock}")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data.index, stock_data['Close'], label="Close Price", color='blue')
    ax.plot(stock_data.index, stock_data['MA_20'], label="20-Day MA", linestyle='dashed', color='orange')
    ax.plot(stock_data.index, stock_data['MA_50'], label="50-Day MA", linestyle='dashed', color='green')
    ax.legend()
    st.pyplot(fig)
    
    def train_model(data):
        X = data[['Open', 'High', 'Low', 'MA_20', 'MA_50', 'Price_Change']]
        y = data['Close']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        models = {
            "Linear Regression": LinearRegression(),
            "SVR": SVR(kernel='rbf', C=10, epsilon=0.1),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10)
        }
        for model in models.values():
            model.fit(X_train, y_train)
        voting_model = VotingRegressor([(name, model) for name, model in models.items()])
        voting_model.fit(X_train, y_train)
        y_pred = voting_model.predict(X_test)
        return y_test, y_pred
    
    y_test, y_pred = train_model(stock_data)
    
    st.subheader(f"Model Performance for {selected_stock}")
    st.write(f"Mean Squared Error: {np.round(mean_squared_error(y_test, y_pred), 2)}")
    st.write(f"R² Score: {np.round(r2_score(y_test, y_pred), 2)}")
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(y_test.index, y_test, label="Actual Price", color='blue')
    ax2.plot(y_test.index, y_pred, label="Predicted Price", linestyle='dashed', color='red')
    ax2.legend()
    st.pyplot(fig2)
    
    def predict_future(data):
        arima_model = ARIMA(data['Close'], order=(5, 1, 0))
        arima_result = arima_model.fit()
        future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 31)]
        future_predictions = arima_result.forecast(steps=30)
        return pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
    
    future_predictions = predict_future(stock_data)
    st.subheader(f"Future Price Predictions for {selected_stock}")
    st.dataframe(future_predictions)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(future_predictions['Date'], future_predictions['Predicted Price'], label="Future Price", color='purple')
    ax3.legend()
    st.pyplot(fig3)
    
    st.success("Analysis Completed!")
