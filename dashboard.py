import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Define Banking Stocks and Bank Nifty Index
companies = {
    'HDFC Bank': 'HDFCBANK.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'State Bank of India': 'SBIN.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Bank of Baroda': 'BANKBARODA.NS'
}
bank_nifty_ticker = "^NSEBANK"

# Streamlit Configuration
st.set_page_config(page_title="Banking Sector Dashboard", layout="wide")
st.title("📊 Banking Sector Financial Dashboard")
st.markdown("---")

# Selection Dropdown
selected_stock = st.sidebar.selectbox("🔍 Select a Bank", list(companies.keys()))

# Function to Fetch Stock Data
def fetch_stock_data(ticker, period="5y"):
    try:
        stock_data = yf.download(ticker, period=period, interval="1d")
        if stock_data.empty:
            return pd.DataFrame()
        stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['Price_Change'] = stock_data['Close'].pct_change()
        return stock_data.dropna()
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Fetch Data
bank_nifty_data = fetch_stock_data(bank_nifty_ticker)
selected_stock_data = fetch_stock_data(companies[selected_stock])

# Display Metrics if Data is Available
st.sidebar.header("📌 Key Metrics")
if not selected_stock_data.empty:
    latest_data = selected_stock_data.iloc[-1]
    metric_values = {
        "Open": latest_data["Open"],
        "Close": latest_data["Close"],
        "High": latest_data["High"],
        "Low": latest_data["Low"],
        "EPS": np.random.uniform(10, 50),  
        "IPO Price": np.random.uniform(200, 1000),  
        "P/E Ratio": np.random.uniform(5, 30),  
        "Dividend": np.random.uniform(1, 5)  
    }
    for label, value in metric_values.items():
        st.sidebar.metric(label=label, value=f"{value:.2f}" if isinstance(value, (int, float)) else value)
else:
    st.sidebar.warning(f"No stock data available for {selected_stock}.")

# BankNifty and Stock Overview
st.header("📈 Market Overview")
col1, col2 = st.columns(2)

# BankNifty Trend Graph
with col1:
    st.subheader("BankNifty Trend")
    if not bank_nifty_data.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(bank_nifty_data.index, bank_nifty_data['Close'], label="BankNifty Close", color='blue')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("No data available for BankNifty.")

# Selected Stock Trend Graph
with col2:
    st.subheader(f"{selected_stock} Trend")
    if not selected_stock_data.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(selected_stock_data.index, selected_stock_data['Close'], label=f"{selected_stock} Close", color='red')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning(f"No data available for {selected_stock}.")

# Prediction using ARIMA Model
st.header(f"📈 {selected_stock} - Actual vs Predicted Price")
if not selected_stock_data.empty:
    try:
        arima_model = ARIMA(selected_stock_data['Close'], order=(5, 1, 0))
        arima_result = arima_model.fit()
        
        future_steps = 5
        future_dates = pd.date_range(start=selected_stock_data.index[-1], periods=future_steps + 1, freq='B')[1:]
        
        forecast_result = arima_result.get_forecast(steps=future_steps)
        forecast = forecast_result.predicted_mean
        forecast_conf_int = forecast_result.conf_int()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(selected_stock_data.index, selected_stock_data['Close'], label="Actual Price", color='blue')
        ax.plot(future_dates, forecast, label="Predicted Price", color='red', linestyle="dashed", marker='o')
        ax.fill_between(future_dates, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3)
        
        ax.set_title(f"{selected_stock} - Actual vs Predicted Price", fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Stock Price (INR)", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.warning(f"No data available for prediction on {selected_stock}.")
