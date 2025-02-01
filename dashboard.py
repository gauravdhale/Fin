import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
st.set_page_config(layout="wide")
st.sidebar.title("📊 AI Banking Sector Stock Dashboard")
selected_stock = st.sidebar.selectbox("🔍 Select a Bank", list(companies.keys()))

def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, period="10y", interval="1d")
    if stock_data.empty:
        st.error(f"⚠️ Error: No data found for {ticker}.")
        return pd.DataFrame()
    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['Price_Change'] = stock_data['Close'].pct_change()
    return stock_data.dropna()

def fetch_live_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Current Price": f"{info.get('currentPrice', 0):.4f}",
        "Open": f"{info.get('open', 0):.4f}",
        "Close": f"{info.get('previousClose', 0):.4f}",
        "P/E Ratio": f"{info.get('trailingPE', 0):.4f}",
        "Volume": f"{info.get('volume', 0):,.4f}",
        "IPO Price": f"{info.get('regularMarketPreviousClose', 0):.4f}",
        "EPS": f"{info.get('trailingEps', 0):.4f}"
    }

# Fetch Data
stock_data = fetch_stock_data(companies[selected_stock])
live_data = fetch_live_data(companies[selected_stock])

if not stock_data.empty:
    # Layout for Metrics
    st.markdown("## 📈 Stock Market Overview")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("📌 Open Price", live_data["Open"])
    col2.metric("💰 Close Price", live_data["Close"])
    col3.metric("📊 Current Price", live_data["Current Price"])
    col4.metric("📉 P/E Ratio", live_data["P/E Ratio"])
    col5.metric("📊 Volume", live_data["Volume"])
    col6.metric("🚀 IPO Price", live_data["IPO Price"])

    # Layout for Charts and Predictions
    col7, col8 = st.columns([2, 1])

    with col7:
        st.subheader(f"📈 Stock Price Trend: {selected_stock}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(stock_data.index, stock_data['Close'], label="Close Price", color='blue')
        ax.plot(stock_data.index, stock_data['MA_20'], label="20-Day MA", linestyle='dashed', color='orange')
        ax.plot(stock_data.index, stock_data['MA_50'], label="50-Day MA", linestyle='dashed', color='green')
        ax.legend()
        st.pyplot(fig)

    with col8:
        st.subheader("📊 EPS & Heatmap")
        st.metric("📈 EPS", live_data["EPS"])

        # Heatmap
        corr = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

    # Predictions
    st.subheader("🚀 Future Price Predictions")
    col9, col10, col11 = st.columns(3)

    def predict_future(data, days):
        try:
            if len(data) < 50:
                return np.nan  # Return NaN if insufficient data
            arima_model = ARIMA(data['Close'], order=(5, 1, 0))
            arima_result = arima_model.fit()
            future_predictions = arima_result.forecast(steps=days)
            return future_predictions[-1] if len(future_predictions) > 0 else np.nan
        except Exception:
            return np.nan  # Return NaN in case of errors

    def calculate_error(predicted, actual):
        return ((predicted - actual) / actual * 100) if not np.isnan(actual) and not np.isnan(predicted) else np.nan

    # 1-Day Prediction
    one_day_pred = predict_future(stock_data, 1)
    error_1day = calculate_error(one_day_pred, stock_data['Close'].iloc[-1])
    col9.metric("1-Day Prediction", f"{one_day_pred:.2f}" if not np.isnan(one_day_pred) else "Unavailable", delta=f"{error_1day:.2f}%" if not np.isnan(error_1day) else "0.00%")

    # 1-Week Prediction
    one_week_pred = predict_future(stock_data, 7)
    error_1week = calculate_error(one_week_pred, stock_data['Close'].iloc[-1])
    col10.metric("1-Week Prediction", f"{one_week_pred:.2f}" if not np.isnan(one_week_pred) else "Unavailable", delta=f"{error_1week:.2f}%" if not np.isnan(error_1week) else "0.00%")

    # Error Percentage
    col11.metric("Error Percentage", f"{error_1day:.2f}%" if not np.isnan(error_1day) else "0.00%")

    st.success("🎯 Analysis Completed!")
