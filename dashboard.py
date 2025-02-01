import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

# Define Banking Stocks
companies = {
    'HDFC Bank': 'HDFCBANK.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'State Bank of India': 'SBIN.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Bank of Baroda': 'BANKBARODA.NS'
}

# Streamlit Layout Settings
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
        "Current Price": f"{info.get('currentPrice', 0):.2f}",
        "Open": f"{info.get('open', 0):.2f}",
        "Close": f"{info.get('previousClose', 0):.2f}",
        "P/E Ratio": f"{info.get('trailingPE', 0):.2f}",
        "Volume": f"{info.get('volume', 0):,}",
        "IPO Price": f"{info.get('regularMarketPreviousClose', 0):.2f}",
        "EPS": f"{info.get('trailingEps', 0):.2f}"
    }

# Fetch Data
stock_data = fetch_stock_data(companies[selected_stock])
live_data = fetch_live_data(companies[selected_stock])

if not stock_data.empty:
    st.markdown("## 📈 Stock Market Overview")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("📌 Open Price", live_data["Open"])
    col2.metric("💰 Close Price", live_data["Close"])
    col3.metric("📊 Current Price", live_data["Current Price"])
    col4.metric("📉 P/E Ratio", live_data["P/E Ratio"])
    col5.metric("📊 Volume", live_data["Volume"])
    col6.metric("🚀 IPO Price", live_data["IPO Price"])

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
        st.subheader("📊 EPS & Correlation")
        st.metric("📈 EPS", live_data["EPS"])
        
        # Heatmap
        corr = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)
    
    # Pie Chart for Market Share
    st.subheader("📊 Market Share Distribution")
    pie_col1, pie_col2 = st.columns([1, 3])
    with pie_col1:
        market_shares = {k: yf.Ticker(v).info.get("marketCap", 0) for k, v in companies.items()}
        labels = list(market_shares.keys())
        sizes = list(market_shares.values())
        fig3, ax3 = plt.subplots()
        ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
        ax3.axis('equal')
        st.pyplot(fig3)

    # Predictions
    st.subheader("🚀 Future Price Predictions")
    col9, col10, col11 = st.columns(3)

    def predict_future(data, days):
        if len(data) < 50:
            return np.nan
        arima_model = ARIMA(data['Close'], order=(5, 1, 0))
        arima_result = arima_model.fit()
        future_predictions = arima_result.forecast(steps=days)
        return future_predictions[-1] if len(future_predictions) > 0 else np.nan

    def calculate_error(predicted, actual):
        if np.isnan(predicted) or np.isnan(actual):
            return np.nan
        return ((predicted - actual) / actual) * 100

    one_day_pred = predict_future(stock_data, 1)
    one_week_pred = predict_future(stock_data, 7)
    actual_close = stock_data['Close'].iloc[-1]
    
    error_1day = calculate_error(one_day_pred, actual_close)
    error_1week = calculate_error(one_week_pred, actual_close)
    
    col9.metric("1-Day Prediction", f"{one_day_pred:.2f}" if not np.isnan(one_day_pred) else "Unavailable", delta=f"{error_1day:.2f}%" if not np.isnan(error_1day) else "0.00%")
    col10.metric("1-Week Prediction", f"{one_week_pred:.2f}" if not np.isnan(one_week_pred) else "Unavailable", delta=f"{error_1week:.2f}%" if not np.isnan(error_1week) else "0.00%")
    col11.metric("Error Percentage", f"{error_1day:.2f}%" if not np.isnan(error_1day) else "0.00%")

    st.success("🎯 Analysis Completed!")
