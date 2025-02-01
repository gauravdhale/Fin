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
    try:
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
    except Exception as e:
        st.error(f"Error fetching live data: {e}")
        return {}

# Fetch Data
stock_data = fetch_stock_data(companies[selected_stock])
live_data = fetch_live_data(companies[selected_stock])

if not stock_data.empty:
    # Layout for Metrics
    st.markdown("## 📈 Stock Market Overview")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("📌 Open Price", live_data.get("Open", "N/A"))
    col2.metric("💰 Close Price", live_data.get("Close", "N/A"))
    col3.metric("📊 Current Price", live_data.get("Current Price", "N/A"))
    col4.metric("📉 P/E Ratio", live_data.get("P/E Ratio", "N/A"))
    col5.metric("📊 Volume", live_data.get("Volume", "N/A"))
    col6.metric("🚀 IPO Price", live_data.get("IPO Price", "N/A"))

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
        st.metric("📈 EPS", live_data.get("EPS", "N/A"))

        # Heatmap
        if not stock_data.empty:
            corr = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
            st.pyplot(fig2)

    # Predictions
    st.subheader("🚀 Future Price Predictions")
    col9, col10, col11 = st.columns(3)

    def predict_future(data, days):
        if data.empty:
            st.error("⚠️ No stock data available for prediction.")
            return None
        try:
            arima_model = ARIMA(data['Close'], order=(5, 1, 0))
            arima_result = arima_model.fit()
            future_predictions = arima_result.forecast(steps=days)
            if future_predictions.empty:
                return None
            return future_predictions.iloc[-1]
        except Exception as e:
            st.error(f"⚠️ Error in prediction: {e}")
            return None

    def calculate_error(predicted, actual):
        if predicted is None or actual is None:
            return None
        return ((predicted - actual) / actual) * 100

    # 1-Day Prediction
    one_day_pred = predict_future(stock_data, 1)
    if one_day_pred is not None:
        error_1day = calculate_error(one_day_pred, stock_data['Close'].iloc[-1])
        col9.metric("1-Day Prediction", f"{one_day_pred:.2f}", delta=f"{error_1day:.2f}%" if error_1day is not None else "N/A")

    # 1-Week Prediction
    one_week_pred = predict_future(stock_data, 7)
    if one_week_pred is not None:
        error_1week = calculate_error(one_week_pred, stock_data['Close'].iloc[-1])
        col10.metric("1-Week Prediction", f"{one_week_pred:.2f}", delta=f"{error_1week:.2f}%" if error_1week is not None else "N/A")

    # Error Percentage
    if one_day_pred is not None:
        col11.metric("Error Percentage", f"{error_1day:.2f}%" if error_1day is not None else "N/A")

    # Pie Chart for Price Change Distribution
    st.subheader("📊 Price Change Distribution")
    if not stock_data.empty:
        price_change_bins = pd.cut(stock_data['Price_Change'], bins=[-np.inf, -0.05, 0, 0.05, np.inf], labels=['<-5%', '-5% to 0%', '0% to 5%', '>5%'])
        price_change_dist = price_change_bins.value_counts(normalize=True) * 100
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.pie(price_change_dist, labels=price_change_dist.index, autopct='%1.1f%%', colors=['red', 'orange', 'green', 'blue'])
        st.pyplot(fig3)

    st.success("🎯 Analysis Completed!")
