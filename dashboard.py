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

# Streamlit Sidebar
st.set_page_config(layout="wide")
st.sidebar.title("📊 AI Banking Sector & BankNifty Dashboard")
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

# Fetch Data
bank_nifty_data = fetch_stock_data(bank_nifty_ticker)

if not bank_nifty_data.empty:
    st.markdown("## 📈 BankNifty & Stock Market Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📈 BankNifty Trend & Prediction")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(bank_nifty_data.index, bank_nifty_data['Close'], label="BankNifty Close", color='blue')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Market Share of Top Banks")
        market_shares = {stock: np.random.rand() for stock in companies.keys()}  # Mock data
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.pie(market_shares.values(), labels=market_shares.keys(), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    
    with col3:
        st.subheader("📈 Net Profit Trend")
        net_profit = {stock: np.random.randint(1000, 5000) for stock in companies.keys()}  # Mock data
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(net_profit.keys(), net_profit.values(), color='green')
        st.pyplot(fig)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.subheader("📊 Heatmap: Contribution of Stocks to BankNifty")
        heatmap_data = pd.DataFrame(market_shares, index=["Impact"])
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", linewidths=0.5)
        st.pyplot(fig)
    
    with col5:
        st.subheader("📉 Price Line Graph of BankNifty")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(bank_nifty_data.index, bank_nifty_data['Close'], label="BankNifty Close", color='purple')
        ax.legend()
        st.pyplot(fig)
    
    with col6:
        st.subheader("📊 BankNifty Index Data Table")
        st.dataframe(bank_nifty_data.tail(20))
    
    st.markdown("### 📌 Key Metrics")
    metrics_col = st.columns(4)
    for i, metric in enumerate(["Open", "Close", "High", "Low", "EPS", "IPO Price", "P/E Ratio", "Dividend"]):
        with metrics_col[i % 4]:
            st.metric(label=metric, value=np.random.randint(100, 1000))
    
    st.success("🎯 Analysis Completed!")
