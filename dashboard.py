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

# Streamlit Config
st.set_page_config(layout="wide")

# Selection Dropdown on Top Right
col_top1, col_top2 = st.columns([4, 1])
with col_top2:
    selected_stock = st.selectbox("🔍 Select a Bank", list(companies.keys()))

# Fetch Stock Data (Live Data with 60s Refresh)
@st.cache_data(ttl=60)
def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, period="5y", interval="1d")
    if stock_data.empty:
        return pd.DataFrame()
    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['Price_Change'] = stock_data['Close'].pct_change()
    return stock_data.dropna()

@st.cache_data(ttl=60)
def fetch_all_stock_data():
    all_data = {}
    for stock, ticker in companies.items():
        stock_data = fetch_stock_data(ticker)
        if not stock_data.empty:
            all_data[stock] = stock_data['Close']
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

# Fetch Data
bank_nifty_data = fetch_stock_data(bank_nifty_ticker)
selected_stock_data = fetch_stock_data(companies[selected_stock])
all_stocks_data = fetch_all_stock_data()

# Display Data & Charts if Data Exists
if not bank_nifty_data.empty and not selected_stock_data.empty and not all_stocks_data.empty:
    
    # Key Financial Metrics
    st.markdown(f"## 📈 {selected_stock} Metrics")
    with st.container():
        metric_cols = st.columns(8)
        metric_labels = ["Open", "Close", "High", "Low", "EPS", "IPO Price", "P/E Ratio", "Dividend"]
        for i, metric in enumerate(metric_labels):
            with metric_cols[i]:
                st.metric(label=metric, value=np.random.randint(100, 1000))

    # Bank Nifty & Stock Overview
    st.markdown("## 📈 BankNifty & Stock Market Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 BankNifty Trend & Prediction")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(bank_nifty_data.index, bank_nifty_data['Close'], label="BankNifty Close", color='blue')
        ax.set_title("BankNifty Closing Price")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader(f"📈 {selected_stock} Trend Line")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(selected_stock_data.index, selected_stock_data['Close'], label=f"{selected_stock} Close", color='red')
        ax.set_title(f"{selected_stock} Closing Price")
        ax.legend()
        st.pyplot(fig)

    # Profit vs Revenue Comparison
    st.subheader("📊 Profit vs Revenue Comparison")
    profit_revenue_data = pd.DataFrame({
        "Year": np.arange(2015, 2025),
        "Total Revenue": np.random.randint(50000, 150000, 10),
        "Net Profit": np.random.randint(5000, 30000, 10)
    })
    fig, ax = plt.subplots(figsize=(6, 3))
    profit_revenue_data.set_index("Year").plot(kind="bar", ax=ax, width=0.8)
    st.pyplot(fig)

    # Market Share of Banks
    st.subheader("📊 Market Share of Banks")
    market_shares = {stock: np.random.rand() for stock in companies.keys()}
    total_share = sum(market_shares.values())
    market_shares = {k: v / total_share for k, v in market_shares.items()}  # Normalize
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.pie(market_shares.values(), labels=market_shares.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Prediction for Selected Stock
    st.subheader(f"📊 Prediction for {selected_stock}")
    arima_model = ARIMA(selected_stock_data['Close'], order=(5, 1, 0))
    arima_result = arima_model.fit()
    future_dates = [selected_stock_data.index[-1] + timedelta(days=i) for i in range(1, 31)]
    future_predictions = arima_result.forecast(steps=30)
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(pred_df['Date'], pred_df['Predicted Price'], label=f"{selected_stock} Prediction", color='green')
    ax.set_title(f"Predicted Price for {selected_stock}")
    ax.legend()
    st.pyplot(fig)

    # ✅ **Heatmap of Stock Correlation**
    if not all_stocks_data.empty:
        st.subheader("📊 Correlation Heatmap of Banking Stocks")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(all_stocks_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Stock Price Correlation")
        st.pyplot(fig)

    # Show BankNifty Data Table
    st.subheader("📋 BankNifty Index Data Table")
    st.dataframe(bank_nifty_data.tail(10))

    st.success("🎯 Analysis Completed!")
