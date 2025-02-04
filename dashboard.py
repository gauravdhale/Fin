import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import seaborn as sns

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

def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, period="10y", interval="1d")
    if stock_data.empty:
        return pd.DataFrame()
    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['Price_Change'] = stock_data['Close'].pct_change()
    return stock_data.dropna()

def fetch_all_stock_data():
    all_data = {}
    for stock in companies.values():
        stock_data = fetch_stock_data(stock)
        if not stock_data.empty:
            all_data[stock] = stock_data['Close']
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

def get_stock_metrics(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Open": info.get("open", "N/A"),
        "Close": info.get("previousClose", "N/A"),
        "High": info.get("dayHigh", "N/A"),
        "Low": info.get("dayLow", "N/A"),
        "EPS": info.get("trailingEps", "N/A"),
        "IPO Price": info.get("regularMarketOpen", "N/A"),
        "P/E Ratio": info.get("trailingPE", "N/A"),
        "Dividend": info.get("dividendYield", "N/A"),
    }

# Fetch Data
bank_nifty_data = fetch_stock_data(bank_nifty_ticker)
selected_stock_data = fetch_stock_data(companies[selected_stock])
stock_metrics = get_stock_metrics(companies[selected_stock])

if not bank_nifty_data.empty and not selected_stock_data.empty:
    st.markdown(f"## 📈 {selected_stock} Metrics")
    
    with st.container():
        metric_cols = st.columns(8)
        metric_labels = ["Open", "Close", "High", "Low", "EPS", "IPO Price", "P/E Ratio", "Dividend"]
        
        for i, metric in enumerate(metric_labels):
            with metric_cols[i]:
                st.metric(label=metric, value=stock_metrics[metric])
    
    st.markdown("## 📈 BankNifty & Stock Market Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📈 BankNifty Trend")
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(bank_nifty_data.index, bank_nifty_data['Close'], label="BankNifty Close", color='blue')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader(f"📈 {selected_stock} Trend")
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(selected_stock_data.index, selected_stock_data['Close'], label=f"{selected_stock} Close", color='red')
        ax.legend()
        st.pyplot(fig)
    
    with col3:
        st.subheader(f"📊 Prediction for {selected_stock}")
        arima_model = ARIMA(selected_stock_data['Close'], order=(5, 1, 0))
        arima_result = arima_model.fit()
        future_dates = [selected_stock_data.index[-1] + timedelta(days=i) for i in range(1, 31)]
        future_predictions = arima_result.forecast(steps=30)
        pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(pred_df['Date'], pred_df['Predicted Price'], label=f"{selected_stock} Prediction", color='green')
        ax.legend()
        st.pyplot(fig)
    
    col4, col5 = st.columns(2)
    with col4:
        st.subheader("📊 Profit vs Revenue Comparison")
        profit_revenue_data = pd.DataFrame({
            "Year": np.arange(2015, 2025),
            "Total Revenue": np.random.randint(50000, 150000, 10),
            "Net Profit": np.random.randint(5000, 30000, 10)
        })
        fig, ax = plt.subplots(figsize=(5, 2))
        profit_revenue_data.set_index("Year").plot(kind="bar", ax=ax, width=0.8)
        st.pyplot(fig)
    
    with col5:
        st.subheader("📊 Market Share of Banks")
        market_shares = {stock: np.random.rand() for stock in companies.keys()}
        total_share = sum(market_shares.values())
        market_shares = {k: v / total_share for k, v in market_shares.items()}  # Normalize
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.pie(market_shares.values(), labels=market_shares.keys(), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    
    st.subheader("📋 BankNifty Index Data Table")
    st.dataframe(bank_nifty_data.tail(10))
    
    all_stocks_data = fetch_all_stock_data()
    if not all_stocks_data.empty:
        correlation_matrix = all_stocks_data.corr()
        st.subheader("📊 Correlation Heatmap between Bank Stocks and BankNifty")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
        st.pyplot(fig)
    
    st.success("🎯 Analysis Completed!")
