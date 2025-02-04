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

# Selection Dropdown
col_top1, col_top2 = st.columns([4, 1])
with col_top2:
    selected_stock = st.selectbox("🔍 Select a Bank", list(companies.keys()))

# Function to Fetch Stock Data
def fetch_stock_data(ticker):
    try:
        stock_data = yf.download(ticker, period="10y", interval="1d")
        if stock_data.empty:
            return pd.DataFrame()
        stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['Price_Change'] = stock_data['Close'].pct_change()
        return stock_data.dropna()
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Function to Fetch All Bank Stocks Data
def fetch_all_stock_data():
    all_data = {}
    for name, ticker in companies.items():
        stock_data = fetch_stock_data(ticker)
        if not stock_data.empty:
            all_data[name] = stock_data['Close']
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

# Fetch Data
bank_nifty_data = fetch_stock_data(bank_nifty_ticker)
selected_stock_data = fetch_stock_data(companies[selected_stock])

# Display Metrics if Data is Available
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
else:
    st.warning(f"No stock data available for {selected_stock}. Showing default values.")
    metric_values = {key: "N/A" for key in ["Open", "Close", "High", "Low", "EPS", "IPO Price", "P/E Ratio", "Dividend"]}

# Display Stock Metrics
st.markdown(f"## 📈 {selected_stock} Metrics")
metric_cols = st.columns(8)
for i, (label, value) in enumerate(metric_values.items()):
    with metric_cols[i]:
        st.metric(label=label, value=f"{value:.2f}" if isinstance(value, (int, float)) else value)

# BankNifty and Stock Overview
st.markdown("## 📈 BankNifty & Stock Market Overview")
col1, col2, col3 = st.columns(3)

# BankNifty Trend Graph
with col1:
    st.subheader("📈 BankNifty Trend")
    if not bank_nifty_data.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(bank_nifty_data.index, bank_nifty_data['Close'], label="BankNifty Close", color='blue')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("No data available for BankNifty.")

# Selected Stock Trend Graph
with col2:
    st.subheader(f"📈 {selected_stock} Trend")
    if not selected_stock_data.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(selected_stock_data.index, selected_stock_data['Close'], label=f"{selected_stock} Close", color='red')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning(f"No data available for {selected_stock}.")

# Prediction using ARIMA Model
with col3:
    st.subheader(f"📊 Prediction for {selected_stock}")
    if not selected_stock_data.empty:
        try:
            arima_model = ARIMA(selected_stock_data['Close'], order=(5, 1, 0))
            arima_result = arima_model.fit()
            future_dates = [selected_stock_data.index[-1] + timedelta(days=i) for i in range(1, 31)]
            future_predictions = arima_result.forecast(steps=30)
            pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
            
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(pred_df['Date'], pred_df['Predicted Price'], label=f"{selected_stock} Prediction", color='green')
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning(f"No data available for prediction on {selected_stock}.")

# Profit vs Revenue Comparison
col4, col5 = st.columns(2)
with col4:
    st.subheader("📊 Profit vs Revenue Comparison")
    profit_revenue_data = pd.DataFrame({
        "Year": np.arange(2015, 2025),
        "Total Revenue": np.random.randint(50000, 150000, 10),
        "Net Profit": np.random.randint(5000, 30000, 10)
    })
    fig, ax = plt.subplots(figsize=(5, 3))
    profit_revenue_data.set_index("Year").plot(kind="bar", ax=ax, width=0.8)
    st.pyplot(fig)

# Market Share of Banks
with col5:
    st.subheader("📊 Market Share of Banks")
    market_shares = {stock: np.random.rand() for stock in companies.keys()}
    total_share = sum(market_shares.values())
    market_shares = {k: v / total_share for k, v in market_shares.items()}  
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.pie(market_shares.values(), labels=market_shares.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

# BankNifty Data Table
st.subheader("📋 BankNifty Index Data Table")
if not bank_nifty_data.empty:
    st.dataframe(bank_nifty_data.tail(10))
else:
    st.warning("No BankNifty data available.")

# Correlation Heatmap
all_stocks_data = fetch_all_stock_data()
if not all_stocks_data.empty:
    correlation_matrix = all_stocks_data.corr()
    st.subheader("📊 Correlation Heatmap between Bank Stocks and BankNifty")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
    st.pyplot(fig)
else:
    st.warning("Not enough data to generate a correlation heatmap.")

st.success("🎯 Analysis Completed!")
