import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

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

# Sidebar Styling
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding: 10px;
    }
    [data-testid="stSidebar"] label {
        font-size: 14px !important;
    }
    [data-testid="stSidebar"] div[data-testid="metric-container"] {
        font-size: 12px !important;
        margin: 2px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Selection Dropdown
selected_stock = st.sidebar.selectbox("🔍 Select a Bank", list(companies.keys()))

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

# Fetch Data
bank_nifty_data = fetch_stock_data(bank_nifty_ticker)
selected_stock_data = fetch_stock_data(companies[selected_stock])

# Sidebar Metrics
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
        st.sidebar.metric(label=label, value=f"{value:.2f}")
else:
    st.sidebar.warning(f"No stock data available for {selected_stock}.") 

# Market Overview
st.header("📈 Market Overview")
col1, col2, col3 = st.columns(3)

# BankNifty Trend Graph
with col1:
    st.subheader("BankNifty Trend")
    if not bank_nifty_data.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(bank_nifty_data.index, bank_nifty_data['Close'], label="BankNifty Close", color='blue')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("No data available for BankNifty.")

# Selected Stock Trend Graph
with col2:
    st.subheader(f"{selected_stock} Trend")
    if not selected_stock_data.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(selected_stock_data.index, selected_stock_data['Close'], label=f"{selected_stock} Close", color='red')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning(f"No data available for {selected_stock}.")

# Prediction using ARIMA Model
with col3:
    st.subheader(f"Prediction for {selected_stock}")
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
st.header("📊 Financial Analysis")
col4, col5 = st.columns(2)
with col4:
    st.subheader("Profit vs Revenue Comparison")
    profit_revenue_data = pd.DataFrame({
        "Year": np.arange(2015, 2025),
        "Total Revenue": np.random.randint(50000, 150000, 10),
        "Net Profit": np.random.randint(5000, 30000, 10)
    })
    fig, ax = plt.subplots(figsize=(6, 3))
    profit_revenue_data.set_index("Year").plot(kind="bar", ax=ax, width=0.8)
    st.pyplot(fig)

# Correlation Heatmap
st.subheader("📊 Correlation Heatmap")
all_stocks_data = pd.DataFrame({name: fetch_stock_data(ticker)['Close'] for name, ticker in companies.items() if not fetch_stock_data(ticker).empty})
if not all_stocks_data.empty:
    correlation_matrix = all_stocks_data.corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
    st.pyplot(fig)
else:
    st.warning("Not enough data to generate a correlation heatmap.")

st.success("🎯 Analysis Completed!")
