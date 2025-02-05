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
            # Train ARIMA Model
            arima_model = ARIMA(selected_stock_data['Close'], order=(5, 1, 0))
            arima_result = arima_model.fit()

            # Define forecast steps
            future_steps = 30
            future_dates = pd.date_range(start=selected_stock_data.index[-1], periods=future_steps + 1, freq='B')[1:]

            # Forecasting
            forecast_result = arima_result.get_forecast(steps=future_steps)
            forecast = forecast_result.predicted_mean

            # Plot Prediction Graph (Only Predicted Prices)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(future_dates, forecast, label="Predicted Price", color='green', linestyle="dashed", marker='o')

            ax.set_title(f"{selected_stock} Predicted Price (Next {future_steps} Days)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price (INR)")
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

# BankNifty Data Table
st.subheader("📋 BankNifty Index Data Table")
if not bank_nifty_data.empty:
    st.dataframe(bank_nifty_data.tail(10))
else:
    st.warning("No BankNifty data available.")

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
