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
            future_steps = 5
            future_dates = pd.date_range(start=selected_stock_data.index[-1], periods=future_steps + 1, freq='B')[1:]

            # Forecasting
            forecast_result = arima_result.get_forecast(steps=future_steps)
            forecast = forecast_result.predicted_mean

            # Plot Prediction Graph (Only Predicted Prices)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(future_dates, forecast, label="Predicted Price", color='green', linestyle="dashed", marker='o')

            ax.set_title(f"{selected_stock} Predicted Price (Next {future_steps} Days)", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Stock Price (INR)", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.6)  # Grid for better readability
            ax.legend(fontsize=12)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning(f"No data available for prediction on {selected_stock}.")

# Financial Analysis Section
st.header("📊 Financial Analysis")

# Create three columns for better layout
col4, col5, col6 = st.columns([2, 1, 1])  # Adjusting width for better visibility

# 🔹 Profit vs Revenue Comparison Graph (Existing Code)
with col4:
    st.subheader("📈 Profit vs Revenue Comparison")
    
    profit_revenue_data = pd.DataFrame({
        "Year": np.arange(2015, 2025),
        "Total Revenue": np.random.randint(50000, 150000, 10),
        "Net Profit": np.random.randint(5000, 30000, 10)
    })

    fig, ax = plt.subplots(figsize=(5, 3))
    profit_revenue_data.set_index("Year").plot(kind="bar", ax=ax, width=0.8, colormap="coolwarm")

    ax.set_title("Total Revenue vs Net Profit", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Amount (INR in Lakhs)", fontsize=12)
    ax.grid(axis='y', linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)

    st.pyplot(fig)

# 🔹 BankNifty Index Data Table (Existing Code)
with col5:
    st.subheader("📋 BankNifty Index Data Table")
    
    if not bank_nifty_data.empty:
        st.dataframe(bank_nifty_data.tail(10).style.format({"Close": "{:.2f}", "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}"}))
    else:
        st.warning("No BankNifty data available.")
# 🔹 Heatmap for Nifty Bank Companies
with col6:
    st.subheader("📊 Correlation Heatmap for Nifty Bank Companies")

# Fetch Data for all companies
data = {name: fetch_stock_data(ticker) for name, ticker in companies.items()}

# Debug: Print the structure of data
for key, value in data.items():
    if value is None or value.empty:
        print(f"⚠️ Warning: No data for {key}")
    else:
        print(f"✅ {key} data loaded: {value.shape}")

# Remove None or empty values
filtered_data = {k: v for k, v in data.items() if v is not None and not v.empty}

if filtered_data:
    try:
        # Ensure each stock's data is a pandas Series or DataFrame
        aligned_data = {}

        for name, stock_data in filtered_data.items():
            # Check if the data is a DataFrame and extract the 'Close' price if it is
            if isinstance(stock_data, pd.DataFrame):
                if 'Close' in stock_data.columns:
                    aligned_data[name] = stock_data['Close']
                else:
                    st.error(f"⚠️ 'Close' column missing in {name} data")
            elif isinstance(stock_data, pd.Series):
                aligned_data[name] = stock_data
            else:
                # If it's a scalar value (int or float), wrap it in a series
                aligned_data[name] = pd.Series([stock_data])

        # Ensure all columns (companies) have the same index for correlation calculation
        # Get the intersection of all dates (to ensure the same timeframe for all companies)
        aligned_data = {k: v for k, v in aligned_data.items() if v is not None}
        
        # If there's data for all companies, proceed
        if aligned_data:
            stock_prices = pd.DataFrame(aligned_data)

            if stock_prices.empty:
                st.warning("Stock data is empty after filtering.")
            else:
                stock_prices.dropna(inplace=True)

                # Correlation Matrix
                correlation_matrix = stock_prices.corr()

                # Plot Heatmap
                st.subheader("📊 Correlation Heatmap for Nifty Bank Companies")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
                st.pyplot(fig)
        else:
            st.warning("No valid stock data available for the heatmap.")
    except Exception as e:
        st.error(f"Error processing stock data: {e}")
else:
    st.warning("No valid stock data available to generate heatmap.")



st.success("🎯 Analysis Completed!")
