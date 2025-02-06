import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

# Function to fetch stock data
def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

# Function to create dataset for LSTM
def create_lstm_dataset(data, time_step=100):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Function to train LSTM model
def train_lstm_model(data, time_step=100, epochs=10, batch_size=32):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)
    X, y = create_lstm_dataset(data_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, scaler

# Function to predict future stock price
def predict_stock_price(model, scaler, data, time_step=100):
    data_scaled = scaler.transform(data)
    X_input = data_scaled[-time_step:].reshape(1, time_step, 1)
    predicted_price = model.predict(X_input)
    return scaler.inverse_transform(predicted_price)[0,0]

# Streamlit App UI
st.title('Banking Sector Financial Dashboard')

# Stock Selection
ticker_list = ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'BANKBARODA.NS']
ticker = st.selectbox('Select Stock', ticker_list)

# Fetch stock data
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=1825)).strftime('%Y-%m-%d')
data = get_stock_data(ticker, start_date, end_date)

# Display live stock price
stock_info = yf.Ticker(ticker).info
current_price = stock_info.get('currentPrice', 'N/A')
st.metric(label="Live Stock Price", value=f"₹{current_price}")

# Sidebar Metrics
st.sidebar.header("📌 Key Metrics")
if not data.empty:
    latest_data = data.iloc[-1]
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
    st.sidebar.warning(f"No stock data available for {ticker}.")

# Market Overview
st.header("📈 Market Overview")
col1, col2, col3 = st.columns(3)

# Stock Price History Graph
with col1:
    st.subheader(f"{ticker} Trend")
    if not data.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(data.index, data['Close'], label=f"{ticker} Close", color='red')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning(f"No data available for {ticker}.")

# Financial Analysis
st.header("📊 Financial Analysis")
col4, col5, col6 = st.columns([2, 1, 1])

# Profit vs Revenue Comparison Graph
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

# Stock Data Table
with col5:
    st.subheader("📋 Stock Data Table")
    if not data.empty:
        st.dataframe(data.tail(10).style.format({"Close": "{:.2f}", "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}"}))
    else:
        st.warning("No stock data available.")
