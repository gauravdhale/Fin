import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
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

# Prepare Data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(selected_stock_data[['Close']])

# Create sequences
sequence_length = 60  # 60 days lookback
X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i+sequence_length])
    y.append(data_scaled[i+sequence_length])
X, y = np.array(X), np.array(y)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define LSTM Model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Make Predictions
future_steps = 5
future_inputs = data_scaled[-sequence_length:]
predictions = []

for _ in range(future_steps):
    pred_input = np.expand_dims(future_inputs, axis=0)
    pred_price = model.predict(pred_input)[0][0]
    predictions.append(pred_price)
    future_inputs = np.vstack((future_inputs[1:], [[pred_price]]))

# Transform predictions back to original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Generate future dates
future_dates = pd.date_range(start=selected_stock_data.index[-1], periods=future_steps + 1, freq='B')[1:]

# Plot Prediction Graph
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(future_dates, predictions, label="Predicted Price", color='green', linestyle="dashed", marker='o')
ax.set_title(f"{selected_stock} Predicted Price (Next {future_steps} Days)", fontsize=14)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Stock Price (INR)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=12)
st.pyplot(fig)
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
            if isinstance(stock_data, pd.DataFrame):
                # If it's a DataFrame, extract the 'Close' column (or any other relevant column)
                if 'Close' in stock_data.columns:
                    aligned_data[name] = stock_data['Close']
                else:
                    st.error(f"⚠️ 'Close' column missing in {name} data")
            elif isinstance(stock_data, pd.Series):
                aligned_data[name] = stock_data
            elif isinstance(stock_data, (int, float)):  # If scalar value
                aligned_data[name] = pd.Series([stock_data], index=[0])  # Wrap in Series with index
            else:
                st.error(f"⚠️ Invalid data format for {name}")

        # Now ensure that the data is aligned
        if aligned_data:
            # If all data is scalar, wrap the data in a list and pass index
            if all(isinstance(v, (int, float)) for v in aligned_data.values()):
                aligned_data = {k: pd.Series([v], index=[0]) for k, v in aligned_data.items()}
            
            # Now create the DataFrame
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
