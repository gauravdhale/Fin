import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

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
        return stock_data[['Close']].dropna()
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Fetch Data
selected_stock_data = fetch_stock_data(companies[selected_stock])

# Prediction using LSTM Model
st.subheader(f"Prediction for {selected_stock}")
if not selected_stock_data.empty:
    try:
        # Preprocessing Data
        data = selected_stock_data.values
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data)
        
        # Prepare Training Data
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        
        def create_dataset(data, time_step=50):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)
        
        time_step = 50
        X_train, Y_train = create_dataset(train_data, time_step)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        # Build LSTM Model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=0)
        
        # Prepare Test Data
        last_50_days = scaled_data[-time_step:]
        last_50_days = last_50_days.reshape(1, time_step, 1)
        
        # Predict Future Prices
        future_steps = 5
        predictions = []
        for _ in range(future_steps):
            pred_price = model.predict(last_50_days)[0][0]
            predictions.append(pred_price)
            last_50_days = np.roll(last_50_days, -1)
            last_50_days[0, -1, 0] = pred_price
        
        future_dates = pd.date_range(start=selected_stock_data.index[-1], periods=future_steps + 1, freq='B')[1:]
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
        
        # Plot Prediction Graph
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(future_dates, predictions, label="Predicted Price", color='green', linestyle="dashed", marker='o')
        ax.set_title(f"{selected_stock} Predicted Price (Next {future_steps} Days)", fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Stock Price (INR)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=12)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.warning(f"No data available for prediction on {selected_stock}.")
