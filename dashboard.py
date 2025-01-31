import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import googleapiclient.discovery

# Function to fetch live stock data from Yahoo Finance
def fetch_stock_data(ticker):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=5*365)
    df = yf.download(ticker, start=start, end=end)
    return df

# Function to prepare data for LSTM model
def prepare_data(df):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

# Function to create dataset
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Function to build and train LSTM model
def train_lstm(X_train, Y_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=20, batch_size=16, verbose=1)
    return model

# Function to make predictions
def predict_stock(model, data, scaler, time_step=60):
    X_test, _ = create_dataset(data, time_step)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    predictions = model.predict(X_test)
    return scaler.inverse_transform(predictions)

# Function to get buy/sell signals
def buy_sell_decision(predictions):
    signals = []
    for i in range(1, len(predictions)):
        if predictions[i] > predictions[i-1]:
            signals.append("BUY")
        else:
            signals.append("SELL")
    return signals

# Streamlit UI
st.title("HDFC Bank Stock Prediction Dashboard")

# Fetch stock data
ticker = "HDFCBANK.NS"
data = fetch_stock_data(ticker)
st.subheader("Stock Data")
st.write(data.tail())

# Prepare data for prediction
data_scaled, scaler = prepare_data(data)
X, Y = create_dataset(data_scaled)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train LSTM model
st.subheader("Training LSTM Model...")
model = train_lstm(X, Y)

# Predict stock price
st.subheader("Stock Price Prediction")
predictions = predict_stock(model, data_scaled, scaler)

# Display prediction results
st.line_chart(predictions)

# Generate buy/sell signals
signals = buy_sell_decision(predictions)
st.subheader("Buy/Sell Decision")
st.write(signals[-5:])
