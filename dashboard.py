import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------- 🏦 Define Banking Stocks --------------------
companies = {
    'HDFC Bank': 'HDFCBANK.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'State Bank of India': 'SBIN.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Bank of Baroda': 'BANKBARODA.NS'
}

# -------------------- 🎨 Streamlit UI Layout --------------------
st.set_page_config(layout="wide")
st.sidebar.title("📊 AI Banking Sector Stock Dashboard")
selected_stock = st.sidebar.selectbox("🔍 Select a Bank", list(companies.keys()))

# -------------------- 📥 Fetch 5 Years of Stock Data --------------------
def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, period="5y", interval="1d")
    if stock_data.empty:
        st.error(f"⚠️ Error: No data found for {ticker}.")
        return pd.DataFrame()

    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['Price_Change'] = stock_data['Close'].pct_change()
    
    # Buy/Sell Decision
    stock_data['Signal'] = np.where(stock_data['MA_20'] > stock_data['MA_50'], "BUY", "SELL")
    
    return stock_data.dropna()

# -------------------- 📡 Fetch Live Market Data --------------------
def fetch_live_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Current Price": f"{info.get('currentPrice', 0):.2f}",
        "Open": f"{info.get('open', 0):.2f}",
        "Close": f"{info.get('previousClose', 0):.2f}",
        "P/E Ratio": f"{info.get('trailingPE', 0):.2f}",
        "Volume": f"{info.get('volume', 0):,}",
        "EPS": f"{info.get('trailingEps', 0):.2f}"
    }

# -------------------- 📊 Data Processing --------------------
stock_data = fetch_stock_data(companies[selected_stock])
live_data = fetch_live_data(companies[selected_stock])

# -------------------- 📈 Stock Market Overview --------------------
if not stock_data.empty:
    st.markdown("## 📈 Stock Market Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("📌 Open Price", live_data["Open"])
    col2.metric("💰 Close Price", live_data["Close"])
    col3.metric("📊 Current Price", live_data["Current Price"])
    col4.metric("📉 P/E Ratio", live_data["P/E Ratio"])
    col5.metric("📊 Volume", live_data["Volume"])

    # -------------------- 🔥 Stock Price Trend Chart --------------------
    st.subheader(f"📈 Stock Price Trend: {selected_stock}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(stock_data.index, stock_data['Close'], label="Close Price", color='blue')
    ax.plot(stock_data.index, stock_data['MA_20'], label="20-Day MA", linestyle='dashed', color='orange')
    ax.plot(stock_data.index, stock_data['MA_50'], label="50-Day MA", linestyle='dashed', color='green')
    ax.legend()
    st.pyplot(fig)

    # -------------------- 🛒 Buy/Sell Decision Graph --------------------
    st.subheader("📊 Buy/Sell Decision")
    fig_bs, ax_bs = plt.subplots(figsize=(10, 4))
    ax_bs.scatter(stock_data.index, stock_data['Close'], c=(stock_data['Signal'] == 'BUY'), cmap='coolwarm', label='BUY')
    ax_bs.scatter(stock_data.index, stock_data['Close'], c=(stock_data['Signal'] == 'SELL'), cmap='coolwarm', label='SELL')
    ax_bs.legend()
    st.pyplot(fig_bs)

    # -------------------- 🔬 Statistical Analysis --------------------
    st.subheader("📊 Correlation Heatmap")
    corr = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # -------------------- 🔮 Future Price Prediction --------------------
    st.subheader("🚀 Future Price Predictions")

    def predict_future_price(data, days):
        if len(data) < 30:
            return np.nan
        
        try:
            data['Days'] = np.arange(len(data)).reshape(-1, 1)
            X = data['Days'].values.reshape(-1, 1)
            y = data['Close'].values

            model = LinearRegression()
            model.fit(X, y)

            future_day = np.array([[len(data) + days]])
            future_price = model.predict(future_day)
            return future_price[0] if len(future_price) > 0 else np.nan
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            return np.nan

    one_day_pred = predict_future_price(stock_data, 1)
    one_week_pred = predict_future_price(stock_data, 7)
    actual_close = stock_data['Close'].iloc[-1]
    
    def calculate_error(predicted, actual):
        if np.isnan(predicted) or np.isnan(actual):
            return np.nan
        return ((predicted - actual) / actual) * 100

    error_1day = calculate_error(one_day_pred, actual_close)
    error_1week = calculate_error(one_week_pred, actual_close)

    col6, col7, col8 = st.columns(3)
    col6.metric("1-Day Prediction", f"{one_day_pred:.2f}" if not np.isnan(one_day_pred) else "Unavailable", delta=f"{error_1day:.2f}%" if not np.isnan(error_1day) else "0.00%")
    col7.metric("1-Week Prediction", f"{one_week_pred:.2f}" if not np.isnan(one_week_pred) else "Unavailable", delta=f"{error_1week:.2f}%" if not np.isnan(error_1week) else "0.00%")
    col8.metric("Error Percentage", f"{error_1day:.2f}%" if not np.isnan(error_1day) else "0.00%")

    # -------------------- 📉 Stock Price Prediction Graph --------------------
    st.subheader("📉 Stock Price Prediction Graph")
    future_days = np.arange(len(stock_data), len(stock_data) + 10).reshape(-1, 1)
    future_prices = [predict_future_price(stock_data, i) for i in range(1, 11)]

    fig_pred, ax_pred = plt.subplots(figsize=(10, 4))
    ax_pred.plot(stock_data.index, stock_data['Close'], label="Actual Prices", color='blue')
    ax_pred.plot(pd.date_range(start=stock_data.index[-1], periods=10), future_prices, label="Predicted Prices", color='red', linestyle='dashed')
    ax_pred.legend()
    st.pyplot(fig_pred)

    st.success("🎯 Analysis Completed!")
