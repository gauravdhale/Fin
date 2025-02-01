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

# -------------------- 🔮 Stock Price Prediction --------------------
def predict_future(data, days=10):
    if len(data) < 30:  # Need enough data for predictions
        return pd.DataFrame(columns=["Date", "Predicted Price"])
    
    data['Days'] = np.arange(len(data)).reshape(-1, 1)
    X = data['Days'].values.reshape(-1, 1)
    y = data['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    future_dates = np.arange(len(data), len(data) + days).reshape(-1, 1)
    future_prices = model.predict(future_dates)

    return pd.DataFrame({"Date": pd.date_range(start=data.index[-1], periods=days, freq='D'), "Predicted Price": future_prices})

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
    buy_signals = stock_data[stock_data['Signal'] == "BUY"]
    sell_signals = stock_data[stock_data['Signal'] == "SELL"]
    ax_bs.scatter(buy_signals.index, buy_signals['Close'], color='green', label='BUY', marker='^')
    ax_bs.scatter(sell_signals.index, sell_signals['Close'], color='red', label='SELL', marker='v')
    ax_bs.legend()
    st.pyplot(fig_bs)

    # -------------------- 🔬 Statistical Analysis --------------------
    st.subheader("📊 Correlation Heatmap")
    corr = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # -------------------- 🔮 Future Price Prediction --------------------
    future_predictions = predict_future(stock_data)
    st.subheader(f"🚀 Future Price Predictions for {selected_stock}")
    st.dataframe(future_predictions)

    # Prediction Line Graph
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(future_predictions['Date'], future_predictions['Predicted Price'], label="Future Price", color='purple')
    ax3.legend()
    st.pyplot(fig3)

    # -------------------- 📉 Stock Price Prediction Graph --------------------
    st.subheader("📉 Stock Price Prediction Graph")
    future_days = np.arange(len(stock_data), len(stock_data) + 10).reshape(-1, 1)
    future_prices = predict_future(stock_data, 10)['Predicted Price']

    fig_pred, ax_pred = plt.subplots(figsize=(10, 4))
    ax_pred.plot(stock_data.index, stock_data['Close'], label="Actual Prices", color='blue')
    ax_pred.plot(pd.date_range(start=stock_data.index[-1], periods=10), future_prices, label="Predicted Prices", color='red', linestyle='dashed')
    ax_pred.legend()
    st.pyplot(fig_pred)

    st.success("🎯 Analysis Completed!")
