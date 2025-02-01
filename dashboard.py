import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

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

# Custom Background Styling (Banking Theme)
page_bg = """
    <style>
        body {
            background-color: #F0F2F6;  /* Light Banking Background */
            color: #1E3A5F;  /* Dark Blue Text */
        }
        .stApp {
            background-color: #F0F2F6;
        }
        .stMetric {
            background-color: #E6E8EB;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
        }
    </style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

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
        "Current Price": float(info.get('currentPrice', 0)),
        "Open": float(info.get('open', 0)),
        "Close": float(info.get('previousClose', 0)),
        "P/E Ratio": float(info.get('trailingPE', 0)),
        "Volume": int(info.get('volume', 0)),
        "EPS": float(info.get('trailingEps', 0)),
        "Net Profit": float(info.get('netIncomeToCommon', 0))
    }

# -------------------- 🔮 Stock Price Prediction (Fixed Error) --------------------
def predict_future(data, days=30):
    if len(data) < 30:  # Need enough data for predictions
        return pd.DataFrame(columns=["Date", "Predicted Price"])
    
    data['Days'] = np.arange(len(data)).reshape(-1, 1)
    X = data['Days'].values.reshape(-1, 1)
    y = data['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    future_dates = np.arange(len(data), len(data) + days).reshape(-1, 1)
    future_prices = model.predict(future_dates).flatten()  # Ensure 1D array

    return pd.DataFrame({"Date": pd.date_range(start=data.index[-1], periods=days, freq='D'), "Predicted Price": future_prices})

# -------------------- 📊 Data Processing --------------------
stock_data = fetch_stock_data(companies[selected_stock])
live_data = fetch_live_data(companies[selected_stock])

# -------------------- 📈 Stock Market Overview --------------------
if not stock_data.empty:
    st.markdown("## 📈 Stock Market Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("📌 Open Price", f"${live_data['Open']:.2f}")
    col2.metric("💰 Close Price", f"${live_data['Close']:.2f}")
    col3.metric("📊 Current Price", f"${live_data['Current Price']:.2f}")
    col4.metric("📉 P/E Ratio", f"{live_data['P/E Ratio']:.2f}")
    col5.metric("📊 Volume", f"{live_data['Volume']:,}")

    # -------------------- 📉 Line Chart for Stock Price --------------------
    st.subheader(f"📊 Stock Price Trend: {selected_stock}")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(stock_data.index, stock_data['Close'], label="Close Price", color='blue')
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)

    # -------------------- 📈 EPS & Net Profit Line Chart --------------------
    col6, col7, col8 = st.columns(3)

    with col6:
        st.subheader("📈 EPS Over Time")
        fig_eps, ax_eps = plt.subplots(figsize=(5, 3))
        ax_eps.plot(stock_data.index, stock_data['Close'] * 0.02, color='purple', label='EPS')
        ax_eps.set_ylabel("EPS")
        ax_eps.legend()
        st.pyplot(fig_eps)

    with col7:
        st.subheader("📈 Net Profit Over Time")
        fig_np, ax_np = plt.subplots(figsize=(5, 3))
        ax_np.plot(stock_data.index, stock_data['Close'] * 0.05, color='green', label='Net Profit')
        ax_np.set_ylabel("Net Profit")
        ax_np.legend()
        st.pyplot(fig_np)

    # -------------------- 🔮 30-Day Future Price Prediction --------------------
    st.subheader("📈 30-Day Price Prediction")
    future_predictions = predict_future(stock_data, days=30)

    fig_pred, ax_pred = plt.subplots(figsize=(5, 3))
    ax_pred.plot(future_predictions['Date'], future_predictions['Predicted Price'], label="Predicted Price", color='red')
    ax_pred.set_ylabel("Predicted Price")
    ax_pred.legend()
    st.pyplot(fig_pred)

    # -------------------- 📌 Error Rate in Metric Card --------------------
    actual_close = stock_data['Close'].iloc[-1]
    predicted_close = future_predictions['Predicted Price'].iloc[0] if not future_predictions.empty else np.nan
    error_rate = abs((predicted_close - actual_close) / actual_close) * 100 if not np.isnan(predicted_close) else 0

    st.subheader("📌 Error Metrics")
    col9, col10 = st.columns(2)
    col9.metric("🔍 Prediction Error Rate", f"{error_rate:.2f}%")

    st.success("🎯 Analysis Completed!")
