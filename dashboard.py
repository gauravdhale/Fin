import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.sidebar.title("📊 AI Banking Stock Dashboard")
selected_stock = st.sidebar.selectbox("🔍 Select a Bank", list(companies.keys()))

# -------------------- 📥 Fetch Stock Data --------------------
@st.cache_data
def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, period="5y", interval="1d")
    if stock_data.empty:
        st.error(f"⚠️ No data found for {ticker}.")
        return pd.DataFrame()
    return stock_data.dropna()

# -------------------- 📡 Fetch Live Market Data --------------------
@st.cache_data
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

# -------------------- 🔮 Stock Price Prediction --------------------
def predict_future(data, days=30):
    if len(data) < 30:
        return pd.DataFrame(columns=["Date", "Predicted Price"])
    
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(data), len(data) + days).reshape(-1, 1)
    future_prices = model.predict(future_X).flatten()  

    return pd.DataFrame({"Date": pd.date_range(start=data.index[-1], periods=days, freq='D'), "Predicted Price": future_prices})

# -------------------- 📊 Data Processing --------------------
stock_data = fetch_stock_data(companies[selected_stock])
live_data = fetch_live_data(companies[selected_stock])

# -------------------- 📅 Date Selection (Dynamically Update Charts) --------------------
st.sidebar.subheader("📅 Select Date Range")
min_date, max_date = stock_data.index.min().date(), stock_data.index.max().date()
start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

stock_data = stock_data.loc[(stock_data.index >= pd.to_datetime(start_date)) & (stock_data.index <= pd.to_datetime(end_date))]

# -------------------- 📈 Stock Market Overview --------------------
st.markdown("## 📈 Stock Market Overview")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📌 Open Price", f"${live_data['Open']:.2f}")
col2.metric("💰 Close Price", f"${live_data['Close']:.2f}")
col3.metric("📊 Current Price", f"${live_data['Current Price']:.2f}")
col4.metric("📉 P/E Ratio", f"{live_data['P/E Ratio']:.2f}")
col5.metric("📊 Volume", f"{live_data['Volume']:,}")

# -------------------- 📉 Stock Price Chart --------------------
st.subheader(f"📊 Stock Price Trend: {selected_stock}")
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(stock_data.index, stock_data['Open'], label="Open Price", color='green')
ax.plot(stock_data.index, stock_data['Close'], label="Close Price", color='blue')
ax.axhline(y=live_data['Current Price'], color='red', linestyle='--', label="Current Price")
ax.legend()
st.pyplot(fig)

# -------------------- 🔮 30-Day Future Price Prediction --------------------
st.subheader("📈 30-Day Price Prediction")
future_predictions = predict_future(stock_data, days=30)

fig_pred, ax_pred = plt.subplots(figsize=(6, 3))
ax_pred.plot(future_predictions['Date'], future_predictions['Predicted Price'], label="Predicted Price", color='red')
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
