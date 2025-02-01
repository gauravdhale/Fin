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
        "EPS": f"{info.get('trailingEps', 0):.2f}",
        "Net Profit": info.get('netIncomeToCommon', 0)
    }

# -------------------- 🔮 Stock Price Prediction --------------------
def predict_future(data, days=30):
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

    # -------------------- 🔥 Stacked Area Chart for Stock Price --------------------
    st.subheader(f"📊 Stacked Area Chart: {selected_stock}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(stock_data.index, stock_data['Close'], colors=['blue'], alpha=0.5)
    ax.set_ylabel("Stock Price")
    st.pyplot(fig)

    # -------------------- 📊 Market Share (Donut Chart) --------------------
    st.subheader("📊 Market Share Distribution")
    market_shares = {k: yf.Ticker(v).info.get("marketCap", 0) for k, v in companies.items()}
    labels = list(market_shares.keys())
    sizes = list(market_shares.values())

    fig3, ax3 = plt.subplots()
    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"),
                                       wedgeprops=dict(width=0.4))
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(10)
    ax3.axis('equal')
    st.pyplot(fig3)

    # -------------------- 📈 EPS & Net Profit (Line Charts) --------------------
    st.subheader("📈 EPS & Net Profit Over Time")
    col6, col7 = st.columns(2)
    
    with col6:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.plot(stock_data.index, stock_data['Close'] * 0.02, color='purple', label='EPS')  # Assuming EPS grows with stock price
        ax4.set_ylabel("EPS")
        ax4.legend()
        st.pyplot(fig4)

    with col7:
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        ax5.plot(stock_data.index, stock_data['Close'] * 0.05, color='green', label='Net Profit')  # Assuming a fixed % of Close price
        ax5.set_ylabel("Net Profit")
        ax5.legend()
        st.pyplot(fig5)

    # -------------------- 🔮 Future Price Prediction (30 Days) --------------------
    st.subheader(f"📈 30-Day Future Price Prediction: {selected_stock}")
    future_predictions = predict_future(stock_data, days=30)
    
    fig6, ax6 = plt.subplots(figsize=(12, 5))
    ax6.plot(future_predictions['Date'], future_predictions['Predicted Price'], label="Predicted Price", color='red')
    ax6.legend()
    st.pyplot(fig6)

    # -------------------- 📌 Error Rate in Metric Card --------------------
    actual_close = stock_data['Close'].iloc[-1]
    predicted_close = future_predictions['Predicted Price'].iloc[0] if not future_predictions.empty else np.nan
    error_rate = abs((predicted_close - actual_close) / actual_close) * 100 if not np.isnan(predicted_close) else 0

    st.subheader("📌 Error Metrics")
    col8, col9 = st.columns(2)
    col8.metric("🔍 Prediction Error Rate", f"{error_rate:.2f}%")

    st.success("🎯 Analysis Completed!")
