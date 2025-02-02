import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score

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

# Streamlit Sidebar
st.set_page_config(layout="wide")
st.sidebar.title("📊 AI Banking Sector & BankNifty Dashboard")
selected_stock = st.sidebar.selectbox("🔍 Select a Bank", list(companies.keys()))

def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, period="10y", interval="1d")
    if stock_data.empty:
        st.error(f"⚠️ Error: No data found for {ticker}.")
        return pd.DataFrame()
    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['Price_Change'] = stock_data['Close'].pct_change()
    return stock_data.dropna()

def fetch_live_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Current Price": f"{info.get('currentPrice', 0):.4f}",
        "Open": f"{info.get('open', 0):.4f}",
        "Close": f"{info.get('previousClose', 0):.4f}",
        "P/E Ratio": f"{info.get('trailingPE', 0):.4f}",
        "EPS": f"{info.get('trailingEps', 0):.4f}",
        "Volume": f"{info.get('volume', 0):,.4f}",
        "IPO Price": f"{info.get('regularMarketPreviousClose', 0):.4f}",
        "Dividend": f"{info.get('dividendYield', 0):.4f}"
    }

# Fetch Data
stock_data = fetch_stock_data(companies[selected_stock])
live_data = fetch_live_data(companies[selected_stock])
bank_nifty_data = fetch_stock_data(bank_nifty_ticker)

if not stock_data.empty:
    st.markdown("## 📈 BankNifty & Stock Market Overview")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("📌 Open Price", live_data["Open"])
    col2.metric("💰 Close Price", live_data["Close"])
    col3.metric("📊 Current Price", live_data["Current Price"])
    col4.metric("📉 P/E Ratio", live_data["P/E Ratio"])
    col5.metric("📊 EPS", live_data["EPS"])
    col6.metric("📊 Volume", live_data["Volume"])
    col7.metric("🚀 Dividend", live_data["Dividend"])
    
    st.subheader(f"📈 BankNifty Trend & Prediction")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bank_nifty_data.index, bank_nifty_data['Close'], label="BankNifty Close", color='blue')
    ax.legend()
    st.pyplot(fig)
    
    def train_model(data):
        X = data[['Open', 'High', 'Low', 'MA_20', 'MA_50', 'Price_Change']]
        y = data['Close']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        models = {
            "Linear Regression": LinearRegression(),
            "SVR": SVR(kernel='rbf', C=10, epsilon=0.1),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10)
        }
        for model in models.values():
            model.fit(X_train, y_train)
        voting_model = VotingRegressor([(name, model) for name, model in models.items()])
        voting_model.fit(X_train, y_train)
        y_pred = voting_model.predict(X_test)
        return y_test, y_pred
    
    y_test, y_pred = train_model(stock_data)
    
    st.subheader(f"📊 Heatmap: Contribution of Stocks to BankNifty")
    stock_contributions = {stock: np.random.rand() for stock in companies.keys()}  # Mock data
    heatmap_data = pd.DataFrame(stock_contributions, index=["Impact"])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", linewidths=0.5)
    st.pyplot(fig)
    
    st.success("🎯 Analysis Completed!")
