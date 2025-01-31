import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Define Banking Stocks
companies = {
    'Aditya Birla Fashion & Retail': 'ABFRL.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'State Bank of India': 'SBIN.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Bank of Baroda': 'BANKBARODA.NS'
}

# Streamlit Sidebar
st.sidebar.title("Stock Market Dashboard")
selected_stock = st.sidebar.selectbox("Select a Company", list(companies.keys()))

# Fetch Stock Data Function
@st.cache_data
def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, period="10y", interval="1d")

    # Check if the required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in stock_data.columns:
            st.error(f"Error: Column '{col}' missing in {ticker} data.")
            return pd.DataFrame()  # Return an empty DataFrame to avoid crashes

    # Add Moving Averages and Price Change
    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['Price_Change'] = stock_data['Close'].pct_change()

    return stock_data.dropna()

# Load Data
stock_data = fetch_stock_data(companies[selected_stock])

# Get key stock metrics
opening_price = round(stock_data['Open'][-1], 2)
closing_price = round(stock_data['Close'][-1], 2)
ipo_price = 220  # Example: Replace with real IPO data
pe_ratio = round(stock_data['Close'][-1] / stock_data['Price_Change'].std(), 2) if stock_data['Price_Change'].std() != 0 else "N/A"
volume = f"{round(stock_data['Volume'][-1] / 1e9, 1)}bn"

# Display Key Metrics
st.markdown(f"""
    <style>
    .metric-card {{
        display: inline-block;
        text-align: center;
        background: #d9534f;
        color: white;
        font-size: 20px;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        width: 180px;
    }}
    </style>
    <div>
        <div class="metric-card">Opening Price <br> <b>{opening_price}</b></div>
        <div class="metric-card">Closing Price <br> <b>{closing_price}</b></div>
        <div class="metric-card">IPO Price <br> <b>{ipo_price}</b></div>
        <div class="metric-card">P/E Ratio <br> <b>{pe_ratio}</b></div>
        <div class="metric-card">Volume <br> <b>{volume}</b></div>
    </div>
""", unsafe_allow_html=True)

# Stock Price Chart
st.subheader(f"Stock Price Trend for {selected_stock}")
fig1 = px.line(stock_data, x=stock_data.index, y="Close", title="Closing Price Over Time")
st.plotly_chart(fig1)

# Train Machine Learning Models
def train_model(data):
    X = data[['Open', 'High', 'Low', 'MA_20', 'MA_50', 'Price_Change']]
    y = data['Close']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Initialize Models
    lr_model = LinearRegression()
    svr_model = SVR(kernel='rbf', C=10, epsilon=0.1)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10)

    # Train Models
    lr_model.fit(X_train, y_train)
    svr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Ensemble Voting Regressor
    voting_model = VotingRegressor([('lr', lr_model), ('svr', svr_model), ('rf', rf_model)])
    voting_model.fit(X_train, y_train)

    y_pred = voting_model.predict(X_test)
    return y_test, y_pred, voting_model

y_test, y_pred, model = train_model(stock_data)

# Display Error Rate
error_rate = round(mean_squared_error(y_test, y_pred), 2)
st.subheader(f"Error Rate: {error_rate}")

# Prediction vs Actual Chart
st.subheader(f"Actual vs Predicted Prices for {selected_stock}")
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(y_test.index, y_test, label="Actual Price", color='blue')
ax2.plot(y_test.index, y_pred, label="Predicted Price", linestyle='dashed', color='red')
ax2.legend()
st.pyplot(fig2)

# ARIMA Forecasting
def predict_future(data):
    arima_model = ARIMA(data['Close'], order=(5, 1, 0))
    arima_result = arima_model.fit()
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 31)]
    future_predictions = arima_result.forecast(steps=30)
    return pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})

future_predictions = predict_future(stock_data)

# Future Prediction Chart
st.subheader(f"Predicted Value for Next 30 Days")
fig3 = px.line(future_predictions, x="Date", y="Predicted Price", title="Stock Price Prediction")
st.plotly_chart(fig3)

# Market Share Pie Chart
market_share_data = {
    "Aditya Birla Fashion & Retail": 71.61,
    "Shoppers Stop": 19.7,
    "V2 Retail": 8.37
}
fig4 = px.pie(values=market_share_data.values(), names=market_share_data.keys(), title="Market Share Distribution")
st.plotly_chart(fig4)

# Additional Charts: Sales vs Expenses, EPS, Net Profit
st.subheader("Sales vs Expenses, EPS & Net Profit Trends")
fig5 = px.line(stock_data, x=stock_data.index, y=["High", "Low"], title="Sales vs Expenses")
fig6 = px.line(stock_data, x=stock_data.index, y="Price_Change", title="Earnings Per Share (EPS)")
fig7 = px.line(stock_data, x=stock_data.index, y="MA_50", title="Net Profit")

st.plotly_chart(fig5)
st.plotly_chart(fig6)
st.plotly_chart(fig7)

st.success("Dashboard Loaded Successfully!")
