import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Title of the Dashboard
st.title("📈 Banking Sector Financial Dashboard")

# Define stock symbols for tracking
bank_stocks = {
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bank of Baroda": "BANKBARODA.NS"
}

# Sidebar for stock selection
st.sidebar.header("Select Stock")
selected_stock = st.sidebar.selectbox("Choose a bank stock:", list(bank_stocks.keys()))
stock_symbol = bank_stocks[selected_stock]

# Fetch historical data
@st.cache_data
def get_stock_data(ticker, period="5y"):
    return yf.download(ticker, period=period)

data = get_stock_data(stock_symbol)

# Display Stock Data
st.subheader(f"Stock Data for {selected_stock} ({stock_symbol})")
st.dataframe(data.tail())

# Plot stock closing price
st.subheader("Closing Price Trend")
fig, ax = plt.subplots()
ax.plot(data.index, data["Close"], label="Closing Price", color='blue')
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.set_title(f"Closing Price of {selected_stock}")
ax.legend()
st.pyplot(fig)

# Stock Price Prediction using Linear Regression
st.subheader("Stock Price Prediction")

# Prepare data for prediction
past_days = 365 * 5  # 5 years of daily data
data.reset_index(inplace=True)
data = data.dropna()
data["Days"] = (data["Date"] - data["Date"].min()).dt.days

X = np.array(data["Days"]).reshape(-1, 1)
y = np.array(data["Close"]).reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)

# Predict future stock prices
future_days = np.array([X[-1] + i for i in range(1, 31)]).reshape(-1, 1)
predicted_prices = model.predict(future_days)

# Plot prediction
fig2, ax2 = plt.subplots()
ax2.plot(data["Date"], data["Close"], label="Actual Price", color='blue')
future_dates = [data["Date"].max() + timedelta(days=i) for i in range(1, 31)]
ax2.plot(future_dates, predicted_prices, label="Predicted Price", linestyle="dashed", color='red')
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (INR)")
ax2.set_title(f"Stock Price Prediction for {selected_stock}")
ax2.legend()
st.pyplot(fig2)

# Prediction Accuracy
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")

# Heatmap of stock correlations
st.subheader("Stock Correlation Heatmap")
all_data = {stock: get_stock_data(ticker)['Close'] for stock, ticker in bank_stocks.items()}
all_df = pd.DataFrame(all_data)
correlation_matrix = all_df.corr()
fig3, ax3 = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# Sector Financial Metrics
st.subheader("Key Financial Metrics")
def get_financial_metrics(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    metrics = {
        "EPS": info.get("trailingEps", "N/A"),
        "PE Ratio": info.get("trailingPE", "N/A"),
        "IPO Price": info.get("regularMarketPreviousClose", "N/A")
    }
    return metrics

metrics = get_financial_metrics(stock_symbol)
st.write(metrics)

st.write("🔍 **This dashboard provides insights into stock trends, future predictions, and sector-wide correlations.**")
