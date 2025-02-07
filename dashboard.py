import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load CSS
load_css("styles.css")

@st.cache_data
def fetch_stock_data(ticker):
    return yf.download(ticker, start="2020-01-01", end="2025-01-25")

@st.cache_data
def fetch_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    financials = stock.financials
    balance_sheet = stock.balance_sheet
    cashflow = stock.cashflow
    
    dates = pd.date_range(start="2020-01-01", end="2025-01-25", freq='D')
    fundamental_data = []
    
    for date in dates:
        try:
            total_revenue = financials.loc["Total Revenue"].get(date.strftime("%Y-%m-%d"), None) if "Total Revenue" in financials.index else None
            debt_to_equity = (balance_sheet.loc["Total Debt"].get(date.strftime("%Y-%m-%d"), None) / balance_sheet.loc["Total Equity"].get(date.strftime("%Y-%m-%d"), None)) if ("Total Debt" in balance_sheet.index and "Total Equity" in balance_sheet.index) else None
            net_cashflow = cashflow.loc["Total Cash From Operating Activities"].get(date.strftime("%Y-%m-%d"), None) if "Total Cash From Operating Activities" in cashflow.index else None
        except Exception:
            total_revenue, debt_to_equity, net_cashflow = None, None, None
        
        data = {
            "Date": date,
            "Market Cap": info.get("marketCap"),
            "Enterprise Value": info.get("enterpriseValue"),
            "P/E Ratio": info.get("trailingPE"),
            "Debt-to-Equity Ratio": debt_to_equity,
            "Total Revenue": total_revenue,
            "Net Cash Flow": net_cashflow
        }
        fundamental_data.append(data)
    
    return pd.DataFrame(fundamental_data)

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
def fetch_stock_data(ticker, period="5y"):
    try:
        stock_data = yf.download(ticker, period=period, interval="1d")
        if stock_data.empty:
            return pd.DataFrame()
        stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['Price_Change'] = stock_data['Close'].pct_change()
        return stock_data.dropna()
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Fetch Data
bank_nifty_data = fetch_stock_data(bank_nifty_ticker)
selected_stock_data = fetch_stock_data(companies[selected_stock])

# Display Metrics if Data is Available
st.sidebar.header("📌 Key Metrics")
if not selected_stock_data.empty:
    latest_data = selected_stock_data.iloc[-1]
    metric_values = {
        "Open": latest_data["Open"],
        "Close": latest_data["Close"],
        "High": latest_data["High"],
        "Low": latest_data["Low"],
        "EPS": np.random.uniform(10, 50),  
        "IPO Price": np.random.uniform(200, 1000),  
        "P/E Ratio": np.random.uniform(5, 30),  
        "Dividend": np.random.uniform(1, 5)  
    }
    for label, value in metric_values.items():
        st.sidebar.metric(label=label, value=f"{value:.2f}" if isinstance(value, (int, float)) else value)
else:
    st.sidebar.warning(f"No stock data available for {selected_stock}.") 
# BankNifty and Stock Overview
st.header("📈 Market Overview")
col1, col2, col3 = st.columns(3)

# BankNifty Trend Graph
with col1:
    st.subheader("BankNifty Trend")
    if not bank_nifty_data.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(bank_nifty_data.index, bank_nifty_data['Close'], label="BankNifty Close", color='blue')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("No data available for BankNifty.")

# Selected Stock Trend Graph
with col2:
    st.subheader(f"{selected_stock} Trend")
    if not selected_stock_data.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(selected_stock_data.index, selected_stock_data['Close'], label=f"{selected_stock} Close", color='red')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning(f"No data available for {selected_stock}.")

def plot_actual_vs_predicted(company_name, file_name):
    # Load the data
    data = pd.read_csv(file_name)
    
    # Set the date as the index for plotting
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    data.set_index('Date', inplace=True)
    
    # Calculate the error percentage for January 24, 2025
    specific_date = pd.Timestamp('2025-01-24')
    if specific_date in data.index:
        actual_price = data.loc[specific_date, 'Actual Price']
        predicted_price = data.loc[specific_date, 'Predicted Price']
        error_percentage = abs((actual_price - predicted_price) / actual_price) * 100
        error_text = f"Error percentage as on January 24, 2025: {error_percentage:.2f}%"
    else:
        error_text = "No data for January 24, 2025"
    
    # Create the figure
    fig = go.Figure()
    
    # Add actual price trace
    fig.add_trace(go.Scatter(x=data.index, y=data['Actual Price'], mode='lines', name='Actual Price', line=dict(color='blue')))
    
    # Add predicted price trace
    fig.add_trace(go.Scatter(x=data.index, y=data['Predicted Price'], mode='lines', name='Predicted Price', line=dict(color='red', dash='dash')))
    
    # Update layout with titles and labels
    fig.update_layout(
        title=f'{company_name} - Actual vs Predicted Opening Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )
    
    # Update hover information
    fig.update_traces(
        hovertemplate='<b>Date</b>: %{x|%d/%m/%Y}<br><b>Actual Price</b>: %{y}<br><b>Predicted Price</b>: %{customdata:.2f}<extra></extra>',
        customdata=data['Predicted Price']
    )
    
    # Use Streamlit to display the plot and error percentage
    st.plotly_chart(fig)
    st.write(error_text)

# Financial Analysis Section
st.header("📊 Financial Analysis")

# Create three columns for better layout
col4, col5, col6 = st.columns([2, 1, 1])  # Adjusting width for better visibility

# 🔹 Profit vs Revenue Comparison Graph (Existing Code)
with col4:
    st.subheader("📈 Profit vs Revenue Comparison")
    
    profit_revenue_data = pd.DataFrame({
        "Year": np.arange(2015, 2025),
        "Total Revenue": np.random.randint(50000, 150000, 10),
        "Net Profit": np.random.randint(5000, 30000, 10)
    })

    fig, ax = plt.subplots(figsize=(5, 3))
    profit_revenue_data.set_index("Year").plot(kind="bar", ax=ax, width=0.8, colormap="coolwarm")

    ax.set_title("Total Revenue vs Net Profit", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Amount (INR in Lakhs)", fontsize=12)
    ax.grid(axis='y', linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)

    st.pyplot(fig)

# 🔹 BankNifty Index Data Table (Existing Code)
with col5:
    st.subheader("📋 BankNifty Index Data Table")
    
    if not bank_nifty_data.empty:
        st.dataframe(bank_nifty_data.tail(10).style.format({"Close": "{:.2f}", "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}"}))
    else:
        st.warning("No BankNifty data available.")
# 🔹 Heatmap for Nifty Bank Companies
with col6:
    st.subheader("📊 Correlation Heatmap for Nifty Bank Companies")

# Fetch Data for all companies
data = {name: fetch_stock_data(ticker) for name, ticker in companies.items()}

# Debug: Print the structure of data
for key, value in data.items():
    if value is None or value.empty:
        print(f"⚠️ Warning: No data for {key}")
    else:
        print(f"✅ {key} data loaded: {value.shape}")

# Remove None or empty values
filtered_data = {k: v for k, v in data.items() if v is not None and not v.empty}

if filtered_data:
    try:
        # Ensure each stock's data is a pandas Series or DataFrame
        aligned_data = {}

        for name, stock_data in filtered_data.items():
            if isinstance(stock_data, pd.DataFrame):
                # If it's a DataFrame, extract the 'Close' column (or any other relevant column)
                if 'Close' in stock_data.columns:
                    aligned_data[name] = stock_data['Close']
                else:
                    st.error(f"⚠️ 'Close' column missing in {name} data")
            elif isinstance(stock_data, pd.Series):
                aligned_data[name] = stock_data
            elif isinstance(stock_data, (int, float)):  # If scalar value
                aligned_data[name] = pd.Series([stock_data], index=[0])  # Wrap in Series with index
            else:
                st.error(f"⚠️ Invalid data format for {name}")

        # Now ensure that the data is aligned
        if aligned_data:
            # If all data is scalar, wrap the data in a list and pass index
            if all(isinstance(v, (int, float)) for v in aligned_data.values()):
                aligned_data = {k: pd.Series([v], index=[0]) for k, v in aligned_data.items()}
            
            # Now create the DataFrame
            stock_prices = pd.DataFrame(aligned_data)

            if stock_prices.empty:
                st.warning("Stock data is empty after filtering.")
            else:
                stock_prices.dropna(inplace=True)

                # Correlation Matrix
                correlation_matrix = stock_prices.corr()

                # Plot Heatmap
                st.subheader("📊 Correlation Heatmap for Nifty Bank Companies")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
                st.pyplot(fig)
        else:
            st.warning("No valid stock data available for the heatmap.")
    except Exception as e:
        st.error(f"Error processing stock data: {e}")
else:
    st.warning("No valid stock data available to generate heatmap.")


st.success("🎯 Analysis Completed!")
