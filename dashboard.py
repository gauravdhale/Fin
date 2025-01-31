import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

def fetch_stock_data(ticker, period='5y', interval='1d'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df

def plot_stock_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
    return fig

def predict_stock_prices(df):
    df = df.reset_index()
    df['Days'] = np.arange(len(df))
    X = df[['Days']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(df) + 30).reshape(-1, 1)
    predictions = model.predict(future_days)
    return predictions[-30:], model.score(X, y)

def main():
    st.title('Banking Sector Financial Dashboard')
    
    banks = {
        'HDFC Bank': 'HDFCBANK.NS',
        'ICICI Bank': 'ICICIBANK.NS',
        'State Bank of India': 'SBIN.NS',
        'Kotak Mahindra Bank': 'KOTAKBANK.NS',
        'Axis Bank': 'AXISBANK.NS',
        'Bank of Baroda': 'BANKBARODA.NS'
    }
    
    selected_bank = st.selectbox('Select a Bank', list(banks.keys()))
    ticker = banks[selected_bank]
    
    with st.spinner('Fetching data...'):
        df = fetch_stock_data(ticker)
        st.plotly_chart(plot_stock_chart(df, f'{selected_bank} Stock Price'))
        
        pred, accuracy = predict_stock_prices(df)
        st.subheader('Stock Prediction (Next 30 Days)')
        st.line_chart(pred)
        st.write(f'Model Accuracy: {accuracy:.2%}')
    
    st.sidebar.title('Live Stock Data')
    live_data = {bank: yf.Ticker(banks[bank]).history(period='1d')['Close'].iloc[-1] for bank in banks}
    st.sidebar.write(pd.DataFrame(live_data.items(), columns=['Bank', 'Latest Price']))

if __name__ == '__main__':
    main()
