import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 🔹 Correct Base URL (Modify Your GitHub Username, Repo, and Branch)
BASE_URL = "https://raw.githubusercontent.com/gauravdhale/Fin/main/"

# 🔹 List of CSV files (Ensure these exist in your GitHub repository)
csv_files = [
    "HDFC.csv",
    "ICICI.csv",
    "SBI.csv",
    "KOTAK.csv",
    "AXISBANK.NS_predicted_data.csv",  # Ensure this is the correct filename
    "BOB.csv"
]

# 🔹 Dropdown to Select CSV File
selected_file = st.sidebar.selectbox("Select a Bank Stock", csv_files)

# 🔹 Function to Read CSV File from GitHub
@st.cache_data
def load_data(file_name):
    url = BASE_URL + file_name  # Dynamically append filename to the base URL
    try:
        df = pd.read_csv(url, parse_dates=["Date"])
        return df
    except Exception as e:
        st.error(f"Error reading {file_name}: {e}")
        return pd.DataFrame()

# 🔹 Load Selected Data
data = load_data(selected_file)

# 🔹 Plot Predictions
st.header(f"📈 Prediction vs Actual - {selected_file.split('.')[0]}")

if not data.empty:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(data["Date"], data["Open"], label="Actual Open", color="blue", linestyle="-")
    ax.plot(data["Date"], data["Predicted_Open"], label="Predicted Open", color="red", linestyle="dashed", marker="o")

    ax.set_title(f"{selected_file.split('.')[0]} - Open Price Prediction", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (INR)", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    st.pyplot(fig)
else:
    st.warning(f"No data available for {selected_file}.")
