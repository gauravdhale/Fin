import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# 🔹 Correct Base URL (Modify Your GitHub Username, Repo, and Branch)
GITHUB_REPO = "gauravdhale/Fin"
BRANCH = "main"

# 🔹 Function to get the list of CSV files from GitHub
@st.cache_data
def get_csv_files():
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents"
    response = requests.get(api_url)
    if response.status_code == 200:
        files = [file["name"] for file in response.json() if file["name"].endswith(".csv")]
        return files
    else:
        st.error("Error fetching file list from GitHub")
        return []

# 🔹 Get List of CSV Files
csv_files = get_csv_files()

# 🔹 Dropdown to Select CSV File
if csv_files:
    selected_file = st.sidebar.selectbox("Select a Bank Stock", csv_files)
else:
    st.error("No CSV files found in GitHub repository.")
    st.stop()

# 🔹 Function to Read CSV File from GitHub
@st.cache_data
def load_data(file_name):
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}/{file_name}"
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
    ax.plot(data["Date"], data["Predicted_Open"], label="Predicted Open", color="green", linestyle="-")

    ax.set_title(f"{selected_file.split('.')[0]} - Open Price Prediction", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (INR)", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    st.pyplot(fig)
else:
    st.warning(f"No data available for {selected_file}.")
