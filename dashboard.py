import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

# 🔹 GitHub Repository Details
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

# Function to Read CSV File from GitHub
@st.cache_data
def load_data(file_name):
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}/{file_name}"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        df.rename(columns={"Open": "Actual Price", "Predicted_Open": "Predicted Price"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", dayfirst=True, errors="coerce")
        df.set_index("Date", inplace=True)
        return df
    except Exception as e:
        st.error(f"Error reading {file_name}: {e}")
        return pd.DataFrame()

# 🔹 Load Selected Data
data = load_data(selected_file)

# 🔹 Function to Plot Actual vs Predicted Prices
def plot_actual_vs_predicted(data, company_name):
    if data.empty:
        st.warning(f"No data available for {company_name}.")
        return

    # ✅ Check for Required Columns
    required_columns = ["Actual Price", "Predicted Price"]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error(f"⚠ Missing columns in CSV: {missing_columns}")
        return
    
    # 🔹 Find Error Percentage on Specific Date
    specific_date = pd.Timestamp("2025-01-24")
    if specific_date in data.index:
        actual_price = data.loc[specific_date, "Actual Price"]
        predicted_price = data.loc[specific_date, "Predicted Price"]
        error_percentage = abs((actual_price - predicted_price) / actual_price) * 100
        error_text = f"📉 **Error Percentage on Jan 24, 2025:** {error_percentage:.2f}%"
    else:
        error_text = "⚠ No data for January 24, 2025"

    # 🔹 Create Plotly Figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data["Actual Price"], 
        mode="lines", name="Actual Price", 
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=data.index, y=data["Predicted Price"], 
        mode="lines", name="Predicted Price", 
        line=dict(color="red", dash="dash")
    ))

    # 🔹 Update Layout
    fig.update_layout(
        title=f"{company_name} - Actual vs Predicted Opening Prices",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )

    # 🔹 Display in Streamlit
    st.plotly_chart(fig)
    st.write(error_text)

# 🔹 Plot Data
st.header(f"📈 Prediction vs Actual - {selected_file.split('.')[0]}")
plot_actual_vs_predicted(data, selected_file.split('.')[0])
