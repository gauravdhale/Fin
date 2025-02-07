import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# 🔹 GitHub Repository Info
GITHUB_REPO = "gauravdhale/Fin"
BRANCH = "main"

# 🔹 Function to Fetch CSV File List from GitHub
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

# 🔹 UI: Dropdown to Select CSV File
if csv_files:
    selected_file = st.sidebar.selectbox("📂 Select a Bank Stock", csv_files)
else:
    st.error("No CSV files found in GitHub repository.")
    st.stop()

# 🔹 Function to Read CSV File from GitHub
@st.cache_data
def load_data(file_name):
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}/{file_name}"
    try:
        df = pd.read_csv(url)
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)  # Ensure Date format is correct
        df.set_index("Date", inplace=True)
        return df
    except Exception as e:
        st.error(f"Error reading {file_name}: {e}")
        return pd.DataFrame()

# 🔹 Load Selected Data
data = load_data(selected_file)

# 🔹 UI: Plot Actual vs Predicted Prices
st.header(f"📊 {selected_file.split('.')[0]} - Actual vs Predicted Prices")

def plot_actual_vs_predicted(df, company_name):
    if df.empty:
        st.warning("No data available.")
        return

    # 🔹 Calculate Error Percentage for January 24, 2025
    specific_date = pd.Timestamp("2025-01-24")
    if specific_date in df.index:
        actual_price = df.loc[specific_date, "Actual Price"]
        predicted_price = df.loc[specific_date, "Predicted Price"]
        error_percentage = abs((actual_price - predicted_price) / actual_price) * 100
        error_text = f"❗ Error percentage on January 24, 2025: **{error_percentage:.2f}%**"
    else:
        error_text = "⚠️ No data for January 24, 2025"

    # 🔹 Create Plotly Figure
    fig = go.Figure()

    # ✅ Actual Price Line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Actual Price"],
        mode="lines", name="Actual Price",
        line=dict(color="blue")
    ))

    # ✅ Predicted Price Line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Predicted Price"],
        mode="lines", name="Predicted Price",
        line=dict(color="red", dash="dash")
    ))

    # 🔹 Update Chart Layout
    fig.update_layout(
        title=f"📈 {company_name} - Actual vs Predicted Prices",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        hovermode="x unified"
    )

    # 🔹 Show Chart & Error Percentage
    st.plotly_chart(fig)
    st.markdown(error_text)

# 🔹 Call Function to Plot Data
plot_actual_vs_predicted(data, selected_file.split('.')[0])
