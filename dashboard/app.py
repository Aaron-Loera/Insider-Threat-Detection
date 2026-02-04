import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Insider Threat Dashboard",
    layout="wide"
)

st.title("🛡️ Insider Threat Detection Dashboard")

st.info(
    "Dashboard structure initialized.\n\n"
    "Waiting for final data source definition from the anomaly detection pipeline."
)

st.markdown(
    """
    ### Planned Features
    - User activity timeline
    - Anomaly alerts table
    - Severity-based filtering
    - Support for ensemble models (Isolation Forest + Autoencoder)
    """
)

# Load data
data = pd.read_csv(r"C:\Users\loera\Documents\Insider-Threat-Detection\static_dashboards\table_1.csv") # change this to your specific path

# Sidebar filter
st.sidebar.header("Filter")
select_users = st.sidebar.selectbox(
    "Select User",
    sorted(data["user"].unique())
)

filtered = data[data["user"] == select_users]

# Metrics
col1, col2 = st.columns(2)
col1.metric("Total Alerts", len(filtered))
col2.metric("Max Anomaly Score", round(filtered["anomaly_scores"].max(), 2))

st.divider()

st.subheader("Alerts")
st.dataframe(filtered, use_container_width=True)
