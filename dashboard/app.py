import streamlit as st

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