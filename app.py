import streamlit as st
import pandas as pd
from src.data_processing import load_and_merge
from src.features import create_features
from src.model import predict_proba
from src.utils import summarize_kpis

st.set_page_config(page_title="Predictive Delivery Optimizer", layout="wide")

st.title("🚚 NexGen Predictive Delivery Optimizer")

# Load data
st.sidebar.header("Data Upload")
orders_file = st.sidebar.file_uploader("Orders CSV", type="csv")
delivery_file = st.sidebar.file_uploader("Delivery CSV", type="csv")

if orders_file and delivery_file:
    df = load_and_merge(orders_file, delivery_file)
    df = create_features(df)

    st.success("Data Loaded Successfully ✅")
    total_orders, on_time_rate, avg_delay = summarize_kpis(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Orders", total_orders)
    col2.metric("On-Time %", f"{on_time_rate:.2f}%")
    col3.metric("Avg Delay (hrs)", f"{avg_delay:.2f}")

    st.write("### Sample Data")
    st.dataframe(df.head())

    # Prediction section
    st.write("### Predict Delay Probability")
    selected_row = st.selectbox("Select a random order:", df["order_id"])
    row = df[df["order_id"] == selected_row]

    if st.button("Predict Delay Risk"):
        proba = predict_proba(row)[0]
        st.metric("Predicted Delay Probability", f"{proba*100:.1f}%")

else:
    st.warning("Please upload required CSV files to continue.")
