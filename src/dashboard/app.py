import streamlit as st
from google.cloud import firestore
import pandas as pd
import json

# Setup Page
st.set_page_config(page_title="CLC Vision Benchmark", layout="wide")

# Connect to Firestore
@st.cache_resource
def get_db():
    return firestore.Client(
        project="clc-group-vision-2026",
        database="clc-group-vision-2026"
    )

db = get_db()

st.title("🏋️ CLC Group: Computer Vision Benchmark")

# Fetch Data
st.subheader("Live Metrics Feed")

# Button to refresh data
if st.button("Refresh Data"):
    docs = db.collection("metrics").stream()
    data = []
    for doc in docs:
        d = doc.to_dict()
        # flatten simple nested structures if necessary
        data.append(d)

    if data:
        df = pd.DataFrame(data)
        
        # 1. KPI Row
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Videos", len(df))
        if "eff_fps" in df.columns:
            avg_fps = df["eff_fps"].mean()
            col2.metric("Avg Effective FPS", f"{avg_fps:.2f}")
        
        # 2. Data Table
        st.dataframe(df)

        # 3. Simple Charts
        if "eff_fps" in df.columns and "model" in df.columns:
            st.bar_chart(df, x="label", y="eff_fps", color="model")
            
    else:
        st.warning("No data found in Firestore 'metrics' collection.")

st.markdown("---")
st.caption("Data source: Google Cloud Firestore")