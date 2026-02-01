import streamlit as st
from google.cloud import firestore
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Setup Page
st.set_page_config(page_title="CLC Vision Benchmark", layout="wide")

# Load environment variables
from pathlib import Path
    
env_path = (Path(__file__).parent.parent / "src" / "env" / "common.env").resolve()
load_dotenv(env_path)

@st.cache_resource
def get_db():
    # Ensure your service account or gcloud auth is set up for this project
    database = os.getenv("FIRESTORE_DATABASE")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    return firestore.Client(project=project, database=database)

db = get_db()

st.title("🏋️ CLC Group: Vision Pipeline Benchmark")

# --- DATA FETCHING ---
def load_data():
    docs = db.collection("metrics").stream()
    data = []
    for doc in docs:
        d = doc.to_dict()
        data.append(d)
    return pd.DataFrame(data)

df = load_data()

if not df.empty:
    # --- 1. TOP LEVEL KPIS ---
    for model in df['model'].unique():
        st.subheader(f"Metrics: {model}")
        m1, m2, m3, m4 = st.columns(4)
        
        # Filter dataframe for the specific model
        model_df = df[df['model'] == model]
        
        m1.metric("Total Runs", len(model_df))
        m2.metric("Median Latency", f"{model_df['infer_lat_median_ms'].median():.1f}ms")
        m3.metric("Avg Visibility", f"{model_df['mean_visibility'].mean()*100:.1f}%")
        m4.metric("Peak RSS Memory", f"{model_df['rss_peak_mb'].max():.0f} MB")

    st.divider()

    # --- 2. LATENCY & PERFORMANCE ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Latency Distribution (p50, p90, p99)")
        # Creating a box plot to show the spread of latency
        fig_lat = px.box(df, x="label", y=["infer_lat_median_ms", "infer_lat_p90_ms", "infer_lat_p99_ms"],
                         title="Inference Latency by Exercise Type",
                         labels={"value": "Time (ms)", "variable": "Percentile"})
        st.plotly_chart(fig_lat, use_container_width=True)

    with col_right:
        st.subheader("🎯 CONFIDENCE (LATEST)")
        
        # Group to get the latest visibility per model
        latest_metrics = df.groupby('model').first()
        num_models = len(latest_metrics)
        
        if num_models > 0:
            fig_gauge = go.Figure()

            # The 'domain' for all gauges must fit between 0 and 1
            # We calculate height per gauge and add a small padding (spacing)
            spacing = 0.05
            available_height = 1.0 - (spacing * (num_models - 1))
            height_per_gauge = available_height / num_models

            for i, (model_name, row) in enumerate(latest_metrics.iterrows()):
                # Calculate the exact Y coordinates for this specific gauge
                # We iterate from bottom (0) to top (1)
                y_bottom = i * (height_per_gauge + spacing)
                y_top = y_bottom + height_per_gauge
                
                # Double-check: Plotly crashes if y_top > 1.0 due to float rounding
                y_top = min(y_top, 1.0)
                
                fig_gauge.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = row['mean_visibility'] * 100,
                    title = {'text': str(model_name).upper(), 'font': {'size': 14}},
                    domain = {'x': [0, 1], 'y': [y_bottom, y_top]},
                    gauge = {
                        'bar': {'color': "#00ff41"},
                        'axis': {'range': [0, 100]}
                    }
                ))

            fig_gauge.update_layout(
                template="plotly_dark", 
                # Adjust total widget height so gauges don't look squashed
                height=250 * num_models if num_models > 1 else 300,
                margin=dict(l=30, r=30, t=50, b=30)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("No confidence data available.")

    # --- 3. RESOURCE & QUALITY ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Memory Leak Analysis")
        # Tracking memory growth (Start vs End)
        fig_mem = go.Figure()
        fig_mem.add_trace(go.Bar(name='Start RAM', x=df.index, y=df['mem_mb_start']))
        fig_mem.add_trace(go.Bar(name='End RAM', x=df.index, y=df['mem_mb_end']))
        fig_mem.update_layout(barmode='group', title="Memory Consumption (Start vs End MB)")
        st.plotly_chart(fig_mem, use_container_width=True)

    with col_b:
        st.subheader("Data Quality (Visibility)")
        # Histogram of landmark visibility
        fig_vis = px.histogram(df, x="mean_visibility", nbins=20, 
                               color_discrete_sequence=['indianred'],
                               title="Landmark Visibility Distribution")
        st.plotly_chart(fig_vis, use_container_width=True)

    # --- 4. RAW DATA ---
    with st.expander("View Raw Metrics Table"):
        st.dataframe(df.sort_values("ts_utc", ascending=False))

else:
    st.warning("No data found in Firestore 'metrics' collection.")