import streamlit as st
from google.cloud import firestore
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Setup Page
st.set_page_config(page_title="CLC Vision Benchmark", layout="wide")

@st.cache_resource
def get_db():
    return firestore.Client(project="clc-group-vision-2026", database="clc-group-vision-2026")

db = get_db()

st.title("CLC Group: Vision Pipeline Benchmark")

# --- DATA FETCHING ---
def load_data():
    docs = db.collection("metrics").stream()
    data = []
    for doc in docs:
        d = doc.to_dict()
        data.append(d)
    return pd.DataFrame(data)

df = load_data()

# --- DATA CLEANING (GLOBAL CHECK) ---
# Ensure we only work with rows that have essential plotting data
if not df.empty:
    df = df.dropna(subset=['model', 'ts_utc'])
    # Optional: Fill missing run identifiers if they don't exist
    if 'run_id' not in df.columns:
        df['run_id'] = df.index.astype(str)

if not df.empty:
    # --- 1. TOP LEVEL KPIS ---
    for model in df['model'].unique():
        st.subheader(f"Metrics: {model}")
        m1, m2, m3, m4 = st.columns(4)
        
        model_df = df[df['model'] == model]
        
        m1.metric("Total Runs", len(model_df))
        m2.metric("Median Latency", f"{model_df['infer_lat_median_ms'].median():.1f}ms")
        m3.metric("Avg Visibility", f"{model_df['mean_visibility'].mean()*100:.1f}%")
        m4.metric("Peak RSS Memory", f"{model_df['rss_peak_mb'].max():.0f} MB")

    st.divider()

    # --- 2. LATENCY & PERFORMANCE ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Latency Distribution by Model")
        # Filter NaNs specifically for this plot
        df_lat = df.dropna(subset=['infer_lat_median_ms', 'label'])
        fig_lat = px.box(
            df_lat, 
            x="label", 
            y="infer_lat_median_ms",
            color="model", 
            title="Median Inference Latency: Model Comparison",
            labels={"infer_lat_median_ms": "Time (ms)", "label": "Exercise Type"},
            points="all"
        )
        fig_lat.update_layout(boxmode='group') 
        st.plotly_chart(fig_lat, use_container_width=True)

    with col_right:
        st.subheader("🎯 CONFIDENCE (LATEST)")
        latest_metrics = df.dropna(subset=['mean_visibility']).groupby('model').first()
        num_models = len(latest_metrics)
        
        if num_models > 0:
            fig_gauge = go.Figure()
            spacing = 0.05
            available_height = 1.0 - (spacing * (num_models - 1))
            height_per_gauge = available_height / num_models

            for i, (model_name, row) in enumerate(latest_metrics.iterrows()):
                y_bottom = i * (height_per_gauge + spacing)
                y_top = min(y_bottom + height_per_gauge, 1.0)
                
                fig_gauge.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = row['mean_visibility'] * 100,
                    title = {'text': str(model_name).upper(), 'font': {'size': 14}},
                    domain = {'x': [0, 1], 'y': [y_bottom, y_top]},
                    gauge = {'bar': {'color': "#00ff41"}, 'axis': {'range': [0, 100]}}
                ))

            fig_gauge.update_layout(template="plotly_dark", height=250 * num_models, margin=dict(l=30, r=30, t=50, b=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # --- 3. MEMORY LEAK ANALYSIS (RUN-BY-RUN TREND) ---
    st.subheader("Memory Trend Analysis (Run-by-Run)")
    
    # Drop NaNs for memory metrics to ensure continuous lines
    df_mem_clean = df.dropna(subset=['mem_mb_start', 'mem_mb_end']).sort_values("ts_utc")
    
    models = df_mem_clean['model'].unique()
    mem_cols = st.columns(len(models))

    for i, model in enumerate(models):
        with mem_cols[i]:
            model_df = df_mem_clean[df_mem_clean['model'] == model].copy()
            
            # Create the figure
            fig_mem = go.Figure()

            # Add "Before" Line
            fig_mem.add_trace(go.Scatter(
                x=model_df['ts_utc'], 
                y=model_df['mem_mb_start'],
                mode='lines+markers',
                name='Start (Baseline)',
                line=dict(color='#636EFA', width=2)
            ))

            # Add "After" Line
            fig_mem.add_trace(go.Scatter(
                x=model_df['ts_utc'], 
                y=model_df['mem_mb_end'],
                mode='lines+markers',
                name='End (Post-Inference)',
                line=dict(color='#EF553B', width=2, dash='dot')
            ))

            fig_mem.update_layout(
                title=f"Memory Stability: {model}",
                xaxis_title="Run Timestamp",
                yaxis_title="Memory (MB)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_dark",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_mem, use_container_width=True)

    # --- 4. DATA QUALITY ---
    st.subheader("Landmark Visibility Distribution")
    df_vis = df.dropna(subset=['mean_visibility'])
    fig_vis = px.histogram(
        df_vis, 
        x="mean_visibility", 
        color="model", 
        nbins=20, 
        marginal="rug", 
        barmode="overlay",
        title="Quality Consistency by Model"
    )
    fig_vis.update_traces(opacity=0.7)
    st.plotly_chart(fig_vis, use_container_width=True)

    # --- 5. RAW DATA ---
    with st.expander("View Raw Metrics Table"):
        st.dataframe(df.sort_values("ts_utc", ascending=False))

else:
    st.warning("No data found in Firestore 'metrics' collection.")