import os
import streamlit as st
from google.cloud import firestore
import pandas as pd
import google.auth

st.set_page_config(page_title="CLC Vision Benchmark", layout="wide")

st.title("🏋️ CLC Group: Computer Vision Benchmark")
st.caption("Data source: Google Cloud Firestore")

# ----------------------------
# Firestore connection (Cloud Run friendly)
# ----------------------------
@st.cache_resource
def get_db():
    project = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or google.auth.default()[1]
    )
    database = os.getenv("FIRESTORE_DATABASE", "clc-group-vision-2026")

    return firestore.Client(project=project, database=database)

db = get_db()

# ----------------------------
# Controls
# ----------------------------
st.subheader("Live Metrics Feed")

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

limit = c1.selectbox("Limit", [50, 100, 250], index=1)

model_filter = c2.selectbox(
    "Model",
    ["(all)", "mediapipe_pose", "openpose_pose"],
    index=0,
)

type_filter = c3.selectbox(
    "Type",
    ["(all)", "video_ok", "video_error", "run_end"],
    index=0,
)

refresh = c4.button("🔄 Refresh data")

# ----------------------------
# Fetch + cache (only refreshes when you click)
# ----------------------------
@st.cache_data
def fetch_metrics(limit: int):
    q = (
        db.collection("metrics")
        .order_by("ts_utc", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    docs = q.stream()
    rows = []
    for doc in docs:
        d = doc.to_dict()
        d["_id"] = doc.id
        rows.append(d)
    return rows

if refresh:
    fetch_metrics.clear()

data = fetch_metrics(limit)

if not data:
    st.warning("No data found in Firestore collection 'metrics'.")
    st.stop()

df = pd.DataFrame(data)

# ----------------------------
# Apply filters (safe)
# ----------------------------
if model_filter != "(all)" and "model" in df.columns:
    df = df[df["model"] == model_filter]

if type_filter != "(all)" and "type" in df.columns:
    df = df[df["type"] == type_filter]

# Coerce numeric columns (safe)
for c in ["eff_fps", "wall_s", "detect_rate", "mean_visibility", "nan_rate_xyz", "mem_mb_delta"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ----------------------------
# KPI Row
# ----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Videos loaded", len(df))

if "type" in df.columns:
    ok = int((df["type"] == "video_ok").sum())
    err = int((df["type"] == "video_error").sum())
    k2.metric("video_ok", ok)
    k3.metric("video_error", err)
else:
    k2.metric("video_ok", "—")
    k3.metric("video_error", "—")

if "eff_fps" in df.columns and df["eff_fps"].notna().any():
    k4.metric("Avg eff_fps", f"{df['eff_fps'].mean():.2f}")
else:
    k4.metric("Avg eff_fps", "—")

st.markdown("---")

# ----------------------------
# Table
# ----------------------------
st.dataframe(
    df.sort_values(by="ts_utc", ascending=False) if "ts_utc" in df.columns else df,
    use_container_width=True,
)

# ----------------------------
# Simple chart: eff_fps by label (video_ok only)
# ----------------------------
if "eff_fps" in df.columns and "label" in df.columns:
    chart_df = df.copy()
    if "type" in chart_df.columns:
        chart_df = chart_df[chart_df["type"] == "video_ok"]

    if not chart_df.empty:
        agg = (
            chart_df.groupby("label", as_index=False)["eff_fps"]
            .mean()
            .sort_values("eff_fps", ascending=False)
        )
        st.subheader("Mean eff_fps by label (video_ok)")
        st.bar_chart(agg, x="label", y="eff_fps")
