# Backend Setup – FastAPI + Pub/Sub (Producer)

This README describes the **Ingestion API** for the CLC Group Project.
It acts as a "Producer": It accepts JSON metrics via HTTP and pushes them to a **Google Pub/Sub Topic**. It does **not** write to the database directly.

---

## 1. Prerequisites

- A **separate virtual environment** for the backend (Strongly Recommended!)
  - *Reason:* MediaPipe requires old `protobuf`, Google Cloud Pub/Sub requires new `protobuf`. They clash if installed together.
- A Google Cloud project (e.g. `clc-group-vision-2026`)
- Billing enabled (required for Pub/Sub & Cloud Run)

## 2. Infrastructure Setup (One-Time)

Ensure the Pub/Sub topic exists:

```bash
gcloud pubsub topics create cv-metrics-topic --project=clc-group-vision-2026
```
## 3. Installation
```bash
pip install fastapi uvicorn google-cloud-pubsub
```

## 4. Run Locally
Activate your backend virtual environment (.venv-backend) and install:
```bash
cd src/backend
uvicorn main:app --reload
```
Server is running at: http://127.0.0.1:8000

## 5. Test
Send a metric using the edge client script:

```bash
python src/extract_pose_mediapipe/extract_pose_mediapipe.py
```
*Note: If you get protobuf errors in the client script, run pip install "protobuf<4.25" --force-reinstall in the client environment.*

