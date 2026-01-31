# 🏋️ Cloud-Native CV Benchmarking Platform
By Christoph Gassner, Simon Bindreiter, Marco Sarkady
## 🚀 Step-by-Step Tutorial (Reproducibility Guide)

This section allows an external reviewer to reproduce the full system.

## 📖 Overview

This project implements a hybrid benchmarking platform designed to evaluate the performance of different Computer Vision models on edge devices. The system decouples local inference from cloud analysis using an **Event-Driven Architecture**.

### Related Work / Research
TODO

**Key Features:**
* **Edge Computing:** Local processing of video datasets using MediaPipe (and OpenPose).
* **Event-Driven Ingestion:** High-throughput ingestion via Google Pub/Sub (decoupling producers from consumers).
* **Serverless Processing:** Google Cloud Functions (Gen 2) to handle data validation and database writes.
* **Visualization:** A Streamlit dashboard connected to Firestore for near real-time analytics.

### Architecture Flow
`Edge Client (Docker/Python)` → `Ingestion API (FastAPI)` → `Google Pub/Sub` → `Cloud Function (Worker)` → `Firestore (NoSQL)` → `Streamlit Dashboard`

---

## 🛠 Prerequisites

To reproduce this project, ensure the following requirements are met:

1.  **Google Cloud Project:**
    * Create a project (e.g., `clc-group-vision-2026`).
    * **Enable Billing:** Required for Cloud Functions, Cloud Run, and Pub/Sub (even for Free Tier usage).
    * **Install Google Cloud SDK:** [Installation Guide](https://cloud.google.com/sdk/docs/install)
    * **Authenticate:**
        ```bash
        gcloud auth application-default login
        gcloud config set project clc-group-vision-2026
        ```

2.  **Python Environment:**
    * Python 3.12+

3. **🔐 Environment Variables (Configuration):**

    This project uses environment variables to separate configuration from code.

         TODO: Where put the file? In src? 
    All environment variables can be provided via:
    - a local `.env` file (for development)

    `.env` files are **not committed** to the repository.

### Environment Variables Overview

| Variable | Component | Purpose |
|--------|---------|--------|
| `GOOGLE_CLOUD_PROJECT` | Backend, Worker, Dashboard | Identifies the GCP project |
| `FIRESTORE_DATABASE` | Worker, Dashboard | Explicit non-default Firestore database |
| `INGESTION_API_URL` | Edge Client | Base URL of the Ingestion API |
TODO: ADD DOCKER ENV VARIABLES
---

## 🚀 Part 1: Cloud Infrastructure Setup

Run the following commands once to initialize the Google Cloud environment.

### 1.1 Enable Required APIs
```bash
gcloud services enable \
  cloudfunctions.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  pubsub.googleapis.com \
  firestore.googleapis.com \
  eventarc.googleapis.com \
  --project=clc-group-vision-2026
```

### 1.2 Create the Message Queue (Pub/Sub)

```bash
gcloud pubsub topics create cv-metrics-topic --project=clc-group-vision-2026
```

The Pub/Sub topic name (`cv-metrics-topic`) is currently **hardcoded**
in both the Ingestion API and the Cloud Worker.

This design choice reduces configuration complexity for this prototype
and improves reproducibility.

### 1.3 Create the Database (Firestore)

1. Go to Google Cloud Firestore and [create a Database](https://console.cloud.google.com/firestore/create-database?project=clc-group-vision-2026).

2. Select Native Mode.

3. Select Region: europe-west3 (Frankfurt).

4. Database ID: clc-group-vision-2026.

### Required Environment Variable

The Firestore database ID must be provided explicitly via an environment variable:

```env
FIRESTORE_DATABASE=clc-group-vision-2026
```

## ☁️ Part 2: Backend (Ingestion API)
The API acts as a "Producer." It accepts JSON metrics via HTTP and pushes them to the Pub/Sub topic. It does not write to the database directly.

### 2.1 Setup
Create a virtual environment (e.g., .venv-backend) and install dependencies:
```bash
pip install fastapi uvicorn google-cloud-pubsub
```

### 2.2 Run Locally
```bash
cd src/backend
uvicorn main:app --reload
```
- Status: Server is running at http://127.0.0.1:8000.

- Verification: POST /metrics endpoint is active.

Local execution allows reviewers to verify the ingestion logic without requiring cloud deployment.
Deployment to Cloud Run represents the production setup.

### 2.3: Deploy Ingestion API to Cloud Run

The Ingestion API is deployed as a **Cloud Run service** and acts as the public
entry point for all edge clients.

### Deploy Command

Run the following command from the `src/backend/` directory:

```bash
cd src/backend

gcloud run deploy ingestion-api \
  --source . \
  --region europe-west3 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=clc-group-vision-2026
```
After deployment, Cloud Run provides a public HTTPS endpoint.

### Required Environment Variable for Edge Clients

The returned URL must be provided to all edge clients via:
```env
INGESTION_API_URL=https://<cloud-run-url>
```

## ⚙️ Part 3: Cloud Worker (Serverless Processing)
The Worker is a Cloud Function (Consumer) that triggers automatically when a message arrives in Pub/Sub and writes it to Firestore.
### 3.1 Deploy to Google Cloud

Run this command from the src/worker/ directory:
```Bash
cd src/worker

gcloud functions deploy process-metrics-worker \
  --gen2 \
  --region=europe-west3 \
  --runtime=python310 \
  --source=. \
  --entry-point=subscribe \
  --trigger-topic=cv-metrics-topic \
  --project=clc-group-vision-2026 \
  --set-env-vars FIRESTORE_DATABASE=clc-group-vision-2026
  ```
*Note*: The first deployment may take a few minutes.

### Required Environment Variables

The Cloud Function expects the following environment variable:

```env
FIRESTORE_DATABASE=clc-group-vision-2026
```
## Part 4: Edge Client (Data Extraction)
The Edge Client processes the raw video dataset, extracts pose landmarks, and sends performance metrics to the Backend.

To ensure consistent execution across different environments (Windows, macOS, Linux), we have containerized (Docker) the extraction process.

    TODO: Docker Instructions

    [Placeholder for Christoph] 
    Please insert instructions here for building and running the Docker containers for MediaPipe and OpenPose.

Each container:
- Processes videos locally
- Measures runtime, FPS, memory, detection quality
- Sends metrics via HTTP to the ingestion API

### Required Environment Variable (Edge Client)

Each edge container requires the following environment variable:

```env
INGESTION_API_URL=https://<cloud-run-url>
```

## 📊 Part 5: Dashboard

The dashboard connects directly to Firestore to visualize the benchmarking results.

### 5.1 Installation
```bash
pip install streamlit pandas google-cloud-firestore
```

### 5.2 Configuration
Ensure you are authenticated locally so the dashboard can access the database:

```Bash
gcloud auth application-default login
gcloud config set project clc-group-vision-2026
```

### 5.3 Run Dashboard
```Bash
streamlit run src/dashboard/app.py
```
Open your browser at http://localhost:8501.

## Part 6: Deploy Dashboard to Cloud Run
We use Cloud Run Buildpacks - no docker required here.

### 6.1 Required files in src/dashboard/

```bash
app.py
requirements.txt
Procfile
```
### 6.2 Deploy
```Bash
cd src/dashboard

gcloud run deploy clc-vision-dashboard \
  --source . \
  --region europe-west3 \
  --allow-unauthenticated \
  --set-env-vars FIRESTORE_DATABASE=clc-group-vision-2026
```
Access is publicly available via HTTPS (unauthenticated).

This setup allows full end-to-end reproducibility of the edge-to-cloud
benchmarking pipeline without modifying application code.

## Lessons learned
