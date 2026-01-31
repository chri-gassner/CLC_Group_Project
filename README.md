# 🏋️ Cloud-Native CV Benchmarking Platform
By Christoph Gassner, Simon Bindreiter, Marco Sarkady
## 🚀 Step-by-Step Tutorial (Reproducibility Guide)

This section allows an external reviewer to reproduce the full system.

## 📖 Overview

This project implements a hybrid benchmarking platform designed to evaluate the performance of different Computer Vision models on edge devices. The system decouples local inference from cloud analysis using an **Event-Driven Architecture**.

### Related Work / Research
This project builds upon existing research in distributed computer vision and event-driven architectures. Our platform addresses the "Edge-to-Cloud" data gap by implementing a decoupled, scalable pipeline.

1. Edge vs. Cloud Trade-offs in Real-time AI
Research consistently highlights that while Cloud AI offers superior computational resources, it suffers from network latency and bandwidth costs. Conversely, Edge computing excels in real-time processing and data privacy.

Key Finding: Hybrid models that use the edge for inference and the cloud for analytical aggregation (as seen in this project) offer the most balanced performance for IoT-scale deployments.

Source: Benchmarking Edge Computing vs. Cloud Computing for Real-Time Data Processing (2025)

2. Event-Driven Architectures for Real-Time Analytics
Modern IoT systems are moving away from direct database writes to Event-Driven Architectures (EDA). By using a message broker (like Pub/Sub) to decouple producers from consumers, systems achieve higher fault tolerance and scalability.

Key Finding: Decoupling ensures that high-frequency telemetry from edge devices does not overwhelm the backend. Asynchronous communication allows the system to handle bursty workloads while maintaining a responsive user dashboard.

Source: Event-Driven Architectures for Real-Time Data Processing: A Deep Dive (2025)

3. Cloud-Native Benchmarking Automation
Traditional benchmarking is often manual and error-prone. Modern research suggests using "Benchmarking Operators" and containerized cloud-native environments to automate the collection of metrics across distributed nodes.

Key Finding: Separating the ingestion (API) from the processing (Worker) ensures that the benchmarking system itself does not become a bottleneck for the edge devices, preserving the integrity of the performance data.

Source: Cloud-Native-Bench: An Extensible Benchmarking Framework (2024)

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

## Lessons Learned
The development of this benchmarking platform provided several key insights regarding security, data integrity, and container orchestration:

1. Security and Authentication
Authentication is Essential: Implementing robust authentication was critical for the secure communication between the edge clients and the Google Cloud backend. This ensures that only authorized metrics are ingested into the pipeline and prevents unauthorized access to the cloud infrastructure.

2. Data Integrity and Comparison
Standardized Metrics for Comparison: For a benchmarking platform to be effective, there must be a unified set of metrics across all tested models. Standardizing how performance is measured ensures that the comparison between different CV frameworks remains meaningful and consistent.

3. Container Orchestration
Path Management via Volume Mounts: Proper configuration of Volume Mounts is vital when working with containerized CV models. We learned that precise path management is necessary to ensure the containers can access local video datasets and write output logs without permission or directory errors.

4. Protobuf Versioning & Dependency Isolation
Managing Dependency Conflicts: We encountered significant "Dependency Hell" because MediaPipe requires older versions of protobuf, while the Google Cloud client libraries require the latest versions.

The Microservices Solution: Decoupling the Edge Client from the Backend into separate virtual environments or Docker containers was the only viable way to resolve these conflicts without breaking core functionality.
