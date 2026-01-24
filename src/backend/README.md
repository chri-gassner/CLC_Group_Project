# Backend Setup – FastAPI + Firestore (Local Development)

This README describes the **minimal steps** to run the local ingestion backend
(**FastAPI → Firestore**) for the CLC Group Project.

The backend receives metric JSONs via HTTP (`POST /metrics`) and stores them in
Google Cloud Firestore.

---

## 1. Prerequisites

- A **separate virtual environment** for the backend  
  (do NOT share with MediaPipe / OpenCV)
- A Google Cloud project (e.g. `clc-group-vision-2026`)
- Billing enabled for the Google Cloud project

---

## 2. Install Google Cloud CLI

Install the Google Cloud SDK (this provides the `gcloud` command):

https://docs.cloud.google.com/sdk/docs/install-sdk#latest-version

After installation:
- Restart your terminal
- Verify installation:

```bash
gcloud --version
```

## 3. Authenticate locally (Application Default Credentials)
```bash
gcloud auth application-default login
```

(Optional but recommended – avoids quota warnings)
```bash
gcloud auth application-default set-quota-project clc-group-vision-2026
```

## 4.Create and set the Google Cloud project
List available Projects:
```bash
gcloud projects list
```
Set the correct project:
```bash 
gcloud config set project clc-group-vision-2026
```
Verify
```bash
gcloud config get-value project
```

## 5.Enable Firestore API
```bash
gcloud services enable firestore.googleapis.com --project=clc-group-vision-2026
```
## 6.Create Firestore Database
Create the database in the Google Cloud Console:

https://console.cloud.google.com/datastore/create-database?project=clc-group-vision-2026

Choose:
- Firestore in Native mode
- Database ID: (default) or a project-specific ID
- Region: EU (e.g. europe-west3)
- Security: Test mode (sufficient for a student project)

Wait ~1 minute for propagation.

## 6.1 Package fix
MediaPipe requires protobuf < 5.

Do NOT install FastAPI, Firestore, or Google Cloud libraries in the same
environment as MediaPipe.

For the edge / MediaPipe environment:
```bash
pip install "protobuf>=4.25.3,<5" --force-reinstall
```
**Recommended**: use a separate virtual environment for edge processing.: \
From the repository root:
```bash
python -m venv .venv-backend
```
Activate it:
```bash
.\.venv-backend\Scripts\Activate.ps1
```

## 7. Backend Pytohn environment
Install backend dependecies:

```bash
pip install fastapi uvicorn google-cloud-firestore
```

## 8. Run the backend Server
```bash
cd src/backend
uvicorn main:app --reload
```
You should see:
```bash
Uvicorn running on http://127.0.0.1:8000
```

## 9. Test Endpoint
Start src\extract_pose_mediapipe\extract_pose_mediapipe.py

Expected Response:
```bash
127.0.0.1:57419 - "POST /metrics HTTP/1.1" 200 OK
```

