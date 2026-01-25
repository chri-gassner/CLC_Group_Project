# Cloud Logic Setup – Pub/Sub + Cloud Functions (Event-Driven)

This guide documents the transition from a synchronous backend (Direct Firestore Write) to an asynchronous **Event-Driven Architecture** using Google Pub/Sub and Cloud Functions (Gen 2).

**Architecture:**
`Client` → `FastAPI` (Producer) → `Pub/Sub` (Queue) → `Cloud Function` (Worker) → `Firestore`

---

## 1. Prerequisites & Billing Setup

Cloud Functions and Eventarc require an active billing account (even for the Free Tier).

1.  **Go to Billing Console:** [Google Cloud Billing](https://console.cloud.google.com/billing)
2.  **Create Account:** Set up a billing profile (requires Credit Card/IBAN for identity verification).
3.  **Link Project:**
    - Go to "Account Management".
    - Find project `clc-group-vision-2026`.
    - Click `...` -> **Change billing** -> Select your new account.
4.  **Verify via CLI:**
    ```bash
    gcloud beta billing projects describe clc-group-vision-2026
    # Must return: "billingEnabled: true"
    ```
5.  **Set Budget Alert (Recommended):**
    - Go to **Budgets & alerts** in the Billing Console.
    - Create a budget for **€1.00** to receive emails if costs occur.

---

## 2. Infrastructure Initialization

Enable required Google Cloud APIs for Serverless and Messaging:

```bash
gcloud services enable \
  cloudfunctions.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  pubsub.googleapis.com \
  eventarc.googleapis.com \
  --project=clc-group-vision-2026
```

Create the Pub/Sub Topic (The "Mailbox"):

```bash 
gcloud pubsub topics create cv-metrics-topic --project=clc-group-vision-2026
```

## 3. Backend Update (The Producer)
The FastAPI service now pushes to Pub/Sub instead of writing to Firestore.

Dependencies:

```bash
pip install google-cloud-pubsub
```

## 4. Worker Setup (The Consumer)
Deploy the Cloud Function: Run this command from the src/worker/ directory:

```bash
cd src/worker

gcloud functions deploy process-metrics-worker \
  --gen2 \
  --region=europe-west3 \
  --runtime=python310 \
  --source=. \
  --entry-point=subscribe \
  --trigger-topic=cv-metrics-topic \
  --project=clc-group-vision-2026
```
*Note: The first deployment may take 2-4 minutes.*

## 5. Dashboard Configuration

To visualize data, the dashboard must connect to the specific Firestore project/database.

```python
@st.cache_resource
def get_db():
    return firestore.Client(
        project="clc-group-vision-2026",
        database="clc-group-vision-2026"  # Crucial: Must match the named DB
    )
```

## 6. Verification & Testing

1. Start Backend: uvicorn main:app --reload (in src/backend)

2. Start Dashboard: streamlit run src/dashboard/app.py

3. Run Client: Execute extract_pose_mediapipe.py.

Expected Data Flow:

1. Client prints: ☁️ Uploading metrics...

2. Backend logs: 200 OK (Immediate response)

3. Cloud Console Logs:

    - Go to Cloud Functions Logs.

    - Select process-metrics-worker.

    - You should see: Worker received metric for: ... and Successfully wrote document....

4. Dashboard: Click "Refresh Data" to see new entries.