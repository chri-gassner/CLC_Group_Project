from fastapi import FastAPI, HTTPException
from google.cloud import firestore
import uuid

app = FastAPI()
db = firestore.Client(
    project="clc-group-vision-2026",
    database="clc-group-vision-2026",
)

@app.get("/")
def root():
    return {"status": "up"}


@app.post("/metrics")
def ingest_metrics(metrics: dict):
    try:
        doc_id = str(uuid.uuid4())
        db.collection("metrics").document(doc_id).set(metrics)
        return {"status": "ok", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))
