import os
import json
from fastapi import FastAPI, HTTPException
from google.cloud import pubsub_v1
import uuid

app = FastAPI()

# Konfiguration
PROJECT_ID = "clc-group-vision-2026"
TOPIC_ID = "cv-metrics-topic"

# Publisher Client initialisieren
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

@app.get("/")
def root():
    return {"status": "Pub/Sub Producer Ready"}

@app.post("/metrics")
async def ingest_metrics(metrics: dict):
    try:
        # 1. Wir generieren die ID hier, damit wir sie tracken können
        doc_id = str(uuid.uuid4())
        metrics["firestore_id"] = doc_id
        
        # 2. Daten für den Versand vorbereiten (Muss Byte-String sein)
        data_str = json.dumps(metrics)
        data_bytes = data_str.encode("utf-8")

        # 3. Asynchron an Pub/Sub senden
        # Das geht extrem schnell, da wir nicht auf die DB warten
        future = publisher.publish(topic_path, data_bytes)
        
        # Optional: Auf Bestätigung vom Pub/Sub Server warten (für Debugging gut)
        msg_id = future.result()

        return {"status": "queued", "msg_id": msg_id, "doc_id": doc_id}

    except Exception as e:
        print(f"Error publishing: {e}")
        raise HTTPException(status_code=500, detail=str(e))