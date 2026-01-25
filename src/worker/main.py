import base64
import json
import functions_framework
from google.cloud import firestore

# DB Client initialisieren (außerhalb der Funktion für Performance/Caching)
db = firestore.Client(
    project="clc-group-vision-2026",
    database="clc-group-vision-2026"
)

@functions_framework.cloud_event
def subscribe(cloud_event):
    """
    Triggered from a message on a Cloud Pub/Sub topic.
    """
    # 1. Daten aus der Pub/Sub Nachricht decodieren
    pubsub_message = base64.b64decode(cloud_event.data["message"]["data"]).decode("utf-8")
    metrics = json.loads(pubsub_message)
    
    print(f"Worker received metric for: {metrics.get('video_path', 'unknown')}")

    # 2. ID extrahieren (die wir im Backend generiert haben) oder neu erstellen
    doc_id = metrics.get("firestore_id")
    if not doc_id:
        # Fallback
        import uuid
        doc_id = str(uuid.uuid4())

    # 3. In Firestore schreiben
    try:
        # Wir schreiben in die gleiche Collection wie vorher
        db.collection("metrics").document(doc_id).set(metrics)
        print(f"Successfully wrote document {doc_id} to Firestore.")
    except Exception as e:
        print(f"Error writing to Firestore: {e}")
        # Fehler werfen sorgt dafür, dass Pub/Sub es nochmal versucht (Retry)
        raise e