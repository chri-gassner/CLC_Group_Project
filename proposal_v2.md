# Projektproposal: Cloud-Native CV Benchmarking Platform

## 1. Ziel des Projekts
Das Ziel ist die Entwicklung einer **hybriden Cloud-Plattform zur systematischen Evaluation und zum Vergleich von Computer-Vision-Modellen**. Wir verbinden ein lokal entwickeltes Modul zur Gesten- bzw. Übungserkennung (**Edge Computing**) mit einer cloudbasierten Analyse- und Visualisierungsschicht. Der Fokus liegt auf dem Vergleich von **MediaPipe vs. OpenPose** unter realistischen Edge-Bedingungen.

Konkret läuft die Bildverarbeitung (MediaPipe vs. OpenPose) lokal auf dem Gerät (Edge), um Bandbreite zu sparen, Latenz zu minimieren und Datenschutz zu gewährleisten. In die Cloud werden **ausschließlich strukturierte Analyse-Ergebnisse (Telemetrie)** wie Latenz, FPS, CPU-Last, Konfidenzwerte und Fehlerindikatoren übertragen. Diese werden zentral gespeichert, aggregiert und live visualisiert.

Der zentrale Erkenntnisgewinn des Projekts besteht darin, **Performance- und Ressourcen-Trade-offs zwischen unterschiedlichen CV-Modellen quantitativ zu bewerten** und gleichzeitig zu demonstrieren, wie eine cloud-native Telemetrie-Architektur für Edge-AI-Anwendungen betrieben und skaliert werden kann.

### Was wird neu gebaut vs. was existiert?
* **Existiert:**  
  * Die CV-Frameworks **OpenPose** und **MediaPipe** (aus dem CV-Kurs)  
  * Google Cloud Services (**Cloud Run**, **Firestore**, **Cloud Monitoring**)

* **Wird neu gebaut:**  
  * **Edge Client (Python/Docker):**  
    Ein lokaler Container, der Webcam-Daten verarbeitet, Pose-Erkennung durchführt und standardisierte Telemetrie (z. B. Latenz, FPS, CPU-Auslastung, Konfidenz) erzeugt. Es werden **keine Video- oder Bilddaten** an die Cloud übertragen.
  * **Cloud API (FastAPI):**  
    Ein Microservice zur validierten Entgegennahme der Telemetriedaten, inkl. Schema-Validierung, Session-IDs und Logging.
  * **Asynchrone Ingestion-Komponente:**  
    Entkopplung von Dateneingang und Persistierung (z. B. mittels Cloud-nativer Messaging-Mechanismen), um Robustheit, Retry-Fähigkeit und Skalierbarkeit zu gewährleisten.
  * **Cloud Dashboard (Streamlit):**  
    Eine Webanwendung, die live darstellt, welches Modell aktuell besser performt (z. B. „MediaPipe ist 20 % schneller als OpenPose“) und verschiedene Modell- sowie Cloud-Metriken visualisiert.

---

## 2. High-Level Architektur

Wir nutzen eine **cloud-native Microservice-Architektur** auf Basis von Containern mit klarer Trennung zwischen Edge-Verarbeitung, Ingestion, Persistierung und Visualisierung.

![alt text](image-1.png)

**Ablauf (vereinfacht):**
1. Edge Client führt die Pose-Erkennung durch und erzeugt Telemetrie.
2. Cloud API nimmt die Daten entgegen, validiert sie und protokolliert sie.
3. Asynchrone Weiterleitung zur Persistierung zur Entkopplung der Komponenten.
4. Speicherung der Metriken in Firestore.
5. Dashboard und Monitoring-Komponenten greifen lesend auf die Daten zu.

---

## 3. Beziehung zu Cloud Computing

Das Projekt zeigt eine realistische **Edge-to-Cloud-Architektur**, wie sie in IoT- und ML-Monitoring-Szenarien eingesetzt wird:

* **Edge Computing:**  
  Die rechenintensive CV-Inferenz erfolgt lokal am Gerät, um Latenz und Bandbreitenbedarf zu minimieren.
* **Microservices:**  
  API, Ingestion und Dashboard sind logisch getrennte Services und unabhängig skalierbar.
* **Containerization:**  
  Docker wird sowohl lokal (reproduzierbare CV-Umgebung) als auch in der Cloud (Deployment) eingesetzt.
* **Serverless Computing:**  
  Die Cloud-Komponenten laufen auf Google Cloud Run und skalieren automatisch bis auf Null.
* **Observability & Monitoring:**  
  Nutzung von Cloud Logging und Cloud Monitoring zur Analyse von Latenzen, Fehlerquoten, Durchsatz und Ressourcenverbrauch.

Die Cloud dient damit nicht nur als Hosting-Plattform, sondern als **zentrale Benchmarking-, Analyse- und Betriebsschicht**.

---

## 4. Meilensteine

Start der Implementierung nach der Proposal-Abnahme (Weihnachtsferien/Jänner).

| Meilenstein | Beschreibung & Ziel | Deadline (Intern) |
| :--- | :--- | :--- |
| **M1: Cloud Setup** | GCP-Projekt, Terraform-Basis, IAM und Container Registry eingerichtet. | 30.12.2025 |
| **M2: Ingestion Service** | FastAPI-Service mit Schema-Validierung, Authentifizierung und strukturiertem Logging. | 07.01.2026 |
| **M3: Dashboard Skeleton** | Streamlit-App visualisiert Dummy-Daten, Vergleichs-Layout steht. | 14.01.2026 |
| **M4: Integration & Metrics** | Edge-Client sendet reale Telemetrie (FPS, Latenz, CPU). | 21.01.2026 |
| **M5: Cloud Monitoring** | Einbindung von Cloud-Metriken, Test mit mehreren parallelen Edge-Clients. | 28.01.2026 |
| **M6: Finalisierung** | Dokumentation, Benchmark-Auswertung und Präsentationsvorbereitung. | 31.01.2026 |

---

## 5. Aufgabenverteilung

| Teammitglied | Rolle | Verantwortungsbereich |
| :--- | :--- | :--- |
| **Christoph** | **Cloud Backend** | FastAPI, Telemetrie-Schema, Ingestion-Logik, Cloud-Run-Deployment |
| **Simon** | **Infra & Data** | Firestore, Terraform, IAM, Monitoring und Logging |
| **Marco** | **Edge & Viz** | CV-Integration (OpenPose/MediaPipe), Edge-Telemetrie, Streamlit-Dashboard |
