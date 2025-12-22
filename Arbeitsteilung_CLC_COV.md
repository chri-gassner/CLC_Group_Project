# Work Packages & Task Distribution  
## Project 1: Gesture Recognition  
## Project 2: Cloud-Native CV Benchmarking Platform

---

# 1. Gesamtübersicht aller Arbeitspakete

## Projekt 1 – Gesture Recognition

| ID | Arbeitspaket | Beschreibung | Abhängigkeiten |
|----|-------------|-------------|----------------|
| P1-1 | Gesture-Definition | Definition der ≥6 Fitness-Gesten | – |
| P1-2 | Dataset-Analyse | Analyse des vorhandenen Datasets | P1-1 |
| P1-3 | MediaPipe Pipeline | Keypoint-Extraktion & Features | P1-2 |
| P1-4 | OpenPose Pipeline | Keypoint-Extraktion & Features | P1-2 |
| P1-5 | Gesture-Labeling | Keypoints → Gestenklassen | P1-3, P1-4 |
| P1-6 | ML-Modellwahl | Auswahl & Begründung | P1-5 |
| P1-7 | Training MediaPipe | Training & Tuning | P1-6 |
| P1-8 | Training OpenPose | Training & Tuning | P1-6 |
| P1-9 | Rep-Counter Logik | Wiederholungszählung | P1-7, P1-8 |
| P1-10 | Evaluation | Precision, Recall, Confusion | P1-7, P1-8 |
| P1-11 | Framework-Vergleich | MediaPipe vs. OpenPose | P1-10 |
| P1-12 | Paper (IEEE) | Wissenschaftliches Paper | P1-11 |
| P1-13 | Demo | Live-Demo | P1-9 |
| P1-14 | Präsentation | Slides & Vortrag | P1-12 |

---

## Projekt 2 – Cloud Computing

| ID | Arbeitspaket | Beschreibung | Abhängigkeiten |
|----|-------------|-------------|----------------|
| C2-1 | Architekturdesign | Cloud-Architektur + Diagramm | – |
| C2-2 | API-Spezifikation | JSON-Metrik-Schema | C2-1 |
| C2-3 | Edge-Service MediaPipe | Dockerisierte Inferenz | P1-3 |
| C2-4 | Edge-Service OpenPose | Dockerisierte Inferenz | P1-4 |
| C2-5 | Ingestion API | FastAPI auf Cloud Run | C2-2 |
| C2-6 | Pub/Sub Setup | Topics, IAM | C2-1 |
| C2-7 | Processing Function | Cloud Function → Firestore | C2-6 |
| C2-8 | Firestore Schema | NoSQL Struktur | C2-2 |
| C2-9 | Dashboard | Streamlit Visualisierung | C2-7, C2-8 |
| C2-10 | IaC | Terraform Setup | C2-6 |
| C2-11 | Benchmarking | Edge vs. Cloud | C2-3, C2-4, C2-7 |
| C2-12 | GitHub Dokumentation | README & Setup | alle |
| C2-13 | Tutorial | Reproduzierbarkeit | C2-12 |
| C2-14 | Lessons Learned | Erkenntnisse | C2-11 |
| C2-15 | Präsentation | Slides & Demo | alle |

---

# 2. Aufgabenverteilung – Person 1

| ID | Arbeitspaket | Abhängigkeiten |
|----|-------------|----------------|
| P1-1 | Gesture-Definition | – |
| P1-2 | Dataset-Analyse | P1-1 |
| P1-3 | MediaPipe Pipeline | P1-2 |
| P1-7 | Training MediaPipe | P1-6 |
| P1-9 | Rep-Counter Logik | P1-7, P1-8 |
| C2-1 | Architekturdesign | – |
| C2-2 | API-Spezifikation | C2-1 |
| C2-3 | Edge-Service MediaPipe | P1-3 |
| C2-6 | Pub/Sub Setup | C2-1 |
| C2-8 | Firestore Schema | C2-2 |
| C2-9 | Dashboard | C2-7, C2-8 |
| C2-13 | Tutorial | C2-12 |

---

# 3. Aufgabenverteilung – Person 2

| ID | Arbeitspaket | Abhängigkeiten |
|----|-------------|----------------|
| P1-4 | OpenPose Pipeline | P1-2 |
| P1-8 | Training OpenPose | P1-6 |
| P1-10 | Evaluation | P1-7, P1-8 |
| P1-11 | Framework-Vergleich | P1-10 |
| C2-4 | Edge-Service OpenPose | P1-4 |
| C2-5 | Ingestion API | C2-2 |
| C2-7 | Processing Function | C2-6 |
| C2-10 | IaC | C2-6 |
| C2-12 | GitHub Dokumentation | alle |

---

# 4. Aufgabenverteilung – Person 3

| ID | Arbeitspaket | Abhängigkeiten |
|----|-------------|----------------|
| P1-5 | Gesture-Labeling | P1-3, P1-4 |
| P1-6 | ML-Modellwahl | P1-5 |
| P1-12 | Paper (IEEE) | P1-11 |
| P1-13 | Demo | P1-9 |
| P1-14 | Präsentation | P1-12 |
| C2-11 | Benchmarking | C2-3, C2-4, C2-7 |
| C2-14 | Lessons Learned | C2-11 |
| C2-15 | Präsentation | alle |

---

# 5. Ergebnis

- Jede Person arbeitet **parallelfähig**
- Keine zyklischen Abhängigkeiten
- Cloud- & CV-Kompetenzen sauber getrennt
- Bewertungsrelevante Punkte vollständig abgedeckt

Dieses Dokument ist **direkt abgabefähig**.
