# Work Packages & Task Distribution  
## Project 1: Gesture Recognition  
## Project 2: Cloud-Native CV Benchmarking Platform

---

# 1. Gesamtübersicht aller Arbeitspakete

## Projekt 1 – Gesture Recognition

# Projekt 1 – Gesture Recognition  
## Detaillierte Arbeitspakete mit Expected Output

| ID | Arbeitspaket | Beschreibung | Abhängigkeiten | Expected Output |
|----|-------------|--------------|----------------|-----------------|
| P1-1 | Gesture-Definition | Festlegung von mindestens sechs klar unterscheidbaren Fitness-Gesten (statisch und/oder dynamisch), inklusive eindeutiger semantischer Bedeutung, Bewegungsablauf und Abgrenzung zueinander (z. B. Squat, Push-up, Jumping Jack). | – | Dokumentierte Gestenliste inkl. Beschreibung, Skizzen/Beispielbilder und Klassendefinition |
| P1-2 | Dataset-Analyse | Analyse des vorhandenen Bild- bzw. Video-Datasets hinsichtlich Datenqualität, Klassenverteilung, Kameraperspektiven, Occlusions und Eignung für Keypoint-basierte Gestenerkennung. | P1-1 | Analysebericht (Statistiken, Class Balance, identifizierte Probleme) |
| P1-3 | MediaPipe Pipeline | Implementierung einer vollständigen MediaPipe-basierten Pipeline zur Extraktion von Körper-Keypoints sowie Ableitung geeigneter Merkmale (z. B. Gelenkwinkel, Abstände, zeitliche Veränderungen). | P1-2 | Lauffähiger MediaPipe-Extraktor inkl. Feature-Vektoren pro Frame/Sequenz |
| P1-4 | OpenPose Pipeline | Implementierung einer OpenPose-basierten Pipeline zur Extraktion von Körper-Keypoints und Feature-Engineering, analog zur MediaPipe-Pipeline, zur Gewährleistung fairer Vergleichbarkeit. | P1-2 | Lauffähiger OpenPose-Extraktor inkl. Feature-Vektoren pro Frame/Sequenz |
| P1-5 | Gesture-Labeling | Zuordnung der extrahierten Keypoints und Feature-Sequenzen zu den definierten Gestenklassen; Erstellung konsistenter Labels für Training und Evaluation. | P1-3, P1-4 | Gelabelter Trainings- und Testdatensatz |
| P1-6 | ML-Modellwahl | Auswahl geeigneter Klassifikationsmodelle (z. B. SVM, k-NN, LSTM) inklusive theoretischer Begründung basierend auf Datenstruktur, Temporalität und Komplexität der Gesten. | P1-5 | Dokumentierte Modellentscheidung inkl. Begründung |
| P1-7 | Training MediaPipe | Training, Validierung und Hyperparameter-Tuning der ML-Modelle auf Basis der MediaPipe-Features; Analyse von Overfitting und Generalisierung. | P1-6 | Trainiertes MediaPipe-Modell inkl. gespeicherter Gewichte und Trainingslogs |
| P1-8 | Training OpenPose | Training, Validierung und Hyperparameter-Tuning der ML-Modelle auf Basis der OpenPose-Features unter identischen Bedingungen wie bei MediaPipe. | P1-6 | Trainiertes OpenPose-Modell inkl. gespeicherter Gewichte und Trainingslogs |
| P1-9 | Rep-Counter Logik | Entwicklung einer robusten Logik zur automatischen Wiederholungszählung von Fitnessübungen (z. B. Zustandsautomat oder Peak-Erkennung in Zeitreihen). | P1-7, P1-8 | Funktionierende Wiederholungszählung pro Übung |
| P1-10 | Evaluation | Quantitative Evaluation der Modelle mittels Precision, Recall, Confusion Matrices und Accuracy für alle Gestenklassen. | P1-7, P1-8 | Evaluationsbericht inkl. Metriken und Confusion Matrices |
| P1-11 | Framework-Vergleich | Systematischer Vergleich von MediaPipe und OpenPose hinsichtlich Erkennungsgenauigkeit, Latenz, Ressourcenverbrauch (CPU/RAM) und Stabilität. | P1-10 | Vergleichstabelle und Diagramme (Accuracy, FPS, Resource Usage) |
| P1-12 | Paper (IEEE) | Erstellung eines wissenschaftlichen Papers im IEEE-Format inklusive Methodik, Ergebnissen, Diskussion und Ausblick. | P1-11 | Finales IEEE-konformes Paper (PDF) |
| P1-13 | Demo | Umsetzung einer Live-Demo mit Webcam, Gestenerkennung und Wiederholungszählung zur praktischen Verifikation des Systems. | P1-9 | Lauffähige Live-Demo-Anwendung |
| P1-14 | Präsentation | Erstellung und Durchführung einer strukturierten Präsentation (Slides + Vortrag) mit Fokus auf Motivation, Methodik, Ergebnisse und Live-Demo. | P1-12 | Finale Präsentationsfolien und gehaltene Präsentation |

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
