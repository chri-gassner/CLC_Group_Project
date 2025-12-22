# Work Packages & Task Distribution  
# 1. Gesamtübersicht aller Arbeitspakete
## Projekt 1 – Gesture Recognition
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
## Detaillierte Arbeitspakete mit Expected Output

| ID | Arbeitspaket | Beschreibung | Abhängigkeiten | Expected Output |
|----|-------------|--------------|----------------|-----------------|
| C2-1 | Architekturdesign | Entwurf einer skalierbaren, ereignisgesteuerten Cloud-Architektur (Edge, Ingestion, Messaging, Processing, Storage, Presentation) inklusive klarer Komponentenabgrenzung. | – | Architekturdiagramm (z. B. Draw.io/Mermaid) + textuelle Architektur­beschreibung |
| C2-2 | API-Spezifikation | Definition eines einheitlichen JSON-Schemas für Metriken (Latenz, FPS, Accuracy, Ressourcenverbrauch), inkl. Versionierung und Validierungsregeln. | C2-1 | Dokumentierte API-Spezifikation (Schema + Beispielpayloads) |
| C2-3 | Edge-Service MediaPipe | Implementierung eines Docker-Containers für lokale MediaPipe-Inferenz inkl. Metrikerfassung und Versand an die Cloud. | P1-3 | Lauffähiger Docker-Container (MediaPipe Edge-Service) |
| C2-4 | Edge-Service OpenPose | Implementierung eines Docker-Containers für lokale OpenPose-Inferenz inkl. Metrikerfassung und Versand an die Cloud. | P1-4 | Lauffähiger Docker-Container (OpenPose Edge-Service) |
| C2-5 | Ingestion API | Entwicklung einer FastAPI als skalierbarer Entry Point auf Cloud Run zur Entgegennahme von Metrikdaten. | C2-2 | Deployed Cloud-Run-Service mit dokumentierten Endpunkten |
| C2-6 | Pub/Sub Setup | Einrichtung von Google Pub/Sub Topics, Subscriptions und IAM-Rollen zur Entkopplung von Ingestion und Processing. | C2-1 | Funktionsfähige Pub/Sub-Infrastruktur |
| C2-7 | Processing Function | Implementierung einer Cloud Function, die Pub/Sub Events verarbeitet, validiert und in Firestore persistiert. | C2-6 | Deployte Cloud Function mit erfolgreichem Firestore-Write |
| C2-8 | Firestore Schema | Design einer flexiblen NoSQL-Datenstruktur zur Speicherung von Benchmark- und Inferenzmetriken. | C2-2 | Dokumentiertes Firestore-Datenmodell |
| C2-9 | Dashboard | Entwicklung eines Streamlit-Dashboards zur Live-Visualisierung von Metriken (Latenz, FPS, Accuracy). | C2-7, C2-8 | Lauffähiges Dashboard mit Live- oder Near-Realtime-Daten |
| C2-10 | IaC | Bereitstellung der Cloud-Infrastruktur mittels Terraform (Cloud Run, Pub/Sub, IAM). | C2-6 | Versioniertes Terraform-Setup inkl. `terraform apply` |
| C2-11 | Benchmarking | Systematischer Vergleich von Edge- vs. Cloud-Verarbeitung sowie synchroner vs. asynchroner Architektur. | C2-3, C2-4, C2-7 | Benchmark-Report mit Tabellen und Diagrammen |
| C2-12 | GitHub Dokumentation | Strukturierte Projektdokumentation inkl. Architektur, Setup, Build- und Run-Anleitung. | alle | Vollständiges GitHub-Repository mit README.md |
| C2-13 | Tutorial | Schritt-für-Schritt-Anleitung zur Reproduzierbarkeit des gesamten Systems. | C2-12 | Reproduzierbares Tutorial (Markdown) |
| C2-14 | Lessons Learned | Zusammenfassung technischer, architektonischer und organisatorischer Erkenntnisse. | C2-11 | Dokumentierte Lessons Learned Sektion |
| C2-15 | Präsentation | Erstellung und Durchführung einer Abschlusspräsentation inkl. Live-Demo des Systems. | alle | Finale Präsentationsfolien + funktionierende Live-Demo |

---
