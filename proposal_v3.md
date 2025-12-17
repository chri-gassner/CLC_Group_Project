# Projektproposal: From CV Monolith to Cloud Microservices

## 1. Ziel des Projekts

Ziel dieses Projekts ist es, eine bestehende **monolithische Computer-Vision-Anwendung** in eine **cloud-native Microservice-Architektur** zu überführen und dabei den **konkreten Mehrwert von Cloud-Computing-Technologien** sichtbar und messbar zu machen.

Als Ausgangspunkt dient eine im CV-Kurs entwickelte Benchmark-Anwendung zum Vergleich von **MediaPipe und OpenPose**. In der ursprünglichen Version läuft diese Anwendung als einzelner Prozess: Videoaufnahme, Inferenz, Metrikberechnung, Speicherung und Visualisierung sind eng gekoppelt und lokal umgesetzt.

Der Fokus des Projekts liegt ausdrücklich **nicht auf der Verbesserung der Machine-Learning-Modelle**, sondern auf Cloud-Aspekten wie Architektur, Entkopplung, Skalierbarkeit, Beobachtbarkeit und CI/CD. Die CV-Logik dient dabei als realistische und fachlich sinnvolle Grundlage.

---

## 2. Ausgangssituation: Der Monolith

Im Ist-Zustand handelt es sich um eine klassische monolithische Python-Anwendung. Die Webcam wird lokal ausgelesen, MediaPipe bzw. OpenPose führen die Pose-Erkennung durch, anschließend werden Metriken wie FPS, Latenz und Konfidenz berechnet und direkt lokal geloggt bzw. visualisiert.

Diese Architektur ist funktional, weist jedoch klare Einschränkungen auf. Die Anwendung ist nicht skalierbar, da sie nur auf einem einzelnen Gerät läuft. Es gibt keine saubere Trennung von Zuständigkeiten, keine Vergleichbarkeit zwischen unterschiedlichen Runs oder Geräten und keine Möglichkeit zur systematischen Beobachtung oder Fehlerisolierung. Genau diese Limitierungen motivieren die Migration in eine Cloud-Architektur.

---

## 3. Zielarchitektur: Zerlegung in Cloud-Microservices

Im Projekt wird der bestehende Monolith funktional zerlegt und als verteiltes System neu aufgebaut. Dabei werden nicht die CV-Modelle selbst gesplittet, sondern die Verantwortlichkeiten rund um Inferenz, Datenannahme, Persistenz und Visualisierung sauber getrennt.

Die Inferenz verbleibt bewusst am Edge: Ein dockerisierter Edge-Client führt MediaPipe bzw. OpenPose lokal aus, berechnet ausschließlich Rohmetriken und sendet diese als strukturierte JSON-Telemetrie an die Cloud. Videodaten werden dabei nicht übertragen, um Bandbreite zu sparen und realistische Edge-Constraints abzubilden.

In der Cloud übernimmt ein dedizierter Ingestion-Service auf Basis von FastAPI (Cloud Run) die Annahme und Validierung dieser Metriken. Dieser Service entkoppelt Edge-Geräte vollständig von der Persistenzschicht und kann bei steigender Last automatisch skalieren.

Die Speicherung erfolgt zentral in Firestore. Dadurch werden Vergleiche über Zeit, Geräte und Modelle hinweg möglich. Ein separater Analytics- und Dashboard-Service (Streamlit auf Cloud Run) greift ausschließlich lesend auf diese Daten zu, aggregiert sie und visualisiert die Ergebnisse. Änderungen an Visualisierung oder Analyse erfordern somit kein Re-Deployment der Inferenz- oder Ingestion-Komponenten.

Alle Services werden containerisiert betrieben und unabhängig voneinander über eine CI/CD-Pipeline gebaut und deployed.

```mermaid
flowchart LR
  %% From Monolith to Microservices (Edge + Cloud)
  %% Paste into Markdown that supports Mermaid

  subgraph MONO[Ausgangslage: Monolith (lokal)]
    M[CV Benchmark App\nWebcam + Inferenz + Metriken + (lokal) Storage + UI]
  end

  subgraph EDGE[Edge Device (lokal, Docker)]
    EC[Edge Inference Client (Docker)\nMediaPipe / OpenPose\nBerechnet Rohmetriken (FPS, Latenz, CPU, Confidence)\nKein Video-Upload]
  end

  subgraph CLOUD[Google Cloud (Serverless)]
    ING[Ingestion Service (FastAPI)\nCloud Run\nValidierung • Auth (optional) • Rate Limit (optional)]
    DB[(Firestore)\nRaw Telemetry + Aggregates]
    DASH[Analytics & Dashboard (Streamlit)\nCloud Run\nAggregation • Vergleich • Visualisierung]
    OBS[Cloud Logging/Monitoring\nMetriken • Logs • Alerts (optional)]
  end

  subgraph CICD[CI/CD]
    CI[Pipeline (z.B. GitHub Actions)\nBuild • Test • Containerize • Deploy]
    REG[(Container Registry)]
  end

  %% Monolith baseline relation (conceptual)
  M -. Referenz/Baseline .-> EC

  %% Data flow
  EC -->|HTTPS JSON Telemetrie| ING
  ING -->|Write (Raw)| DB
  DB -->|Read| DASH

  %% Observability
  ING --> OBS
  DASH --> OBS

  %% CI/CD flows
  CI -->|Push Images| REG
  REG -->|Deploy| ING
  REG -->|Deploy| DASH

---

## 4. Vorteile der eingesetzten Cloud-Technologien

Der Einsatz von Cloud-Technologien bringt in diesem Szenario einen klaren Mehrwert. Durch die serverlose Ausführung auf Cloud Run ist keine manuelle Infrastrukturverwaltung notwendig, während gleichzeitig automatische Skalierung und Ausfallsicherheit gewährleistet sind. Mehrere Edge-Clients können parallel Benchmarks durchführen und ihre Ergebnisse zentral einspeisen, was mit dem ursprünglichen Monolithen nicht möglich wäre.

Die Entkopplung der einzelnen Verantwortlichkeiten erhöht die Wartbarkeit und Erweiterbarkeit des Systems erheblich. Zudem ermöglicht die Cloud eine systematische Beobachtbarkeit: API-Latenzen, Datenbank-Write-Zeiten und Fehlerquoten können gemessen und mit der eigentlichen CV-Inferenzlatenz verglichen werden. Damit wird Cloud-Overhead nicht nur akzeptiert, sondern explizit analysiert.

Die CI/CD-Pipeline sorgt für reproduzierbare Builds und saubere Deployments, was den Übergang von einer experimentellen Studienanwendung zu einer realistischen Cloud-Anwendung widerspiegelt.

---

## 5. Schwierigkeiten und Trade-offs

Die Umstellung auf eine verteilte Cloud-Architektur bringt bewusst zusätzliche Komplexität mit sich. Netzwerkkommunikation, Schema-Versionierung und verteiltes Debugging stellen neue Herausforderungen dar, die im monolithischen Setup nicht existierten.

Darüber hinaus entsteht durch die Entkopplung ein messbarer Cloud-Overhead in Form zusätzlicher Latenz. Dieser Nachteil wird nicht verborgen, sondern gezielt gemessen und den Vorteilen gegenübergestellt. Auch Logging und Monitoring über mehrere Services hinweg erfordern einen höheren operativen Aufwand.

Diese Aspekte sind integraler Bestandteil des Projekts, da sie zentrale Eigenschaften realer Cloud-Systeme widerspiegeln.

---

## 6. Ziel und Erkenntnisgewinn

Das Projekt soll beantworten, ab welchem Punkt eine Cloud-basierte Architektur gegenüber einem lokalen Monolithen Vorteile bietet. Insbesondere wird untersucht, wie hoch der Cloud-Overhead im Verhältnis zur CV-Inferenz ist, wie sich die Architektur bei mehreren gleichzeitig aktiven Edge-Clients verhält und inwiefern die Zerlegung in Services Wartbarkeit und Skalierbarkeit verbessert.

Der Erkenntnisgewinn ist damit klar architektonisch motiviert und nicht algorithmisch.

---

## 7. Bezug zur Lehrveranstaltung Cloud Computing

Das Projekt erfüllt die zentralen Anforderungen der Lehrveranstaltung, indem es eine bestehende monolithische Anwendung in eine Microservice-Architektur überführt, konsequent containerisiert, serverlose Cloud-Services nutzt, CI/CD einsetzt und ein verteiltes System mit messbarem Cloud-Mehrwert implementiert.

---

## 8. Meilensteine

Die Umsetzung erfolgt iterativ und orientiert sich an klar abgegrenzten technischen Zwischenschritten. Dadurch ist jederzeit eine lauffähige Teilarchitektur vorhanden, die erweitert und evaluiert werden kann.

| Meilenstein | Beschreibung | Ziel / Ergebnis | Termin |
|------------|--------------|----------------|--------|
| M1 | Cloud-Setup & Basis-Infrastruktur | GCP-Projekt, Container Registry, Basis-IaC eingerichtet | 30.12.2025 |
| M2 | Ingestion-Service | FastAPI-Service auf Cloud Run nimmt Telemetrie entgegen | 07.01.2026 |
| M3 | Dashboard-Grundgerüst | Streamlit-Dashboard visualisiert synthetische Vergleichsdaten | 14.01.2026 |
| M4 | Edge-Integration | Edge-Client sendet Live-Metriken an die Cloud | 21.01.2026 |
| M5 | Monitoring & Analyse | Messung von API-Latenz, DB-Write-Zeit und Fehlerquoten | 28.01.2026 |
| M6 | Finalisierung | Dokumentation, Demo und Präsentationsvorbereitung | 31.01.2026 |

---

## 9. Aufgabenverteilung

Die Aufgabenverteilung folgt der Architekturaufteilung und stellt sicher, dass Cloud-, Infrastruktur- und Edge-Aspekte parallel bearbeitet werden können.

| Teammitglied | Rolle | Verantwortungsbereich |
|-------------|------|----------------------|
| Christoph | Cloud Backend | Ingestion-Service, Cloud-Run-Deployment, API-Design |
| Simon | Infrastruktur & CI/CD | Firestore, Terraform, CI/CD-Pipelines |
| Marco | Edge & Visualisierung | CV-Integration (MediaPipe/OpenPose), Telemetrie, Dashboard |
