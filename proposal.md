# Projektvorschlag: Cloud-basierte Fitness-Tracking- und Analyseplattform

## Ziel des Projekts

Das Ziel ist eine Cloud-basierte **Web-App** für mobiles Fitness-Tracking. Der Nutzer filmt sich direkt mit dem Smartphone-Browser, die Pose-Erkennung läuft vorzugsweise **on-device** in der Web-App, Wiederholungen werden lokal gezählt und Ereignisse in Echtzeit an die Cloud gesendet. Die Plattform speichert die Historie, analysiert Fortschritte und erzeugt KI-gestützte Trainingspläne.

## High-Level Ziel

Das Projekt soll eine vollständige Cloud-Architektur bereitstellen, die als Erweiterung des bestehenden lokalen Trackers fungiert. Die Plattform dient als historisches Tracking- und Analysesystem, das automatisch Fortschritte erfasst und visuell darstellt. Zusätzlich werden Trainingspläne generiert, die auf realen Nutzungsdaten basieren. Der Fokus liegt auf der Verbindung von Edge-Computing (lokale Pose-Erkennung) und Cloud-Computing (Datenanalyse, Speicherung, KI-gestützte Planung).

## Neuerungen und bestehende Komponenten

Bereits vorhanden ist COV3: lokale Übungserkennung und Wiederholungszählung. Neu entstehen eine **Progressive Web App (PWA)** für mobile Erfassung, ein skalierbares Cloud-Backend mit Auth, Storage und Analytics sowie eine KI-Schicht für Plan-Generierung. Die PWA nutzt MediaPipe Tasks im Browser (oder TF.js MoveNet) für on-device Pose; alternativ kann ein Server-Endpunkt Frames/Keypoints verarbeiten.

## High-Level Architektur

```mermaid
flowchart LR
  subgraph Client (Mobile Web-App)
    PWA[PWA: Kamera + Pose (on-device) + Rep-Zähler] -- events(JSON) --> APIGW
    PWA <-- auth/OIDC --> AUTH[Auth Provider]
    PWA <-- WebSocket/SSE --> RT[Realtime API]
  end

  subgraph Cloud
    APIGW[API Gateway] --> BE[FastAPI on Cloud Run]
    BE --> DB[(SQL/Firestore)]
    BE --> BUS[(Pub/Sub/EventBridge)]
    BE --> OBJ[Object Storage]
    BUS --> DWH[(BigQuery/Analytics)]
    DWH --> DASH[Dashboard Service]
    BE --> GPT[ChatGPT API]
    REG[Model Registry] --> BE
  end

  PWA <--> DASH
```

## Bezug zu Cloud Computing

Die Lösung nutzt eine **PWA** mit WebRTC/Media Capture für die mobile Kamera, **on-device Inferenz** zur Latenz- und Kostensenkung und ein **serverloses Backend** (Cloud Run/Lambda) hinter einem API Gateway. Authentifizierung erfolgt über OIDC/Identity Provider. Ereignisse werden als **Event Stream** in Pub/Sub/EventBridge verarbeitet und in ein **Data Warehouse** (BigQuery) für Analytics geschrieben. **Object Storage** archiviert optionale Rohdaten. **Realtime**-Kanäle (WebSocket/SSE) liefern Live-Feedback. Eine **Model Registry** verwaltet Modellversionen, die **ChatGPT-API** generiert Trainingspläne. Das Frontend wird über Firebase Hosting/Vercel bereitgestellt.

## Milestones

| Meilenstein | Inhalt                                                      | Geplanter Abschluss |
| ----------- | ----------------------------------------------------------- | ------------------- |
| M1          | Projektsetup, Architekturdefinition, CI/CD-Umgebung         | 10.10.              |
| M2          | Implementierung der Cloud-API und Datenbankanbindung        | 17.10.              |
| M3          | Integration des Edge-Trackers mit Cloud-API                 | 24.10.              |
| M4          | Entwicklung der KI-Komponente (ChatGPT API, Trainingspläne) | 31.10.              |
| M5          | Aufbau des Dashboards (Visualisierung der Daten)            | 07.11.              |
| M6          | Analytics Layer und automatisiertes Reporting               | 14.11.              |
| M7          | Abschluss, Dokumentation, Präsentation, Live-Demo           | 21.11.              |

## Aufgabenverteilung

| Teammitglied  | Verantwortungsbereich             | Beschreibung                                                                                                        |
| ------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Christoph** | Cloud-Backend und KI-Integration  | Architekturdesign, Implementierung der API, ChatGPT-Anbindung, Modell-Integration, CI/CD.                           |
| **Marco**     | Frontend und Datenanalyse         | Entwicklung des Dashboards, Datenvisualisierung, Weboberfläche, Anbindung an die Datenbank und den Analytics-Layer. |
| **Simon**     | Datenmanagement und Infrastruktur | Datenbankmodellierung, Speicherung, Deployment auf Cloud-Plattform, Logging, Monitoring und Sicherheit.             |

## Zusammenfassung

Dieses Projekt zeigt die Verbindung eines bestehenden KI-basierten Edge-Systems mit modernen Cloud-Technologien. Durch die Erweiterung um eine Cloud-Plattform wird aus einer lokalen Anwendung ein skalierbares, datengetriebenes System, das Trainingsfortschritte erfasst, auswertet und durch KI personalisierte Empfehlungen bereitstellt. Es erfüllt die zentralen Anforderungen an ein Cloud-Computing-Projekt, indem es reale Datenverarbeitung, Cloud-Architektur, Serverless-Technologie und KI-Integration in einem praxisorientierten Anwendungsszenario vereint.
