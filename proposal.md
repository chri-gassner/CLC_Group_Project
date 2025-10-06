# Cloud-basierte Fitness-Tracking- und Analyseplattform

Dieses Projekt erweitert den bestehenden Fitness-Tracker (COV3) um eine Cloud-Komponente und eine Webplattform. Ziel ist es, Trainingsdaten zentral zu speichern, Fortschritte darzustellen und mithilfe von KI individuelle Trainingspläne zu generieren. Die Verbindung von lokaler Datenerfassung und Cloud-Verarbeitung schafft eine skalierbare, moderne Infrastruktur.

Der bestehende Tracker erkennt lokal mit MediaPipe die Körperbewegungen und zählt Wiederholungen. Diese Daten werden in Echtzeit an eine Cloud-API gesendet. In der Cloud werden sie gespeichert, analysiert und im Dashboard visualisiert. Nutzer sehen ihre Trainingshistorie, Fortschritte und erhalten automatisch generierte Pläne, die auf ihrem Leistungsniveau basieren. Die ChatGPT-API dient zur Erstellung personalisierter Empfehlungen und Trainingspläne.

Das System kombiniert Edge- und Cloud-Computing. Die Pose-Erkennung bleibt lokal, um Latenz zu minimieren, während Analyse, Aggregation und Visualisierung in der Cloud stattfinden. Dies ermöglicht Datensicherung, Geräteunabhängigkeit und langfristige Speicherung. Die Architektur besteht aus einem Edge-Client, einer Cloud-API, einer Datenbank, einer Analyseschicht und einem Web-Frontend.

Die Cloud dient als zentrale Plattform für Datenhaltung, Analyse und Benutzerinteraktion. Das Projekt verdeutlicht den praktischen Einsatz von Cloud-Technologien, Datenanalyse und KI in einem verteilten System. Damit erfüllt es die Anforderungen eines Cloud-Computing-Projekts, zeigt gleichzeitig einen klaren Bezug zur bestehenden Arbeit und erweitert sie um eine skalierbare, intelligente Ebene.
