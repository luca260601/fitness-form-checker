# Architektur (MVP)

```mermaid
flowchart LR
  CLI[CLI-Client] -->|POST /analyze_squat (Video)| API[(FastAPI Pose-Service)]
  API -->|JSON: Winkel/Flags/Keyframes| CLI
  CLI -->|User-Nachricht + JSON + Keyframes| Assistant
  Assistant -->|Text-Feedback| CLI
```
