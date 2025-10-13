# Fitness Form Checker & Injury Prevention (MVP)

Ein Minimalprojekt für einen Squat-Form-Checker: FastAPI-Pose-Service (Stub) + CLI-Client. 
Später kommt der OpenAI Assistant (File Search + Functions) hinzu.

## Schnellstart

```bash
# 1) Klonen & Environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 2) Tests & Lint
pytest -q
ruff check .
black --check .

# 3) Service starten (Entwicklung)
uvicorn app.pose_service.main:app --reload

# 4) CLI-Beispiel (in zweitem Terminal)
python cli/client.py --video assets/sample_clips/dummy.mp4
```

> Hinweis: Die Analyse ist aktuell ein **Stub** (keine echte Pose-Schätzung).
> Für die echte Pose-Analyse fügt ihr später MediaPipe/YOLO-Pose hinzu.

## Struktur

```
.
├── app/
│   └── pose_service/
│       └── main.py
├── cli/
│   └── client.py
├── tests/
│   └── test_health.py
├── docs/
│   ├── assistant_prompt.md
│   ├── architecture.md
│   └── report_outline.md
├── assets/
│   └── sample_clips/
│       └── dummy.mp4 (Platzhalter)
├── .github/
│   ├── workflows/ci.yml
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
├── .pre-commit-config.yaml
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Team-Flow (zu zweit)

- **Branch-Strategie:** feature-branches (`feature/pose-service`, `feature/assistant-cli`) → PR → Review → `main`.
- **Konventionen:** Conventional Commits (z. B. `feat: add analyze endpoint`), PR-Template (Auto über GitHub).
- **Aufteilung:**
  - Person A: Pose-Service (Endpoints, später Pose-Logik)
  - Person B: CLI + Assistant-Integration (später)
  - Beide: Tests, Doku, Review

## Nächste Schritte

1. Repo auf GitHub anlegen und diesen Code pushen (siehe unten).
2. CI läuft (Lint + Tests). 
3. Danach: Assistant + Function-Schema aus `docs/assistant_prompt.md` in der OpenAI Console anlegen.
4. Pose-Analyse implementieren (MediaPipe/YOLO-Pose) und Response erweitern.

## GitHub anlegen & pushen (Beispiel mit gh CLI)

```bash
# in dem Ordner, in dem dieses Repo liegt
gh repo create <ORG-ODER-USER>/fitness-form-checker --public --source=. --remote=origin --push
# Falls ohne gh CLI:
# 1) Neues Repo im Browser erstellen
# 2) Dann:
git init
git add .
git commit -m "chore: bootstrap repo skeleton"
git branch -M main
git remote add origin git@github.com:<ORG-ODER-USER>/fitness-form-checker.git
git push -u origin main
```
