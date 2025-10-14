# Fitness Form Checker & Injury Prevention (MVP)

Ein Minimalprojekt für einen Squat-Form-Checker: FastAPI-Pose-Service (Stub) + CLI-Client. 
Später kommt der OpenAI Assistant (File Search + Functions) hinzu.

## Schnellstart

```bash
# 1) Klonen & Environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt



# 3) Service starten (Entwicklung)
python -m uvicorn app.pose_service.main_generic:app --reload

# 4) CLI-Beispiel (in zweitem Terminal)
python cli\client_multi.py --video "assets\sample_clips\Kniebeugen_Seitenansicht_trim.mp4" --exercise squat --fps 8
```

## Struktur

```
.
├── app/
│   ├── __init_.py
│   └── pose_service/
│       ├── main_generic.py
│       └── engine 
│           ├── registry.py
│           ├── types.py
│           ├── utils.py
│           ├── __init_.py
│           └── exercises
│                └── squat.py
├── cli/
│   └── client_multi.py
├── tests/
│   └── test_health.py
├── docs/
│   ├── assistant_prompt.md
│   ├── architecture.md
│   └── report_outline.md
├── assets/
│   └── sample_clips/
│       └── Kniebeugen_Seitenansicht_trim.mp4
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



