
# Fitness Form Assistant (Console / Python)

Ein kleines, einsteigerfreundliches Projekt, das ein Foto deiner Übung (z. B. Squat oder Überkopfdrücken) einliest, deinen Körper mit **MediaPipe** erkennt, daraus einen **Positionsvektor** (Pose-Landmarks) und **Winkel** berechnet, eine grobe **Gelenkbelastung** schätzt und dir mit einem **OpenAI‑Modell** personalisiertes Feedback gibt. Zusätzlich werden **Diagramme** lokal mit Matplotlib gezeichnet.

> ⚠️ **Hinweis**: Die Last-/Momenten-Schätzungen sind stark vereinfacht (statisch, 2D) und ersetzen **keine** professionelle Biomechanik, Diagnose oder Beratung. Nutze die Ergebnisse nur als grobe Orientierung.

## Schnellstart

1) **Python 3.11+** installieren.
2) Projekt entpacken und Terminal ins Projekt wechseln:
   ```bash
   cd fitness_form_assistant
   ```
3) (Empfohlen) **virtuelle Umgebung** erstellen:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
4) **Abhängigkeiten** installieren:
   ```bash
   pip install -r requirements.txt
   ```
5) OpenAI **API-Key** setzen (Account unter platform.openai.com).  
   Kopiere `.env.example` nach `.env` und trage deinen Key ein **ODER** exportiere ihn direkt:
   ```bash
   export OPENAI_API_KEY=sk-...   # Windows PowerShell: $Env:OPENAI_API_KEY='sk-...'
   ```
6) Lege ein Testfoto in den Ordner `input/` (du kannst diesen Ordner selbst anlegen) oder gib später den Pfad an.
7) **Starten**:
   ```bash
   python app.py
   ```

## Was passiert?

- Du gibst **Personendaten** (Gewicht, Größe, ggf. Stangen-/Zusatzgewicht) ein.
- Du wählst **Übung** und **Bildpfad**.
- `pose_tools.py` extrahiert mit **MediaPipe** einen **Vektor** aus 33 Körper-Landmarks und berechnet **Winkel** (Knie, Hüfte, Rücken, Schulter je nach Übung).
- Es wird eine **grobe Momentenschätzung** (z. B. Knie/Hüfte) berechnet und als **Balkendiagramm** gespeichert (`output/diagram_*.png`).
- Ein **OpenAI‑Modell** (Text) bekommt deine Winkel/Parameter und die eingebauten **Grundlagen-Notizen** (siehe `knowledge/*.md`) und erstellt präzises, freundliches **Form-Feedback**.

## OpenAI Plattform – optionale Pro‑Features

Du wolltest **File Search**, **Code Interpreter** und **Systemanweisungen** – das MVP nutzt bereits **Systemanweisungen** (System-Prompt).  
Für File Search & Code Interpreter findest du unten einen Erweiterungspfad (separat – bewusst optional, damit der Start einfach bleibt).

### Erweiterung A: File Search

- Ziel: Dem Modell zusätzliche Wissensdateien (z. B. `knowledge/`) über **Vector Stores** verfügbar machen.
- Vorgehen (kurz):
  1. Erstelle einen **Vector Store** und lade Dateien hoch.
  2. Verbinde den Vector Store mit deinem Assistant oder deinen Requests.
- Doku (offizielle OpenAI‑Seiten):
  - Assistants **File Search**: https://platform.openai.com/docs/assistants/tools/file-search
  - Quickstart / SDK: https://platform.openai.com/docs/quickstart/build-your-application
  - Migrationshinweis zu **Responses API**: https://platform.openai.com/docs/assistants/migration

> In diesem Starter ist File Search noch nicht aktiv, weil es für Einsteiger mehr Schritte (Vector Store, Upload, IDs) bedeutet. Du kannst es später leicht nachrüsten (siehe Kommentare in `app.py`).

### Erweiterung B: Code Interpreter

- Ziel: Diagramme **serverseitig** im OpenAI Code Interpreter erzeugen (statt lokal in Matplotlib).
- Doku:
  - Tools/Agents Überblick: https://platform.openai.com/docs/guides/agents-sdk
  - Changelog-Hinweis (Responses unterstützt Bild-/Datei-Outputs): https://platform.openai.com/docs/changelog/may-13th-2024

> In diesem Starter werden Charts lokal erzeugt. Für Einsteiger ist das robuster und planbar. Später kannst du auf den Code Interpreter umstellen (Beispiel-Snippets sind in `app.py` kommentiert).

## Eingebaute Wissens-Notizen

- `knowledge/squat_basics.md`: Technikgrundlagen, Sicherheits- und Coaching-Hinweise für Kniebeuge.
- `knowledge/overhead_press_basics.md`: Kurznotizen fürs Überkopfdrücken.

Diese Texte werden aktuell **lokal in den Prompt** gegeben. Mit File Search kannst du sie künftig als Retrieval-Wissen anbinden.

## Typische Stolpersteine

- **Pose wird nicht erkannt** → Bild ausreichend groß/hell? Person im Vollkörper sichtbar? Probier ein anderes Foto.
- **OpenAI‑Key fehlt** → `.env` korrekt? Oder Umgebungsvariable gesetzt?
- **MediaPipe Build**: Falls Installation auf deinem System zickt, versuche eine aktuelle Python‑Version und ein frisches virtuelles Environment.

Viel Spaß – und sag Bescheid, wenn du die Pro‑Features einschalten willst. :)
