from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import uuid
import os

app = FastAPI(title="Pose Service (MVP Stub)")

class AnalysisFlags(BaseModel):
    knee_over_toe_high: bool = False
    low_depth: bool = False
    torso_lean_high: bool = False

class AnalysisAngles(BaseModel):
    knee_min: float
    knee_max: float
    torso_max: float

class AnalysisResponse(BaseModel):
    angles: AnalysisAngles
    flags: AnalysisFlags
    notes: Optional[str] = None
    keyframes_base64: Optional[List[str]] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze_squat", response_model=AnalysisResponse)
async def analyze_squat(
    file: UploadFile = File(...),
    athlete_height_cm: Optional[float] = Form(None),
    fps: Optional[int] = Form(None),
):
    # Speichere Datei temporär (für später echte Analyse)
    tmp_name = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(tmp_name, "wb") as f:
        f.write(await file.read())

    # --- STUB-Analyse: feste Werte/Heuristiken als Platzhalter ---
    # Diese Werte werden später durch echte Pose-Schätzungen ersetzt.
    angles = AnalysisAngles(knee_min=60.0, knee_max=120.0, torso_max=35.0)
    flags = AnalysisFlags(
        knee_over_toe_high=False,  # später aus Dorsalflexion ableiten
        low_depth=False,           # später aus Hüfttiefe ggü. Knie ableiten
        torso_lean_high=False      # später aus Oberkörperwinkel ableiten
    )

    # Aufräumen
    try:
        os.remove(tmp_name)
    except OSError:
        pass

    return AnalysisResponse(
        angles=angles,
        flags=flags,
        notes="Stub-Analyse. Füge MediaPipe/YOLO-Pose hinzu, um echte Messwerte zu erhalten.",
        keyframes_base64=None,
    )
