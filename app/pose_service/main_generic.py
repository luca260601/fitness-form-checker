from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import uuid, os, cv2, tempfile
import mediapipe as mp

from .engine.exercises.squat import SquatExercise
from .engine.types import AnalyzeResult
from .engine.utils import draw_overlay, encode_b64_img, angle, torso_angle  # angle/torso_angle aus utils

app = FastAPI(title="Multi-Exercise Pose Service (Squat MVP)")
mp_pose = mp.solutions.pose

class GenericResponse(BaseModel):
    exercise: str
    reps: int
    issues: List[Dict]
    metrics: Dict[str, float]
    notes: Optional[str] = None
    keyframes_base64: Optional[List[str]] = None

def landmarks_to_frame(lms) -> Dict[str, Tuple[float, float]]:
    # Enum sicher zu int casten
    return {i.name: (lms[int(i)].x, lms[int(i)].y) for i in mp_pose.PoseLandmark}

@app.post("/analyze", response_model=GenericResponse)
async def analyze(
    file: UploadFile = File(...),
    exercise_hint: Optional[str] = Form(None),
    fps: Optional[int] = Form(8),
):
    # --- Upload sicher im System-Temp speichern ---
    tmp_dir = tempfile.gettempdir()
    safe_name = os.path.basename(file.filename) or "upload.mp4"
    tmp = os.path.join(tmp_dir, f"{uuid.uuid4()}_{safe_name}")
    with open(tmp, "wb") as f:
        f.write(await file.read())

    try:
        # --- Video öffnen ---
        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            return GenericResponse(
                exercise=exercise_hint or "squat",
                reps=0, issues=[], metrics={},
                notes="Video konnte nicht geöffnet werden.",
                keyframes_base64=None
            )

        # --- Frames subsamplen ---
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(src_fps / (fps or 8))))
        frames: List[Dict[str, Tuple[float, float]]] = []

        # ---- Keyframe-Kandidaten
        best_bottom = None          # (img_bgr, points, knee_deg, torso_deg, side_label)
        best_knee_forward = None
        best_knee_forward_score = -1e9
        max_hip_y = -1.0

        def pick_side(lms):
            L = [lms[mp_pose.PoseLandmark.LEFT_HIP],
                 lms[mp_pose.PoseLandmark.LEFT_KNEE],
                 lms[mp_pose.PoseLandmark.LEFT_ANKLE]]
            R = [lms[mp_pose.PoseLandmark.RIGHT_HIP],
                 lms[mp_pose.PoseLandmark.RIGHT_KNEE],
                 lms[mp_pose.PoseLandmark.RIGHT_ANKLE]]
            sumL = sum(p.visibility for p in L)
            sumR = sum(p.visibility for p in R)
            return "LEFT" if sumL >= sumR else "RIGHT"

        with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
            i = 0
            ok, frame = cap.read()
            while ok:
                if i % step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = pose.process(rgb)
                    if res.pose_landmarks:
                        lms = res.pose_landmarks.landmark
                        frames.append(landmarks_to_frame(lms))

                        # --- Punkte für Overlay sammeln (nur eine Seite nutzen)
                        side = pick_side(lms)
                        idx = mp_pose.PoseLandmark
                        if side == "LEFT":
                            hip = (lms[int(idx.LEFT_HIP)].x, lms[int(idx.LEFT_HIP)].y)
                            knee = (lms[int(idx.LEFT_KNEE)].x, lms[int(idx.LEFT_KNEE)].y)
                            ankle = (lms[int(idx.LEFT_ANKLE)].x, lms[int(idx.LEFT_ANKLE)].y)
                            shoulder = (lms[int(idx.LEFT_SHOULDER)].x, lms[int(idx.LEFT_SHOULDER)].y)
                            heel = (lms[int(idx.LEFT_HEEL)].x, lms[int(idx.LEFT_HEEL)].y)
                            toe = (lms[int(idx.LEFT_FOOT_INDEX)].x, lms[int(idx.LEFT_FOOT_INDEX)].y)
                        else:
                            hip = (lms[int(idx.RIGHT_HIP)].x, lms[int(idx.RIGHT_HIP)].y)
                            knee = (lms[int(idx.RIGHT_KNEE)].x, lms[int(idx.RIGHT_KNEE)].y)
                            ankle = (lms[int(idx.RIGHT_ANKLE)].x, lms[int(idx.RIGHT_ANKLE)].y)
                            shoulder = (lms[int(idx.RIGHT_SHOULDER)].x, lms[int(idx.RIGHT_SHOULDER)].y)
                            heel = (lms[int(idx.RIGHT_HEEL)].x, lms[int(idx.RIGHT_HEEL)].y)
                            toe = (lms[int(idx.RIGHT_FOOT_INDEX)].x, lms[int(idx.RIGHT_FOOT_INDEX)].y)

                        # Winkel grob berechnen
                        knee_deg = angle(hip, knee, ankle)
                        torso_deg = torso_angle(hip, shoulder)

                        # Tiefster Punkt (größtes hip.y)
                        if hip[1] > max_hip_y:
                            max_hip_y = hip[1]
                            best_bottom = (frame.copy(),
                                           {"hip": hip, "knee": knee, "ankle": ankle,
                                            "shoulder": shoulder, "heel": heel, "toe": toe},
                                           knee_deg, torso_deg, side)

                        # Max. Knie über Zehen (relativ zur Fußlänge)
                        foot_len = abs(toe[0] - heel[0])
                        fdir = 1 if toe[0] >= heel[0] else -1
                        knee_forward = (knee[0] - toe[0]) * fdir
                        score = (knee_forward / foot_len) if foot_len > 0 else -1e9
                        if score > best_knee_forward_score:
                            best_knee_forward_score = score
                            best_knee_forward = (frame.copy(),
                                                 {"hip": hip, "knee": knee, "ankle": ankle,
                                                  "shoulder": shoulder, "heel": heel, "toe": toe},
                                                 knee_deg, torso_deg, side)
                i += 1
                ok, frame = cap.read()

        cap.release()

        if not frames:
            return GenericResponse(
                exercise="squat", reps=0, issues=[], metrics={},
                notes="Keine Landmarks erkannt – bitte Seitenansicht, Ganzkörper, gutes Licht.",
                keyframes_base64=None
            )

        # --- Analyse (issues + metrics)
        ex = SquatExercise()
        res: AnalyzeResult = ex.analyze(frames)
        issues = [vars(i) for i in res.issues]

        # --- Flags-Map für Overlay
        flags_map = {
            "knee_over_toe_high": any(i["code"] == "knee_over_toe_high" for i in issues),
            "low_depth":          any(i["code"] == "low_depth"          for i in issues),
            "torso_lean_high":    any(i["code"] == "torso_lean_high"    for i in issues),
        }

        # --- Keyframes rendern & encoden
        kf_list: List[str] = []
        for candidate in [best_bottom, best_knee_forward]:
            if candidate is None:
                continue
            img_bgr, pts, kdeg, tdeg, side = candidate
            draw_overlay(img_bgr, pts, kdeg, tdeg, flags_map, side_label=side)
            b64 = encode_b64_img(img_bgr)
            if b64:
                kf_list.append(b64)

        return GenericResponse(
            exercise=res.exercise,
            reps=res.reps,
            issues=issues,
            metrics=res.metrics,
            notes="OK",
            keyframes_base64=kf_list or None
        )

    except Exception as e:
        # kontrollierter Fehlerpfad statt 500
        return GenericResponse(
            exercise=exercise_hint or "squat",
            reps=0, issues=[], metrics={},
            notes=f"Analyze error: {e}",
            keyframes_base64=None
        )
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
