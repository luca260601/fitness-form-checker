# pose_service/engine.py
from __future__ import annotations
import math
import numpy as np
import cv2
from typing import Dict, Any, List, Optional

# --- MediaPipe Setup ---
def _load_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except Exception as e:
        raise RuntimeError("MediaPipe ist nicht installiert/kompatibel (Python 3.11).") from e

# Landmarks-Index (MediaPipe BlazePose full)
LMS = {
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,      "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,     "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,    "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,     "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
}

def get_pose_vector(image_path: str,
                    confidence_threshold: float = 0.5,
                    enhanced: bool = True,
                    **_kwargs) -> Dict[str, Any]:
    """Liest ein Bild, schÃ¤tzt Pose, gibt dict mit 'image_size' und 'landmarks'."""
    mp = _load_mediapipe()
    mp_pose = mp.solutions.pose

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=(2 if enhanced else 1),
        enable_segmentation=False,
        min_detection_confidence=(0.7 if enhanced else 0.5)
    ) as pose:
        res = pose.process(rgb)

    if not res.pose_landmarks:
        raise RuntimeError("Keine Pose erkannt.")

    lms = []
    for lm in res.pose_landmarks.landmark:
        lms.append({"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})

    out = {"image_size": {"width": w, "height": h}, "landmarks": lms}
    # einfache QualitÃ¤tsmetrik
    vis = [lm["visibility"] for lm in lms]
    out["quality_metrics"] = {
        "average_visibility": float(np.mean(vis)),
        "minimum_visibility": float(np.min(vis)),
        "pose_quality_score": float(sum(v >= confidence_threshold for v in vis) / len(vis)),
        "is_high_quality": bool(sum(v >= confidence_threshold for v in vis) / len(vis) >= 0.7),
        "high_confidence_count": int(sum(v >= confidence_threshold for v in vis)),
        "total_landmarks": len(vis),
    }
    return out

# ---- Hilfen (exportiert, weil Visualization sie importiert) ----
def _detect_exercise_type(landmarks: List[Dict]) -> str:
    """
    Erkennt Übungstyp basierend auf Beinstellung:
    - 'bilateral': Beide Füße parallel (Squat, Overhead Press)
    - 'unilateral': Ein Bein vorne, eins hinten (Lunges, Split Squat)
    
    Returns: 'bilateral' oder 'unilateral'
    """
    try:
        l_ankle = landmarks[LMS["LEFT_ANKLE"]]
        r_ankle = landmarks[LMS["RIGHT_ANKLE"]]
        
        # X-Position Differenz (normalisiert 0-1)
        x_diff = abs(l_ankle["x"] - r_ankle["x"])
        
        # Y-Position Differenz (normalisiert 0-1)  
        y_diff = abs(l_ankle["y"] - r_ankle["y"])
        
        # Bei Lunges: deutlicher X-Unterschied (>0.15) UND ähnliche Y-Höhe
        # Bei Squats: Füße nebeneinander, minimaler X-Unterschied
        if x_diff > 0.15 and y_diff < 0.1:
            return "unilateral"  # Lunges, Split Squat
        else:
            return "bilateral"   # Squat, Deadlift, Press
            
    except Exception:
        return "bilateral"  # Fallback

def _choose_optimal_side(landmarks: List[Dict], joints: List[str]) -> str:
    """
    Wählt beste Körperseite für bilaterale Übungen (Squat, Press).
    Basiert auf durchschnittlicher Landmark-Sichtbarkeit.
    
    Für unilaterale Übungen (Lunges) sollte stattdessen front/rear genutzt werden!
    """
    left = right = cnt = 0.0
    for j in joints:
        li = LMS.get(f"LEFT_{j.upper()}"); ri = LMS.get(f"RIGHT_{j.upper()}")
        if li is None or ri is None: continue
        try:
            left += float(landmarks[li]["visibility"])
            right += float(landmarks[ri]["visibility"])
            cnt += 1.0
        except Exception:
            pass
    if cnt == 0:
        return "LEFT"
    return "LEFT" if (left/cnt) >= (right/cnt) else "RIGHT"

def _get_enhanced_xy(pose: Dict[str, Any], side: str, joint: str) -> np.ndarray:
    """joint âˆˆ {shoulder,hip,knee,ankle,elbow,wrist, ...}"""
    idx = LMS.get(f"{side}_{joint.upper()}")
    if idx is None:
        raise ValueError(f"Joint {side}_{joint} unbekannt")
    lm = pose["landmarks"][idx]
    return np.array([float(lm["x"]), float(lm["y"])])

def _which_front_side(landmarks: List[Dict]) -> str:
    try:
        lx = float(landmarks[LMS["LEFT_ANKLE"]]["x"])
        rx = float(landmarks[LMS["RIGHT_ANKLE"]]["x"])
        return "LEFT" if lx > rx else "RIGHT"
    except Exception:
        return "LEFT"

def _resolve_xy_any(pose: Dict[str, Any], default_side: str, token: str) -> Optional[np.ndarray]:
    t = token.lower()
    if t == "vertical":
        return None
    if t.startswith("left_"):
        return _get_enhanced_xy(pose, "LEFT", t[5:])
    if t.startswith("right_"):
        return _get_enhanced_xy(pose, "RIGHT", t[6:])
    if t.startswith("front_") or t.startswith("rear_"):
        front = _which_front_side(pose["landmarks"])
        rear = "RIGHT" if front == "LEFT" else "LEFT"
        side = front if t.startswith("front_") else rear
        joint = t.split("_", 1)[1]
        return _get_enhanced_xy(pose, side, joint)
    return _get_enhanced_xy(pose, default_side, t)

def _angle_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    v1 = p1 - p2; v2 = p3 - p2
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    c = np.clip(np.dot(v1/n1, v2/n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def compute_angles_config(pose: Dict[str, Any], cfg: Dict[str, Any], use_3d: bool=False) -> Dict[str, float]:
    """
    Berechnet alle in cfg['angles'] definierten Winkel.
    
    WICHTIG: Erkennt automatisch Übungstyp:
    - Bilateral (Squat): Nutzt beste Körperseite (left/right)
    - Unilateral (Lunges): Nutzt front/rear basierend auf X-Position
    
    Returns: Dict mit Keys "<angle_id>_deg" und "__side__" / "__exercise_type__"
    """
    if not pose or "angles" not in cfg: return {}
    
    # 1) Übungstyp erkennen
    exercise_type = _detect_exercise_type(pose["landmarks"])
    
    # 2) Seite wählen (nur für bilaterale Übungen relevant)
    if exercise_type == "bilateral":
        side = _choose_optimal_side(pose["landmarks"], ["shoulder","hip","knee","ankle"])
    else:
        # Bei unilateralen Übungen: Wir brauchen front/rear, nicht left/right
        # Aber für Schulter/Oberkörper nutzen wir trotzdem die beste Seite
        side = _choose_optimal_side(pose["landmarks"], ["shoulder","hip"])
    
    out = {
        "__side__": side,
        "__exercise_type__": exercise_type
    }
    
    # 3) Alle Winkel berechnen
    for a in cfg.get("angles", []):
        try:
            aid = a["id"]; pts = a["points"]
            if not isinstance(pts, list) or len(pts)!=3: continue
            
            if pts[0] == "vertical":
                # Vertikalwinkel: z.B. Rumpfneigung
                b = _resolve_xy_any(pose, side, pts[1])
                c = _resolve_xy_any(pose, side, pts[2])
                if b is None or c is None: raise ValueError("vertical: null")
                v = c - b; u = np.array([0.0, -1.0])
                nu = np.linalg.norm(u); nv = np.linalg.norm(v)
                if nu==0 or nv==0: ang = 0.0
                else:
                    cosang = np.clip(np.dot(u,v)/(nu*nv), -1.0, 1.0)
                    ang = float(np.degrees(np.arccos(cosang)))
                out[f"{aid}_deg"] = ang
            else:
                # Standard 3-Punkt-Winkel
                p1 = _resolve_xy_any(pose, side, pts[0])
                p2 = _resolve_xy_any(pose, side, pts[1])
                p3 = _resolve_xy_any(pose, side, pts[2])
                ang = _angle_2d(p1, p2, p3)
                out[f"{aid}_deg"] = float(ang)
        except Exception as e:
            print(f"Warning: Could not compute angle {a.get('id')}: {e}")
            out[f"{a.get('id')}_deg"] = 0.0
    
    # 4) Zusatzwinkel: Trunk Inclination (nur wenn nicht schon vorhanden)
    if "trunk_deg" not in out and "trunk_inclination_deg" not in out:
        try:
            shoulder = _get_enhanced_xy(pose, side, "shoulder")
            hip = _get_enhanced_xy(pose, side, "hip")
            vertical = np.array([0.0, -1.0])
            trunk = shoulder - hip
            nu = np.linalg.norm(vertical); nv = np.linalg.norm(trunk)
            if nu>0 and nv>0:
                c = np.clip(np.dot(vertical/nu, trunk/nv), -1.0, 1.0)
                out["trunk_inclination_deg"] = float(np.degrees(np.arccos(c)))
        except Exception:
            pass
    
    return out