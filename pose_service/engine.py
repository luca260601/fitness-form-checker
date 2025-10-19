import math, numpy as np, cv2
from typing import Dict, Any

# --- Mediapipe Pose ---
def _load_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except Exception as e:
        raise RuntimeError("Mediapipe ist nicht installiert/kompatibel (nutze Python 3.11).") from e

def get_pose_vector(image_path: str) -> Dict[str, Any]:
    mp = _load_mediapipe()
    mp_pose = mp.solutions.pose
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, enable_segmentation=False) as pose:
        res = pose.process(image_rgb)
    if not res.pose_landmarks:
        raise RuntimeError("Keine Pose erkannt.")
    landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                 for lm in res.pose_landmarks.landmark]
    return {"image_size": {"width": w, "height": h}, "landmarks": landmarks}

# --- Landmarks Indices (MediaPipe) ---
LMS = {
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
}

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b; bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-9)
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def _choose_side_for_points(lms, points: list[str]) -> str:
    name_to_key = {
        "shoulder": ("LEFT_SHOULDER","RIGHT_SHOULDER"),
        "elbow":    ("LEFT_ELBOW","RIGHT_ELBOW"),
        "wrist":    ("LEFT_WRIST","RIGHT_WRIST"),
        "hip":      ("LEFT_HIP","RIGHT_HIP"),
        "knee":     ("LEFT_KNEE","RIGHT_KNEE"),
        "ankle":    ("LEFT_ANKLE","RIGHT_ANKLE"),
    }
    left_ids, right_ids = [], []
    for p in points:
        base = p.split("_")[0]
        if base in name_to_key:
            L, R = name_to_key[base]
            left_ids.append(LMS[L]); right_ids.append(LMS[R])
    L = sum(lms[i]["visibility"] for i in left_ids) if left_ids else 0
    R = sum(lms[i]["visibility"] for i in right_ids) if right_ids else 0
    return "LEFT" if L >= R else "RIGHT"

def _get_xy(pose: Dict, side: str, name: str) -> np.ndarray:
    w = pose["image_size"]["width"]; h = pose["image_size"]["height"]
    lms = pose["landmarks"]
    if name == "hip_down":
        i = LMS[f"{side}_HIP"]; lm = lms[i]
        return np.array([lm["x"]*w, lm["y"]*h + 1.0], dtype=float)
    key = {
        "shoulder":"SHOULDER","elbow":"ELBOW","wrist":"WRIST",
        "hip":"HIP","knee":"KNEE","ankle":"ANKLE"
    }.get(name)
    if not key: raise ValueError(f"Unknown point '{name}'")
    i = LMS[f"{side}_{key}"]; lm = lms[i]
    return np.array([lm["x"]*w, lm["y"]*h], dtype=float)

def compute_angles_config(pose: Dict, cfg: Dict) -> Dict[str, float]:
    needed = []
    for seg in cfg.get("segments", []): needed += seg
    for a in cfg.get("angles", []):     needed += a.get("points", [])
    side = _choose_side_for_points(pose["landmarks"], needed)
    out = {"__side__": side}
    for a in cfg.get("angles", []):
        A,B,C = (_get_xy(pose, side, a["points"][0]),
                 _get_xy(pose, side, a["points"][1]),
                 _get_xy(pose, side, a["points"][2]))
        out[f"{a['id']}_deg"] = round(float(_angle(A,B,C)), 1)
    return out

def estimate_moments_config(cfg: Dict, angles_deg: Dict[str, float], body_mass_kg: float, external_load_kg: float) -> Dict[str, float]:
    import math as _m
    def sin_deg(x): return _m.sin(_m.radians(x))
    def cos_deg(x): return _m.cos(_m.radians(x))
    g = 9.81
    env = {
        "g": g,
        "total_N": (body_mass_kg + external_load_kg) * g,
        "per_arm_N": (external_load_kg/2.0) * g,
        "sin_deg": sin_deg, "cos_deg": cos_deg,
        "angles": {k.replace("_deg",""): v for k,v in angles_deg.items() if k.endswith("_deg")}
    }
    out = {}
    for joint, expr in (cfg.get("moments") or {}).items():
        try: out[f"{joint}_moment_Nm"] = round(float(eval(expr, {"__builtins__": {}}, env)), 1)
        except Exception: out[f"{joint}_moment_Nm"] = 0.0
    return out
