
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import os, time
import matplotlib.pyplot as plt

# MediaPipe erst importieren, wenn tatsächlich benötigt (Lazy-Import)
def _load_mediapipe():
    import mediapipe as mp
    return mp

def get_pose_vector(image_path: str) -> Dict:
    """
    Erzeugt aus einem Bild einen Pose-Vektor: 33 Landmark-Punkte (x,y,z,visibility).
    Gibt zusätzlich Bildbreite/-höhe zurück.
    """
    mp = _load_mediapipe()
    mp_pose = mp.solutions.pose
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Konnte Bild nicht öffnen: {image_path}")
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, enable_segmentation=False) as pose:
        res = pose.process(image_rgb)

    if not res.pose_landmarks:
        raise RuntimeError("Keine Pose erkannt. Versuche ein anderes Foto (volle Person, gute Beleuchtung).")

    # 33 Landmarks
    landmarks = []
    for lm in res.pose_landmarks.landmark:
        landmarks.append({
            "x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility
        })

    return {"image_size": {"width": w, "height": h}, "landmarks": landmarks}

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Winkel ABC in Grad (mit B als Eckpunkt)."""
    ba = a - b
    bc = c - b
    # numerische Stabilität
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def _p2d(lm: Dict, img_w: int, img_h: int) -> np.ndarray:
    """Konvertiert normierte Landmark zu Pixelkoordinaten (2D)."""
    return np.array([lm["x"] * img_w, lm["y"] * img_h], dtype=float)

# Indizes gemäß MediaPipe Pose
LMS = {
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
}

def compute_basic_angles(pose: Dict) -> Dict[str, float]:
    """
    Berechnet einfache 2D-Winkel (Knie, Hüfte, Rücken, Schulter) in Grad.
    Nutzt linke Seite; fällt auf rechte zurück, wenn links unzuverlässig.
    """
    w = pose["image_size"]["width"]
    h = pose["image_size"]["height"]
    lms = pose["landmarks"]

    def lm(i: int) -> Dict: return lms[i]
    def good(i: int) -> bool: return lms[i]["visibility"] > 0.5

    # Wähle Seite mit besserer Sichtbarkeit
    side = "LEFT" if good(LMS["LEFT_KNEE"]) else "RIGHT"

    hip = _p2d(lm(LMS[f"{side}_HIP"]), w, h)
    knee = _p2d(lm(LMS[f"{side}_KNEE"]), w, h)
    ankle = _p2d(lm(LMS[f"{side}_ANKLE"]), w, h)
    shoulder = _p2d(lm(LMS[f"{side}_SHOULDER"]), w, h)
    elbow = _p2d(lm(LMS[f"{side}_ELBOW"]), w, h)
    wrist = _p2d(lm(LMS[f"{side}_WRIST"]), w, h)

    knee_angle = _angle(hip, knee, ankle)            # 180° = gestreckt, < 90° = tief gebeugt
    hip_angle = _angle(shoulder, hip, knee)          # kleiner = mehr Hüftbeugung
    shoulder_angle = _angle(elbow, shoulder, hip)    # grob Arm zum Oberkörper
    # Rücken-Neigung: Winkel des Oberkörpers relativ zur Vertikalen
    torso_vec = shoulder - hip
    back_tilt = _angle(hip + np.array([0, 1.0]), hip, shoulder)  # 0° ~ aufrecht, größer = nach vorn geneigt

    return {
        "side_used": side.lower(),
        "knee_angle_deg": round(float(knee_angle), 1),
        "hip_angle_deg": round(float(hip_angle), 1),
        "shoulder_angle_deg": round(float(shoulder_angle), 1),
        "back_tilt_deg": round(float(back_tilt), 1),
    }

def estimate_joint_moments(angles_deg: Dict[str, float], body_mass_kg: float, external_load_kg: float, exercise: str) -> Dict[str, float]:
    """
    Sehr vereinfachte, 2D-statische Momentenschätzung (N·m).
    - Nimmt effektive Hebelarme proportional zu Segmentlängen und sin(Winkel) an.
    - Nur zu Lern-/Vergleichszwecken.
    """
    g = 9.81
    total_weight_n = (body_mass_kg + external_load_kg) * g

    knee = max(0.0, min(angles_deg.get("knee_angle_deg", 0.0), 180.0))
    hip = max(0.0, min(angles_deg.get("hip_angle_deg", 0.0), 180.0))
    shoulder = max(0.0, min(angles_deg.get("shoulder_angle_deg", 0.0), 180.0))

    # grobe "effektive Hebelarme" in Metern (willkürliche Basen, nur für Demo)
    r_knee = 0.05 * math.sin(math.radians(180 - knee))    # 0 bei gestreckt
    r_hip = 0.06 * math.sin(math.radians(180 - hip))
    r_shoulder = 0.04 * math.sin(math.radians(180 - shoulder))

    M_knee = total_weight_n * r_knee
    M_hip = total_weight_n * r_hip
    M_shoulder = total_weight_n * r_shoulder if ("press" in exercise.lower() or "drück" in exercise.lower()) else 0.0

    return {
        "knee_moment_Nm": round(M_knee, 1),
        "hip_moment_Nm": round(M_hip, 1),
        "shoulder_moment_Nm": round(M_shoulder, 1),
    }

def _choose_side(landmarks):
    left_vis = landmarks[LMS["LEFT_KNEE"]]["visibility"]
    right_vis = landmarks[LMS["RIGHT_KNEE"]]["visibility"]
    return "LEFT" if left_vis >= right_vis else "RIGHT"

def draw_force_overlay(image_path: str, pose: Dict, moments: Dict[str, float], target_joint: str, out_dir: str = "output") -> str:
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(image_path)
    h, w = pose["image_size"]["height"], pose["image_size"]["width"]
    lms = pose["landmarks"]
    side = _choose_side(lms)
    def P(name):
        i = LMS[f"{side}_{name}"]
        return (int(lms[i]["x"]*w), int(lms[i]["y"]*h))
    HIP, KNEE, ANKLE = P("HIP"), P("KNEE"), P("ANKLE")
    SHO, ELB, WRI = P("SHOULDER"), P("ELBOW"), P("WRIST")
    def L(a,b,t=2,c=(220,220,220)): cv2.line(img,a,b,c,t,cv2.LINE_AA)
    L(SHO,HIP); L(HIP,KNEE); L(KNEE,ANKLE); L(SHO,ELB); L(ELB,WRI)

    tgt, mom = None, 0.0
    if target_joint=="knee": tgt,mom=KNEE,float(moments.get("knee_moment_Nm",0))
    elif target_joint=="hip": tgt,mom=HIP,float(moments.get("hip_moment_Nm",0))
    elif target_joint=="shoulder": tgt,mom=SHO,float(moments.get("shoulder_moment_Nm",0))

    if tgt and mom>0:
        scale = min(1.0, mom/300.0)
        length = int(60 + 140*scale)
        thick  = int(2 + 6*scale)
        dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]
        for dx,dy in dirs:
            end=(tgt[0]+int(dx*length), tgt[1]+int(dy*length))
            cv2.arrowedLine(img, end, tgt, (80,160,255), thick, tipLength=0.25)
        cv2.circle(img, tgt, 10, (0,140,255), -1, cv2.LINE_AA)
        cv2.putText(img, f"{target_joint} ~ {mom:.0f} N·m", (tgt[0]+12, tgt[1]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, f"{target_joint} ~ {mom:.0f} N·m", (tgt[0]+12, tgt[1]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    out_path = os.path.join(out_dir, f"force_overlay_{target_joint}_{time.strftime('%Y%m%d_%H%M%S')}.png")
    cv2.imwrite(out_path, img)
    return out_path

def draw_vector_body(pose: Dict, angles: Dict[str, float], moments: Dict[str, float], out_dir: str = "output") -> Dict[str, str]:
    """
    Zeichnet eine 'Stick Figure' aus den Landmarks (2D) + Winkel-Labels + Kräftepfeilen (knee/hip/shoulder ∝ Moment).
    Speichert als SVG (vektor) + PNG. Rückgabe: {"svg": "...", "png": "..."}.
    """
    os.makedirs(out_dir, exist_ok=True)
    w = pose["image_size"]["width"]; h = pose["image_size"]["height"]
    lms = pose["landmarks"]
    side = _choose_side(lms)

    def p(name):
        i = LMS[f"{side}_{name}"]
        return np.array([lms[i]["x"]*w, lms[i]["y"]*h], dtype=float)

    SHO, HIP, KNEE, ANKLE = p("SHOULDER"), p("HIP"), p("KNEE"), p("ANKLE")
    ELB, WRI = p("ELBOW"), p("WRIST")

    # Normalisierung auf eine Einheits-Box, damit die Figur immer hübsch skaliert ist
    pts = np.vstack([SHO, HIP, KNEE, ANKLE, ELB, WRI])
    minxy = pts.min(axis=0); maxxy = pts.max(axis=0)
    span = np.maximum(maxxy - minxy, 1)
    def N(v):  # 0..1
        return (v - minxy) / span

    SHO_, HIP_, KNEE_, ANKLE_, ELB_, WRI_ = map(N, [SHO, HIP, KNEE, ANKLE, ELB, WRI])

    fig, ax = plt.subplots(figsize=(4.5, 6))  # schmale Figur
    ax.set_axis_off()
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(1.15, -0.15)  # y invertiert wie Bildkoordinaten

    def L(a,b,lw=3):
        ax.plot([a[0],b[0]],[a[1],b[1]], linewidth=lw)

    # Skelett
    L(SHO_, HIP_); L(HIP_, KNEE_); L(KNEE_, ANKLE_)
    L(SHO_, ELB_); L(ELB_, WRI_)
    for P in [SHO_, HIP_, KNEE_, ANKLE_, ELB_, WRI_]:
        ax.scatter([P[0]],[P[1]], s=30)

    # Winkel-Labels
    ax.text(KNEE_[0]+0.02, KNEE_[1]-0.02, f"Knie: {angles.get('knee_angle_deg',0):.0f}°", fontsize=9)
    ax.text(HIP_[0]+0.02, HIP_[1]-0.02, f"Hüfte: {angles.get('hip_angle_deg',0):.0f}°", fontsize=9)
    ax.text(SHO_[0]+0.02, SHO_[1]-0.02, f"Schulter: {angles.get('shoulder_angle_deg',0):.0f}°", fontsize=9)
    ax.text(HIP_[0]-0.02, HIP_[1]-0.08, f"Back tilt: {angles.get('back_tilt_deg',0):.0f}°", fontsize=9)

    # Kräftepfeile (Länge ≈ Moment; Richtung radial zum Gelenk)
    def arrows(center, moment_nm, label):
        if moment_nm <= 0: return
        scale = min(1.0, moment_nm/300.0)
        Lpx = 0.15 + 0.35*scale  # Normallänge 0..0.5
        dirs = np.array([
            [ 1, 0], [-1, 0], [0, 1], [0,-1],
            [ 1, 1], [-1, 1], [1,-1], [-1,-1]
        ], dtype=float)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        for d in dirs:
            tail = center + d*Lpx
            ax.annotate("",
                xy=center, xytext=tail,
                arrowprops=dict(arrowstyle="->", lw=2))
        ax.text(center[0]+0.02, center[1]-0.06, f"{label}: {moment_nm:.0f} N·m", fontsize=9)

    arrows(KNEE_, float(moments.get("knee_moment_Nm",0)), "Knie")
    arrows(HIP_,  float(moments.get("hip_moment_Nm",0)),  "Hüfte")
    arrows(SHO_,  float(moments.get("shoulder_moment_Nm",0)), "Schulter")

    ts = time.strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(out_dir, f"vector_body_{ts}.png")
    svg_path = os.path.join(out_dir, f"vector_body_{ts}.svg")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return {"svg": svg_path, "png": png_path}