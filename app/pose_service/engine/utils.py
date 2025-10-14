import math
from typing import Dict,Tuple
import base64 
import cv2


def angle(a: Tuple[float,float], b: Tuple[float,float], c: Tuple[float,float]) -> float:
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cosang = max(-1.0, min(1.0, dot/(n1*n2)))
    return math.degrees(math.acos(cosang))

def torso_angle(hip: Tuple[float,float], shoulder: Tuple[float,float]) -> float:
    # 0° = aufrecht; größer = mehr Vorlage
    vx, vy = shoulder[0]-hip[0], shoulder[1]-hip[1]
    dot = (vx*0.0) + (vy*-1.0)   # gegen (0,-1) = vertikal nach oben
    n = math.hypot(vx, vy)
    if n == 0:
        return float("nan")
    cosang = max(-1.0, min(1.0, dot / n))
    return math.degrees(math.acos(cosang))

def to_px(pt, w, h):
    return (int(pt[0] * w), int(pt[1] * h))

def draw_overlay(
    frame_bgr,
    points: Dict[str, Tuple[float, float]],
    knee_deg: float,
    torso_deg: float,
    flags: Dict[str, bool],
    side_label: str = ""
):
    """Zeichnet Gelenke/Linien + Winkel + gesetzte Flags ins Bild."""
    h, w = frame_bgr.shape[:2]
    def P(name): return to_px(points[name], w, h)

    # Linien
    for a, b in [("hip","knee"), ("knee","ankle"), ("hip","shoulder"), ("heel","toe")]:
        if a in points and b in points:
            cv2.line(frame_bgr, P(a), P(b), (255,255,255), 2)

    # Punkte
    for k in ["hip","knee","ankle","shoulder","heel","toe"]:
        if k in points:
            cv2.circle(frame_bgr, P(k), 5, (255,255,255), -1)

    # Beschriftung
    y = 24
    if side_label:
        cv2.putText(frame_bgr, f"{side_label}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2); y += 26
    cv2.putText(frame_bgr, f"knee={knee_deg:.1f} deg", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2); y += 26
    cv2.putText(frame_bgr, f"torso={torso_deg:.1f} deg", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2); y += 26

    # Flags
    for code, active in flags.items():
        if active:
            cv2.putText(frame_bgr, f"FLAG: {code}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            y += 24

def encode_b64_img(img_bgr) -> str | None:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return None
    return base64.b64encode(buf).decode("ascii")