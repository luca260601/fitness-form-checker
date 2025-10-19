import os, time, numpy as np, matplotlib.pyplot as plt, cv2
from typing import Dict
from .engine import LMS, _get_xy

def draw_force_overlay(image_path: str, pose: Dict, moments: Dict[str, float], target_joint: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(image_path)
    h, w = pose["image_size"]["height"], pose["image_size"]["width"]
    lms = pose["landmarks"]
    # einfache Seitenwahl
    side = "LEFT"
    if target_joint in ("elbow","shoulder","wrist"):
        L = sum(lms[LMS[k]]["visibility"] for k in ["LEFT_SHOULDER","LEFT_ELBOW","LEFT_WRIST"])
        R = sum(lms[LMS[k]]["visibility"] for k in ["RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST"])
        side = "LEFT" if L >= R else "RIGHT"
    else:
        L = sum(lms[LMS[k]]["visibility"] for k in ["LEFT_HIP","LEFT_KNEE","LEFT_ANKLE"])
        R = sum(lms[LMS[k]]["visibility"] for k in ["RIGHT_HIP","RIGHT_KNEE","RIGHT_ANKLE"])
        side = "LEFT" if L >= R else "RIGHT"

    def P(j): 
        idx = LMS[f"{side}_{j.upper()}"]; lm = lms[idx]
        return (int(lm["x"]*w), int(lm["y"]*h))

    SHO, HIP, KNEE, ANKLE, ELB, WRI = P("shoulder"), P("hip"), P("knee"), P("ankle"), P("elbow"), P("wrist")
    def L(a,b,t=2,c=(220,220,220)): cv2.line(img,a,b,c,t,cv2.LINE_AA)
    L(SHO,HIP); L(HIP,KNEE); L(KNEE,ANKLE); L(SHO,ELB); L(ELB,WRI)

    label = target_joint.capitalize()
    center = None; key = f"{target_joint}_moment_Nm"
    if target_joint == "knee": center = KNEE
    elif target_joint == "hip": center = HIP
    elif target_joint == "shoulder": center = SHO
    elif target_joint == "elbow": center = ELB
    mom = float(moments.get(key, 0.0))
    if center and mom > 0:
        scale = min(1.0, mom/(80.0 if target_joint=="elbow" else 300.0))
        Lpx = int(60 + 140*scale); thick = int(2 + 6*scale)
        dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]
        for dx,dy in dirs:
            end=(center[0]+int(dx*Lpx), center[1]+int(dy*Lpx))
            cv2.arrowedLine(img, end, center, (80,160,255), thick, tipLength=0.25)
        cv2.circle(img, center, 10, (0,140,255), -1, cv2.LINE_AA)
        cv2.putText(img, f"{label}: {mom:.0f} N·m", (center[0]+12, center[1]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, f"{label}: {mom:.0f} N·m", (center[0]+12, center[1]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    out_path = os.path.join(out_dir, f"force_overlay_{target_joint}_{time.strftime('%Y%m%d_%H%M%S')}.png")
    cv2.imwrite(out_path, img); return out_path

def draw_vector_body_config(cfg: Dict, pose: Dict, angles: Dict[str, float], moments: Dict[str, float], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    side = angles.get("__side__", "LEFT")
    pts = []
    for a,b in cfg.get("segments", []):
        pts += [_get_xy(pose, side, a), _get_xy(pose, side, b)]
    P = np.vstack(pts); minxy = P.min(axis=0); span = np.maximum(P.max(axis=0)-minxy, 1)
    def N(v): return (v - minxy)/span

    fig, ax = plt.subplots(figsize=(4.5,6))
    ax.set_axis_off(); ax.set_xlim(-0.1,1.1); ax.set_ylim(1.1,-0.1)

    # Segmente & Punkte
    seen = {}
    for a,b in cfg.get("segments", []):
        A, B = N(_get_xy(pose, side, a)), N(_get_xy(pose, side, b))
        ax.plot([A[0],B[0]],[A[1],B[1]], linewidth=5)
        for nm, Pn in [(a,A),(b,B)]:
            if nm not in seen:
                seen[nm]=Pn; ax.scatter([Pn[0]],[Pn[1]], s=35)

    # Winkel-Labels
    for a in cfg.get("angles", []):
        pid, label = a["id"], a.get("label", a["id"])
        B = N(_get_xy(pose, side, a["points"][1]))
        val = angles.get(f"{pid}_deg", 0.0)
        ax.text(B[0]+0.02, B[1]-0.02, f"{label}: {val:.0f}°", fontsize=10)

    # Pfeile
    for joint in cfg.get("overlays", {}).get("arrows_at", []):
        key = f"{joint}_moment_Nm"; mom = float(moments.get(key, 0.0))
        if mom <= 0: continue
        C = N(_get_xy(pose, side, joint if joint!="back" else "hip"))
        scale = min(1.0, mom/(80.0 if joint=="elbow" else 300.0))
        Lpx = 0.15 + 0.35*scale
        dirs = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]], dtype=float)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        for d in dirs:
            tail = C + d*Lpx
            ax.annotate("", xy=C, xytext=tail, arrowprops=dict(arrowstyle="->", lw=2))
        ax.text(C[0]+0.02, C[1]+0.06, f"{joint.capitalize()}: {mom:.0f} N·m", fontsize=10)

    ts = time.strftime("%Y%m%d_%H%M%S")
    svg_path = os.path.join(out_dir, f"vector_body_{ts}.svg")
    png_path = os.path.join(out_dir, f"vector_body_{ts}.png")
    fig.savefig(svg_path, bbox_inches="tight"); fig.savefig(png_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return {"svg": svg_path, "png": png_path}
