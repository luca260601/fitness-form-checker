# pose_service/combined_visualization.py
from __future__ import annotations
import os, time
from typing import Dict, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import cv2

from .engine import (
    LMS,
    _get_enhanced_xy,
    _choose_optimal_side,
    _resolve_xy_any,
    _detect_exercise_type,
)

COLORS = {
    "skeleton": "#1E3A8A",
    "joints": "#DC2626",
    "text": "#1F2937",
    "background": "#FFFFFF",
    "panel_bg": "#F9FAFB",
    "accent": "#3B82F6",
    "success": "#059669",
    "warning": "#D97706",
    "error": "#DC2626",
    "border": "#E5E7EB",
}

def create_combined_analysis_image(
    image_path: str,
    pose: dict,
    cfg: dict,
    angles: dict,
    moments: dict,
    profile_data: dict,
    out_dir: str,
    feedback_text: str = "",
) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    fig = plt.figure(figsize=(16, 10), facecolor=COLORS["background"], dpi=300)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.2, 1],
        hspace=0.1, wspace=0.15, left=0.05, right=0.95, top=0.92, bottom=0.08
    )
    ax_original = fig.add_subplot(gs[0, 0])
    ax_vector   = fig.add_subplot(gs[0, 1])

    _draw_original_with_overlay(ax_original, img, pose, cfg)
    _draw_vector_body_diagram(ax_vector, pose, cfg, angles)

    fig.suptitle(
        "Fitness Form Analysis - Professional Report",
        fontsize=24, fontweight="bold", color=COLORS["text"], y=0.97, ha="center"
    )
    fig.text(
        0.5, 0.94, f'Exercise: {cfg.get("name","Unknown")}',
        ha="center", va="top", fontsize=16, color=COLORS["accent"], style="italic"
    )
    _place_panel_titles(fig, ax_original, ax_vector)

    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"combined_analysis_{stamp}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=COLORS["background"], edgecolor="none")
    plt.close(fig)
    return out_path


# ------------------------- LINKES PANEL -------------------------

def _draw_original_with_overlay(ax, img, pose, cfg):
    over = _add_pose_overlay_to_image(img.copy(), pose)
    ax.imshow(cv2.cvtColor(over, cv2.COLOR_BGR2RGB))
    ax.axis("off")

def _add_pose_overlay_to_image(img, pose):
    """
    Overlay: unilateral → beide Beine (front+rear) + Oberkörper,
             bilateral → eine Seite.
    """
    h = pose["image_size"]["height"]
    w = pose["image_size"]["width"]
    lms = pose["landmarks"]
    exercise_type = _detect_exercise_type(lms)

    def get_px(j):
        try:
            lm = lms[LMS[j.upper()]]
            return (int(lm["x"] * w), int(lm["y"] * h))
        except Exception:
            return None

    joints: Dict[str, Tuple[int,int]] = {}
    connections = []

    if exercise_type == "unilateral":
        try:
            l_ankle_x = lms[LMS["LEFT_ANKLE"]]["x"]
            r_ankle_x = lms[LMS["RIGHT_ANKLE"]]["x"]
            front_side = "LEFT" if l_ankle_x > r_ankle_x else "RIGHT"
            rear_side  = "RIGHT" if front_side == "LEFT" else "LEFT"
        except Exception:
            front_side, rear_side = "LEFT", "RIGHT"

        for s in [front_side, rear_side]:
            for j in ["shoulder", "hip"]:
                name = f"{s}_{j}"
                px = get_px(name)
                if px: joints[name] = px

        for side, prefix in [(front_side, "front"), (rear_side, "rear")]:
            for j in ["hip", "knee", "ankle"]:
                px = get_px(f"{side}_{j}")
                if px: joints[f"{prefix}_{j}"] = px

        connections = [
            ("front_hip", "front_knee"),
            ("front_knee", "front_ankle"),
            ("rear_hip",  "rear_knee"),
            ("rear_knee", "rear_ankle"),
        ]
        for s in [front_side, rear_side]:
            sh, hp = f"{s}_shoulder", f"{s}_hip"
            if sh in joints and hp in joints:
                connections.append((sh, hp))
                break
    else:
        side = _choose_optimal_side(lms, ["shoulder", "hip", "knee", "ankle"])
        for j in ["shoulder","hip","knee","ankle","elbow","wrist"]:
            px = get_px(f"{side}_{j}")
            if px: joints[f"{side}_{j}"] = px
        connections = [
            (f"{side}_shoulder", f"{side}_hip"),
            (f"{side}_hip", f"{side}_knee"),
            (f"{side}_knee", f"{side}_ankle"),
            (f"{side}_shoulder", f"{side}_elbow"),
            (f"{side}_elbow", f"{side}_wrist"),
        ]

    for a, b in connections:
        if a in joints and b in joints:
            cv2.line(img, joints[a], joints[b], (30, 58, 138), 3, cv2.LINE_AA)
    for pos in joints.values():
        cv2.circle(img, pos, 4, (220, 38, 38), -1, cv2.LINE_AA)
        cv2.circle(img, pos, 6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


# ------------------------- RECHTES PANEL -------------------------

def _draw_vector_body_diagram(ax, pose, cfg, angles):
    """
    Vector Body Diagram
    - Perspektive default: "lateral" (keine künstliche Spreizung)
    - Optionaler Frontal-Tweak nur wenn __perspective__ == "frontal"
    - Label-Platzierung entlang der Winkelhalbierenden (Fallback radial)
    """
    side         = angles.get("__side__", "LEFT")
    exercise_type= angles.get("__exercise_type__", "bilateral")
    perspective  = angles.get("__perspective__", cfg.get("meta", {}).get("perspective", "lateral"))
    segs = cfg.get("segments", [])

    def resolve_segment_point(token: str) -> np.ndarray:
        if token.startswith(("left_","right_","front_","rear_")):
            return _resolve_xy_any(pose, side, token)
        if exercise_type == "unilateral":
            if token in ["shoulder","elbow","wrist"]:
                return _get_enhanced_xy(pose, side, token)
            try:
                return _resolve_xy_any(pose, side, f"front_{token}")
            except Exception:
                return _get_enhanced_xy(pose, side, token)
        return _get_enhanced_xy(pose, side, token)

    resolved_segs = []
    for a, b in segs:
        try:
            A = resolve_segment_point(a)
            B = resolve_segment_point(b)
            resolved_segs.append((A, B, a, b))
        except Exception as e:
            print(f"Segment {a}-{b} nicht auflösbar: {e}")

    if not resolved_segs:
        ax.text(0.5, 0.5, "No pose data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off"); return

    # Optional: Frontal-Spreizung auf echte Segmente anwenden
    tweaked = []
    if perspective == "frontal":
        trunk_lean = float(angles.get("rumpf_neigung_deg", angles.get("trunk_inclination_deg", 0.0)) or 0.0)
        lean = (trunk_lean / 90.0) * 0.10
        for A, B, na, nb in resolved_segs:
            A2, B2 = A.copy(), B.copy()
            if (("shoulder" in na and "hip" in nb) or ("hip" in na and "shoulder" in nb)):
                if "shoulder" in na: A2[0] += lean
                else:                B2[0] += lean
            if (("hip" in na and "knee" in nb) or ("knee" in na and "hip" in nb)):
                if "knee" in nb: B2[0] -= 0.05
                else:            A2[0] -= 0.05
            tweaked.append((A2, B2, na, nb))
    else:
        tweaked = resolved_segs

    # Punkte für Auto-Scaling
    pts = []
    for A, B, *_ in tweaked: pts += [A, B]
    P = np.vstack(pts).astype(float)
    pmin, pmax = P.min(axis=0), P.max(axis=0)
    center = (pmin + pmax) / 2.0
    half   = (pmax - pmin) / 2.0
    maxh   = float(np.max(half) or 1.0)
    scale  = 0.80 / (2 * maxh)

    def tp(p: np.ndarray) -> np.ndarray:
        return (p - center) * scale + np.array([0.5, 0.5])

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # invertiert, wie zuvor
    ax.set_aspect("equal")

    # Segmente + Gelenke zeichnen
    drawn_joints: Dict[str, np.ndarray] = {}
    for A, B, na, nb in tweaked:
        a2, b2 = tp(A), tp(B)
        drawn_joints[na] = a2; drawn_joints[nb] = b2
        ax.plot([a2[0], b2[0]], [a2[1], b2[1]], color=COLORS["skeleton"], linewidth=6, zorder=1)
    for pos in drawn_joints.values():
        ax.add_patch(Circle(pos, 0.02, facecolor=COLORS["joints"],
                            edgecolor="white", linewidth=2, zorder=3))

    # ------- Winkel-Labels sauber platzieren -------
    def _resolve_any(token: str) -> np.ndarray | None:
        if token == "vertical":  # kein Punkt, Sonderfall
            return None
        if token.startswith(("left_","right_","front_","rear_")):
            return _resolve_xy_any(pose, side, token)
        if exercise_type == "unilateral":
            if token in ["shoulder","elbow","wrist"]:
                return _get_enhanced_xy(pose, side, token)
            try:
                return _resolve_xy_any(pose, side, f"front_{token}")
            except Exception:
                return _get_enhanced_xy(pose, side, token)
        return _get_enhanced_xy(pose, side, token)

    for a in cfg.get("angles", []):
        try:
            t0, t1, t2 = a["points"]
            P0, P1, P2 = _resolve_any(t0), _resolve_any(t1), _resolve_any(t2)
            if P1 is None:  # ohne Vertex geht nichts
                continue

            v1 = tp(P1)  # Vertex in Axes-Koords
            # Richtung bestimmen: Winkelhalbierende (wenn beide Seiten echte Punkte)
            dir_vec = None
            if P0 is not None and P2 is not None:
                A, B, C = tp(P0), v1, tp(P2)
                u = A - B; v = C - B
                nu = np.linalg.norm(u); nv = np.linalg.norm(v)
                if nu > 1e-6 and nv > 1e-6:
                    u /= nu; v /= nv
                    bis = u + v
                    if np.linalg.norm(bis) > 1e-6:
                        dir_vec = bis / np.linalg.norm(bis)

            # Fallback: radial vom Diagrammzentrum
            if dir_vec is None:
                cx, cy = 0.5, 0.5
                dx, dy = v1[0] - cx, v1[1] - cy
                r = (dx*dx + dy*dy) ** 0.5 or 1.0
                dir_vec = np.array([dx/r, dy/r])

            offset = 0.08
            lx = np.clip(v1[0] + offset * dir_vec[0], 0.05, 0.95)
            ly = np.clip(v1[1] + offset * dir_vec[1], 0.05, 0.95)
            val = float(angles.get(f"{a['id']}_deg", 0.0) or 0.0)

            ax.annotate(
                f"{a.get('label', a['id'])}: {val:.1f}°",
                xy=(v1[0], v1[1]), xytext=(lx, ly),
                textcoords="data", fontsize=10, color=COLORS["text"], zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["accent"], alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1,
                                shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0")
            )
        except Exception as e:
            print(f"Winkel-Label {a.get('id')} nicht platzierbar: {e}")

    ax.axis("off")


# ------------------------- Zusatz (kompatibel) -------------------------

def _draw_angles_panel(ax, angles: Dict[str, Any]):
    ax.set_facecolor(COLORS["panel_bg"])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(COLORS["border"]); s.set_linewidth(1)
    data = [(k.replace("_deg","").replace("_"," ").title(), v)
            for k,v in angles.items() if k.endswith("_deg")]
    if not data:
        ax.text(0.5,0.5,"No angle data", ha='center', va='center', transform=ax.transAxes)
    else:
        y=0.85
        for name,val in data[:6]:
            ax.add_patch(patches.Rectangle((0.02,y-0.06),0.96,0.12,fc='white',ec=COLORS['border'],alpha=0.5,transform=ax.transAxes))
            ax.text(0.05,y,f"{name}:", transform=ax.transAxes, fontsize=10, color=COLORS['text'], fontweight='bold')
            if "trunk" in name.lower() and angles.get("__trunk_estimated__", False):
                ax.text(0.95,y,f"~{val:.1f}°", ha='right', transform=ax.transAxes, fontsize=10, color=COLORS['warning'], fontweight='bold')
            else:
                ax.text(0.95,y,f"{val:.1f}°", ha='right', transform=ax.transAxes, fontsize=10, color=COLORS['accent'])
            y-=0.16
    ax.set_title('Joint Angles', fontsize=14, fontweight='bold', color=COLORS['text'], pad=15)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

def _draw_quick_panel(ax, feedback_text: str):
    ax.set_facecolor(COLORS["panel_bg"])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(COLORS["border"]); s.set_linewidth(1)
    pros, cons = [], []
    if feedback_text:
        lines = feedback_text.splitlines()
        in_pros = in_cons = False
        for ln in lines:
            L = ln.strip().lower()
            if "was gut ist" in L: in_pros, in_cons = True, False; continue
            if "was zu verbessern ist" in L: in_pros, in_cons = False, True; continue
            if L.startswith("#") or "konkrete cues" in L or "belastung" in L:
                in_pros = in_cons = False; continue
            if ln.strip().startswith(("-", "*")):
                txt = ln.strip().lstrip("-*").strip().replace("**","")
                key = (txt.replace("–", "-").split(":")[0].split("-")[0].split("(")[0].split(",")[0].strip())
                if in_pros and len(pros) < 2 and len(key) > 3: pros.append(f"✔ {key}")
                if in_cons and len(cons) < 2 and len(key) > 3: cons.append(f"✘ {key}")
    if not pros and not cons:
        ax.text(0.5,0.5,"Generiere Analyse...",ha="center",va="center",transform=ax.transAxes)
    else:
        y=0.8
        for p in pros[:2]:
            ax.text(0.05,y,p,fontsize=10,color=COLORS['success'],fontweight='bold',transform=ax.transAxes); y-=0.2
        for c in cons[:2]:
            ax.text(0.05,y,c,fontsize=10,color=COLORS['error'],fontweight='bold',transform=ax.transAxes); y-=0.2
    ax.set_title('Quick Assessment', fontsize=14, fontweight='bold',color=COLORS['text'], pad=15)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

def _draw_quality_panel(ax, pose: Dict[str,Any]):
    ax.set_facecolor(COLORS['panel_bg'])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(COLORS['border']); s.set_linewidth(1)
    q = pose.get("quality_metrics", {})
    score = q.get("pose_quality_score", 0.0); avg = q.get("average_visibility", 0.0)
    text = "EXCELLENT" if score>0.8 else ("GOOD" if score>0.6 else ("FAIR" if score>0.3 else "POOR"))
    color = COLORS['success'] if score>0.8 else (COLORS['warning'] if score>0.6 else COLORS['error'])
    ax.add_patch(Circle((0.5,0.65),0.25,fc=color,ec=color,alpha=0.2,transform=ax.transAxes))
    ax.text(0.5,0.65,text,ha='center',va='center',transform=ax.transAxes,fontsize=12,fontweight='bold',color=color)
    ax.text(0.5,0.35,f"Quality Score: {score:.3f}",ha='center',va='center',transform=ax.transAxes,fontsize=10,color=COLORS['text'])
    ax.text(0.5,0.2,f"Visibility: {avg:.3f}",ha='center',va='center',transform=ax.transAxes,fontsize=10,color=COLORS['text'])
    ax.set_title('Pose Quality', fontsize=14, fontweight='bold', color=COLORS['text'], pad=15)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

def _draw_info_panel(ax, profile_data: Dict[str,Any]):
    ax.set_facecolor(COLORS['panel_bg'])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(COLORS['border']); s.set_linewidth(1)
    items = [("Name", profile_data.get("name","?")),
             ("Weight", f"{profile_data.get('body_mass_kg',0):.1f} kg"),
             ("Height", f"{profile_data.get('height_cm',0):.0f} cm")]
    y=0.85
    for k,v in items:
        ax.add_patch(patches.Rectangle((0.02,y-0.06),0.96,0.12,fc='white',ec=COLORS['border'],alpha=0.5,transform=ax.transAxes))
        ax.text(0.05,y,k,transform=ax.transAxes,fontsize=10,color=COLORS['text'],fontweight='bold')
        ax.text(0.95,y,str(v),ha='right',transform=ax.transAxes,fontsize=10,color=COLORS['accent'])
        y-=0.18
    ax.set_title('Profile Info', fontsize=14, fontweight='bold', color=COLORS['text'], pad=15)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

def _place_panel_titles(fig, ax_left, ax_right, y=0.895):
    pL = ax_left.get_position(fig); pR = ax_right.get_position(fig)
    xL = (pL.x0+pL.x1)/2; xR = (pR.x0+pR.x1)/2
    kw = dict(ha='center', va='bottom', fontsize=14, fontweight='bold', color=COLORS['text'])
    fig.text(xL, y, 'Original Image with Pose Analysis', **kw)
    fig.text(xR, y, 'Vector Body Diagram', **kw)

def _draw_moments_panel(ax, feedback_text: str):
    ax.set_facecolor(COLORS['panel_bg'])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color(COLORS['border']); spine.set_linewidth(1)
    pros, cons = [], []
    txt = (feedback_text or "").strip()
    if txt:
        lines = txt.splitlines()
        in_pros = in_cons = False
        for line in lines:
            s = line.strip(); low = s.lower()
            if "was gut ist" in low or "was gut" in low: in_pros, in_cons = True, False; continue
            if "was zu verbessern ist" in low or "zu verbessern" in low: in_pros, in_cons = False, True; continue
            if low.startswith("#") or "konkrete cues" in low or "belastung" in low: in_pros = in_cons = False; continue
            if s.startswith(("-", "*")):
                text = s[1:].strip().replace("**", "")
                key = text.split(":", 1)[0].split("â€“", 1)[0].split("(", 1)[0].split(",", 1)[0].strip()
                if len(key) > 30: key = key[:27] + "..."
                if in_pros and len(pros) < 2 and len(key) > 2: pros.append(f"✔ {key}")
                elif in_cons and len(cons) < 2 and len(key) > 2: cons.append(f"✘ {key}")
    if not pros and not cons:
        ax.text(0.5,0.5,'Generiere Analyse...', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color=COLORS['text'], style='italic')
    else:
        y = 0.80
        for p in pros[:2]:
            ax.text(0.05, y, (p[:45] + "..." if len(p) > 45 else p),
                    fontsize=9, color=COLORS['success'], fontweight='bold', transform=ax.transAxes, wrap=True)
            y -= 0.20
        for c in cons[:2]:
            ax.text(0.05, y, (c[:45] + "..." if len(c) > 45 else c),
                    fontsize=9, color=COLORS['error'], fontweight='bold', transform=ax.transAxes, wrap=True)
            y -= 0.20
    ax.set_title('Quick Assessment', fontsize=14, fontweight='bold', color=COLORS['text'], pad=15)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
