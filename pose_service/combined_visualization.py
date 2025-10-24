# pose_service/combined_visualization.py
from __future__ import annotations
import os, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import cv2
from typing import Dict, Tuple, Any
from .engine import LMS, _get_enhanced_xy, _choose_optimal_side, _resolve_xy_any, _detect_exercise_type

COLORS = {
    'skeleton': '#1E3A8A', 'joints': '#DC2626',
    'text': '#1F2937', 'background': '#FFFFFF', 'panel_bg': '#F9FAFB',
    'accent': '#3B82F6', 'success': '#059669', 'warning': '#D97706',
    'error': '#DC2626', 'border': '#E5E7EB'
}

def create_combined_analysis_image(
    image_path: str,
    pose: dict,
    cfg: dict,
    angles: dict,
    moments: dict,
    profile_data: dict,
    out_dir: str,
    feedback_text: str = ""
) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    fig = plt.figure(figsize=(20, 14), facecolor=COLORS['background'], dpi=300)
    gs = fig.add_gridspec(3, 4, height_ratios=[2.5,2.5,1.2], width_ratios=[1.2,1.2,1,1],
                          hspace=0.2, wspace=0.15, left=0.05, right=0.95, top=0.88, bottom=0.08)

    ax_original = fig.add_subplot(gs[0:2, 0:2])
    ax_vector = fig.add_subplot(gs[0:2, 2:4])
    ax_angles  = fig.add_subplot(gs[2, 0])
    ax_quick   = fig.add_subplot(gs[2, 1])
    ax_quality = fig.add_subplot(gs[2, 2])
    ax_info    = fig.add_subplot(gs[2, 3])

    _draw_original_with_overlay(ax_original, img, pose, cfg)
    _draw_vector_body_diagram(ax_vector, pose, cfg, angles)
    _draw_angles_panel(ax_angles, angles)
    _draw_quick_panel(ax_quick, feedback_text)
    _draw_quality_panel(ax_quality, pose)
    _draw_info_panel(ax_info, profile_data)

    fig.suptitle('Fitness Form Analysis - Professional Report', fontsize=24, fontweight='bold',
                 color=COLORS['text'], y=0.97, ha='center')
    fig.text(0.5, 0.94, f'Exercise: {cfg.get("name","Unknown")}', ha='center', va='top',
             fontsize=16, color=COLORS['accent'], style='italic')

    _place_panel_titles(fig, ax_original, ax_vector)

    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f'combined_analysis_{stamp}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor=COLORS['background'], edgecolor='none')
    plt.close(fig)
    return out_path

def _draw_original_with_overlay(ax, img, pose, cfg):
    over = _add_pose_overlay_to_image(img.copy(), pose)
    ax.imshow(cv2.cvtColor(over, cv2.COLOR_BGR2RGB))
    ax.axis('off')

def _add_pose_overlay_to_image(img, pose):
    """
    Zeichnet Pose-Overlay auf Original-Bild.
    
    NEUE LOGIK:
    - Bilateral (Squat): Zeige eine Körperseite
    - Unilateral (Lunges): Zeige BEIDE Beine (front + rear)
    """
    h = pose["image_size"]["height"]
    w = pose["image_size"]["width"]
    lms = pose["landmarks"]
    
    # Erkenne Übungstyp
    exercise_type = _detect_exercise_type(lms)
    
    # Hilfsfunktion: Hole Pixel-Position
    def get_px(joint_name: str):
        try:
            idx = LMS[joint_name.upper()]
            lm = lms[idx]
            return (int(lm["x"] * w), int(lm["y"] * h))
        except Exception:
            return None
    
    joints = {}
    connections = []
    
    if exercise_type == "unilateral":
        # LUNGES: Zeige beide Beine + Oberkörper
        # Bestimme front/rear
        try:
            l_ankle_x = lms[LMS["LEFT_ANKLE"]]["x"]
            r_ankle_x = lms[LMS["RIGHT_ANKLE"]]["x"]
            front_side = "LEFT" if l_ankle_x > r_ankle_x else "RIGHT"
            rear_side = "RIGHT" if front_side == "LEFT" else "LEFT"
        except Exception:
            front_side, rear_side = "LEFT", "RIGHT"
        
        # Oberkörper (wähle sichtbarere Seite)
        for s in [front_side, rear_side]:
            for j in ["shoulder", "hip"]:
                name = f"{s}_{j}"
                px = get_px(name)
                if px:
                    joints[name] = px
        
        # BEIDE Beine
        for side, prefix in [(front_side, "front"), (rear_side, "rear")]:
            for j in ["hip", "knee", "ankle"]:
                name = f"{side}_{j}"
                px = get_px(name)
                if px:
                    joints[f"{prefix}_{j}"] = px
        
        # Connections für Lunges
        connections = [
            ("front_hip", "front_knee"),
            ("front_knee", "front_ankle"),
            ("rear_hip", "rear_knee"),
            ("rear_knee", "rear_ankle"),
        ]
        
        # Oberkörper-Verbindung
        for s in [front_side, rear_side]:
            sh = f"{s}_shoulder"
            hp = f"{s}_hip"
            if sh in joints and hp in joints:
                connections.append((sh, hp))
                break
    else:
        # BILATERAL (Squat): Eine Seite
        side = _choose_optimal_side(lms, ["shoulder", "hip", "knee", "ankle"])
        
        for j in ["shoulder", "hip", "knee", "ankle", "elbow", "wrist"]:
            name = f"{side}_{j}"
            px = get_px(name)
            if px:
                joints[name] = px
        
        connections = [
            (f"{side}_shoulder", f"{side}_hip"),
            (f"{side}_hip", f"{side}_knee"),
            (f"{side}_knee", f"{side}_ankle"),
            (f"{side}_shoulder", f"{side}_elbow"),
            (f"{side}_elbow", f"{side}_wrist"),
        ]
    
    # Zeichne Linien
    for a, b in connections:
        if a in joints and b in joints:
            cv2.line(img, joints[a], joints[b], (30, 58, 138), 6, cv2.LINE_AA)
    
    # Zeichne Gelenke
    for pos in joints.values():
        cv2.circle(img, pos, 10, (220, 38, 38), -1, cv2.LINE_AA)
        cv2.circle(img, pos, 14, (255, 255, 255), 2, cv2.LINE_AA)
    
    return img

def _draw_vector_body_diagram(ax, pose, cfg, angles):
    """
    Zeichnet Vector Body Diagram mit FLEXIBLER Segment-Auflösung.
    
    NEUE LOGIK:
    1. Segments können generic sein: ["hip", "knee"] 
       → Wird automatisch zu left_hip-left_knee ODER front_hip-front_knee
    2. Oder spezifisch: ["front_hip", "front_knee"]
       → Wird direkt genutzt
    3. Übungstyp aus angles nutzen für Entscheidung
    """
    side = angles.get("__side__", "LEFT")
    exercise_type = angles.get("__exercise_type__", "bilateral")
    segs = cfg.get("segments", [])
    
    # HILFSFUNKTION: Löse generic/spezifische Segmente auf
    def resolve_segment_point(token: str, side: str, exercise_type: str) -> np.ndarray:
        """
        Löst ein Segment-Token auf:
        - "hip" → je nach exercise_type: "left_hip" oder "front_hip"
        - "front_hip" → direkt "front_hip"
        - "left_hip" → direkt "left_hip"
        """
        # Bereits spezifisch?
        if token.startswith(("left_", "right_", "front_", "rear_")):
            return _resolve_xy_any(pose, side, token)
        
        # Generic → je nach Übungstyp
        if exercise_type == "unilateral":
            # Bei Lunges: Unterscheide Ober-/Unterkörper
            if token in ["shoulder", "elbow", "wrist"]:
                # Oberkörper: Nutze side (left/right)
                return _get_enhanced_xy(pose, side, token)
            else:
                # Unterkörper: Versuche FRONT zuerst, dann side
                try:
                    return _resolve_xy_any(pose, side, f"front_{token}")
                except Exception:
                    return _get_enhanced_xy(pose, side, token)
        else:
            # Bei bilateralen Übungen (Squat): Nutze side
            return _get_enhanced_xy(pose, side, token)
    
    # Sammle alle Punkte für Auto-Scaling
    pts = []
    resolved_segs = []  # [(pt_A, pt_B, name_A, name_B), ...]
    
    for a, b in segs:
        try:
            pt_a = resolve_segment_point(a, side, exercise_type)
            pt_b = resolve_segment_point(b, side, exercise_type)
            pts.extend([pt_a, pt_b])
            resolved_segs.append((pt_a, pt_b, a, b))
        except Exception as e:
            print(f"Segment {a}-{b} nicht auflösbar: {e}")
            continue
    
    if not pts:
        ax.text(0.5, 0.5, "No pose data", ha="center", va="center", transform=ax.transAxes)
        ax.axis('off')
        return
    
    # Auto-Scaling: Zentriere und normalisiere
    P = np.vstack(pts).astype(float)
    pmin = P.min(axis=0)
    pmax = P.max(axis=0)
    center = (pmin + pmax) / 2
    half = (pmax - pmin) / 2
    maxh = float(np.max(half) or 1.0)
    scale = 0.80 / (2 * maxh)
    
    def tp(p):
        return (p - center) * scale + np.array([0.5, 0.5])
    
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect('equal')
    
    # Zeichne Segmente
    drawn_joints = {}
    for pt_a, pt_b, name_a, name_b in resolved_segs:
        A = tp(pt_a)
        B = tp(pt_b)
        drawn_joints[name_a] = A
        drawn_joints[name_b] = B
        ax.plot([A[0], B[0]], [A[1], B[1]], color=COLORS['skeleton'], linewidth=6)
    
    # Zeichne Gelenk-Punkte
    for pos in drawn_joints.values():
        ax.add_patch(Circle(pos, 0.02, facecolor=COLORS['joints'], 
                           edgecolor='white', linewidth=2, zorder=3))
    
    # Winkel-Labels SMART platzieren
    for a in cfg.get("angles", []):
        try:
            # Mittelpunkt des Winkels finden (pts[1] ist der Vertex)
            vertex_token = a["points"][1]
            
            # Gleiche Auflösungslogik wie bei Segmenten
            if vertex_token.startswith(("left_", "right_", "front_", "rear_")):
                vertex = _resolve_xy_any(pose, side, vertex_token)
            elif exercise_type == "unilateral" and vertex_token not in ["shoulder", "elbow", "wrist"]:
                try:
                    vertex = _resolve_xy_any(pose, side, f"front_{vertex_token}")
                except Exception:
                    vertex = _get_enhanced_xy(pose, side, vertex_token)
            else:
                vertex = _get_enhanced_xy(pose, side, vertex_token)
            
            v = tp(vertex)
            val = angles.get(f"{a['id']}_deg", 0.0)
            
            # Label-Position: Leicht offset vom Gelenk
            label_x = min(max(v[0] + 0.06, 0.05), 0.95)
            label_y = min(max(v[1] - 0.06, 0.05), 0.95)
            
            ax.text(label_x, label_y,
                   f"{a.get('label', a['id'])}: {val:.1f}°",
                   fontsize=10, color=COLORS['text'],
                   bbox=dict(boxstyle="round,pad=0.3", fc='white', 
                            ec=COLORS['accent'], alpha=0.9))
        except Exception as e:
            print(f"Winkel-Label {a.get('id')} nicht platzierbar: {e}")
            pass
    
    ax.axis('off')

def _draw_angles_panel(ax, angles: Dict[str, Any]):
    ax.set_facecolor(COLORS['panel_bg'])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(COLORS['border']); s.set_linewidth(1)
    data = [(k.replace('_deg','').replace('_',' ').title(), v)
            for k,v in angles.items() if k.endswith("_deg")]
    if not data:
        ax.text(0.5,0.5,"No angle data", ha='center', va='center', transform=ax.transAxes)
    else:
        y=0.85
        for name,val in data[:6]:
            ax.add_patch(patches.Rectangle((0.02,y-0.06),0.96,0.12,fc='white',ec=COLORS['border'],alpha=0.5,transform=ax.transAxes))
            ax.text(0.05,y,f"{name}:", transform=ax.transAxes, fontsize=10, color=COLORS['text'], fontweight='bold')
            ax.text(0.95,y,f"{val:.1f}", ha='right', transform=ax.transAxes, fontsize=10, color=COLORS['accent'])
            y-=0.16
    ax.set_title('ðŸ“ Joint Angles', fontsize=14, fontweight='bold', color=COLORS['text'], pad=15)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

def _draw_quick_panel(ax, feedback_text: str):
    ax.set_facecolor(COLORS['panel_bg'])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(COLORS['border']); s.set_linewidth(1)
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
                if in_cons and len(cons) < 2 and len(key) > 3:cons.append(f"✘ {key}")
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
    ax.set_title('ðŸŽ¯ Pose Quality', fontsize=14, fontweight='bold', color=COLORS['text'], pad=15)
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
    ax.set_title('ðŸ‘¥ Profile Info', fontsize=14, fontweight='bold', color=COLORS['text'], pad=15)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

def _place_panel_titles(fig, ax_left, ax_right, y=0.895):
    pL = ax_left.get_position(fig); pR = ax_right.get_position(fig)
    xL = (pL.x0+pL.x1)/2; xR = (pR.x0+pR.x1)/2
    kw = dict(ha='center', va='bottom', fontsize=14, fontweight='bold', color=COLORS['text'])
    fig.text(xL, y, 'Original Image with Pose Analysis', **kw)
    fig.text(xR, y, 'Vector Body Diagram', **kw)

def _draw_moments_panel(ax, feedback_text: str):
    """Zeigt Quick-Assessment aus deinem KI-Feedback (Pros/Cons kompakt)."""
    ax.set_facecolor(COLORS['panel_bg'])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['border'])
        spine.set_linewidth(1)

    pros, cons = [], []
    txt = (feedback_text or "").strip()
    if txt:
        lines = txt.splitlines()
        in_pros = in_cons = False
        for line in lines:
            s = line.strip()
            low = s.lower()

            # Abschnittserkennung
            if "was gut ist" in low or "was gut" in low:
                in_pros, in_cons = True, False
                continue
            if "was zu verbessern ist" in low or "zu verbessern" in low:
                in_pros, in_cons = False, True
                continue
            if low.startswith("#") or "konkrete cues" in low or "belastung" in low:
                in_pros = in_cons = False
                continue

            # Bulletpoints erkennen
            if s.startswith(("-", "*")):
                text = s[1:].strip().replace("**", "")
                # kurzer key-term vorn ziehen
                key = text.split(":", 1)[0].split("â€“", 1)[0].split("(", 1)[0].split(",", 1)[0].strip()
                if len(key) > 30: key = key[:27] + "..."
                if in_pros and len(pros) < 2 and len(key) > 2:
                    pros.append(f"âœ“ {key}")
                elif in_cons and len(cons) < 2 and len(key) > 2:
                    cons.append(f"âœ— {key}")

    if not pros and not cons:
        ax.text(0.5, 0.5, 'Generiere Analyse...', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color=COLORS['text'], style='italic')
    else:
        y = 0.80
        for p in pros[:2]:
            ax.text(0.05, y, (p[:45] + "..." if len(p) > 45 else p),
                    fontsize=9, color=COLORS['success'], fontweight='bold',
                    transform=ax.transAxes, wrap=True)
            y -= 0.20
        for c in cons[:2]:
            ax.text(0.05, y, (c[:45] + "..." if len(c) > 45 else c),
                    fontsize=9, color=COLORS['error'], fontweight='bold',
                    transform=ax.transAxes, wrap=True)
            y -= 0.20

    ax.set_title('ðŸ“‹ Quick Assessment', fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')