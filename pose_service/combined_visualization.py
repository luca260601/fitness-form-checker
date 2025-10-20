"""
Combined visualization module - creates a single comprehensive analysis image.
Combines pose overlay, vector body, angles, and moments in one professional layout.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import cv2
from typing import Dict, Tuple
from .engine import LMS, _get_enhanced_xy, _choose_optimal_side

# Professional color palette - Enhanced for better visual appeal
COLORS = {
    'skeleton': '#1E3A8A',        # Deep blue for skeleton
    'joints': '#DC2626',          # Red for joints
    'force_high': '#EF4444',      # Bright red for high force
    'force_medium': '#F59E0B',    # Amber for medium force
    'force_low': '#10B981',       # Emerald for low force
    'text': '#1F2937',            # Dark gray for text
    'background': '#FFFFFF',      # Pure white background
    'grid': '#F3F4F6',            # Light gray grid
    'panel_bg': '#F9FAFB',        # Very light gray panels
    'accent': '#3B82F6',          # Blue accent
    'success': '#059669',         # Success green
    'warning': '#D97706',         # Warning orange
    'error': '#DC2626',           # Error red
    'border': '#E5E7EB',          # Light border
    'shadow': '#00000020'         # Subtle shadow
}

def create_combined_analysis_image(image_path: str, pose: Dict, cfg: Dict,
                                   angles: Dict[str, float], moments: Dict[str, float],
                                   profile_data: Dict, out_dir: str, ai_feedback: str = "") -> str:
    """
    Creates a single comprehensive analysis image combining all visualizations.

    Layout:
    +------------------+------------------+
    |                  |                  |
    |  Original Image  |   Vector Body    |
    |   with Overlay   |   Diagram        |
    |                  |                  |
    +------------------+------------------+
    |           Analysis Panel            |
    |  Angles | Moments | Quality | Info  |
    +-------------------------------------+
    """

    # Load original image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Create figure with enhanced layout and styling
    fig = plt.figure(figsize=(20, 14), facecolor=COLORS['background'], dpi=300)

    # Add subtle border around entire figure
    fig.patch.set_edgecolor(COLORS['border'])
    fig.patch.set_linewidth(2)

    # Define enhanced grid layout with better proportions
    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[2.5, 2.5, 1.2],
        width_ratios=[1.2, 1.2, 1, 1],
        hspace=0.2, wspace=0.15,
        left=0.05, right=0.95, top=0.88, bottom=0.08
    )

    # 1. Original image with pose overlay (top-left, spans 2x2)
    ax_original = fig.add_subplot(gs[0:2, 0:2])
    _draw_original_with_overlay(ax_original, original_img, pose, cfg)

    # 2. Vector body diagram (top-right, spans 2x2)
    ax_vector = fig.add_subplot(gs[0:2, 2:4])
    _draw_vector_body_diagram(ax_vector, pose, cfg, angles)

    # 3. Analysis panels (bottom row, spans full width)
    ax_angles = fig.add_subplot(gs[2, 0])
    ax_moments = fig.add_subplot(gs[2, 1])
    ax_quality = fig.add_subplot(gs[2, 2])
    ax_info = fig.add_subplot(gs[2, 3])

    _draw_angles_panel(ax_angles, angles)
    _draw_moments_panel(ax_moments, ai_feedback)   # Quick Assessment mit KI-Feedback
    _draw_quality_panel(ax_quality, pose)
    _draw_info_panel(ax_info, profile_data)

    # Main title & Subtitle
    title_text = 'Fitness Form Analysis - Professional Report'
    fig.suptitle(title_text, fontsize=24, fontweight='bold',
                 color=COLORS['text'], y=0.97, ha='center')

    exercise_name = cfg.get('name', 'Unknown Exercise')
    fig.text(0.5, 0.94, f'Exercise: {exercise_name}',
             ha='center', va='top', fontsize=16, color=COLORS['accent'],
             style='italic', fontweight='medium')

    # Footer
    fig.suptitle('Fitness Form Analysis - Professional Report', fontsize=24, fontweight='bold',
             color=COLORS['text'], y=0.97, ha='center')
    exercise_name = cfg.get('name', 'Unknown Exercise')
    fig.text(0.5, 0.94, f'Exercise: {exercise_name}', ha='center', va='top',
         fontsize=16, color=COLORS['accent'], style='italic', fontweight='medium')

    _place_panel_titles(fig, ax_original, ax_vector)
    
    # Save combined image
    os.makedirs(out_dir, exist_ok=True)
    timestamp_file = time.strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(out_dir, f'combined_analysis_{timestamp_file}.png')

    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=COLORS['background'], edgecolor='none')
    plt.close(fig)

    return output_path


def _draw_original_with_overlay(ax, img: np.ndarray, pose: Dict, cfg: Dict):
    """Draw original image with pose overlay and force arrows."""
    # Convert BGR to RGB for matplotlib
    img_with_overlay = _add_pose_overlay_to_image(img.copy(), pose, cfg)
    img_with_overlay_rgb = cv2.cvtColor(img_with_overlay, cv2.COLOR_BGR2RGB)

    ax.imshow(img_with_overlay_rgb)
    ax.axis('off')  # Titel wird außerhalb dieser Funktion gesetzt


def _add_pose_overlay_to_image(img: np.ndarray, pose: Dict, cfg: Dict) -> np.ndarray:
    """Add pose skeleton to the original image."""
    h, w = pose["image_size"]["height"], pose["image_size"]["width"]
    lms = pose["landmarks"]

    # Choose best side
    side = _choose_optimal_side(lms, ["shoulder", "hip", "knee", "ankle"])

    def get_point(joint_name: str) -> Tuple[int, int]:
        try:
            idx = LMS[f"{side}_{joint_name.upper()}"]
            lm = lms[idx]
            return (int(lm["x"] * w), int(lm["y"] * h))
        except Exception:
            return None

    # Get joint positions
    joints = {}
    for joint in ["shoulder", "hip", "knee", "ankle", "elbow", "wrist"]:
        pos = get_point(joint)
        if pos:
            joints[joint] = pos

    # Draw enhanced skeleton with professional styling
    connections = [
        ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"),
        ("shoulder", "elbow"), ("elbow", "wrist")
    ]

    for start_joint, end_joint in connections:
        if start_joint in joints and end_joint in joints:
            start_pos = joints[start_joint]
            end_pos = joints[end_joint]
            # Shadow/Outline
            cv2.line(img, start_pos, end_pos, (0, 0, 0), 12, cv2.LINE_AA)
            cv2.line(img, start_pos, end_pos, (255, 255, 255), 10, cv2.LINE_AA)
            # Main colored line (Deep blue in BGR)
            cv2.line(img, start_pos, end_pos, (138, 58, 30), 6, cv2.LINE_AA)

    # Draw joints
    for _, position in joints.items():
        cv2.circle(img, (position[0] + 2, position[1] + 2), 15, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, position, 15, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, position, 11, (38, 38, 220), -1, cv2.LINE_AA)  # Rot in BGR
        cv2.circle(img, (position[0] - 3, position[1] - 3), 4, (255, 255, 255), -1, cv2.LINE_AA)

    return img


def _draw_vector_body_diagram(ax, pose: Dict, cfg: Dict, angles: Dict):
    """Draw the vector body diagram, perfectly centered and scaled."""
    side = angles.get("__side__", "LEFT")

    segments = cfg.get("segments", [])
    pts = []
    for a, b in segments:
        try:
            pts.append(_get_enhanced_xy(pose, side, a))
            pts.append(_get_enhanced_xy(pose, side, b))
        except Exception:
            continue

    if not pts:
        ax.text(0.5, 0.5, 'No pose data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color=COLORS['text'])
        ax.axis('off')
        return

    P = np.vstack(pts).astype(float)

    # --- Bounding Box & zentrierte, isotrope Skalierung ---
    p_min = P.min(axis=0)
    p_max = P.max(axis=0)
    center = (p_min + p_max) / 2.0
    half_size = (p_max - p_min) / 2.0
    max_half = float(np.max(half_size))
    if max_half == 0:
        max_half = 1.0

    # 80% der Panelfläche nutzen (Rand)
    target_extent = 0.80
    scale = target_extent / (2.0 * max_half)

    def to_panel(p):
        q = (p - center) * scale + np.array([0.5, 0.5])  # Mittelpunkt -> (0.5, 0.5)
        return q

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)  # Inverted Y
    ax.set_aspect('equal')

    # --- Skeleton zeichnen ---
    joint_positions = {}
    for a, b in segments:
        try:
            A = to_panel(_get_enhanced_xy(pose, side, a))
            B = to_panel(_get_enhanced_xy(pose, side, b))
        except Exception:
            continue

        joint_positions[a] = A
        joint_positions[b] = B

        ax.plot([A[0], B[0]], [A[1], B[1]],
                color=COLORS['skeleton'], linewidth=6, alpha=0.8,
                solid_capstyle='round')

    # --- Gelenke ---
    for _, pos in joint_positions.items():
        ax.add_patch(Circle(pos, 0.025, facecolor='white',
                            edgecolor=COLORS['joints'], linewidth=2, zorder=10))
        ax.add_patch(Circle(pos, 0.015, facecolor=COLORS['joints'], zorder=11))

    # --- Winkel-Labels mit Clamping ins Panel ---
    def clamp(v, lo=0.05, hi=0.95):
        return np.clip(v, lo, hi)

    angle_configs = cfg.get("angles", [])
    for angle_config in angle_configs:
        angle_id = angle_config["id"]
        label = angle_config.get("label", angle_id)
        points_cfg = angle_config["points"]
        if len(points_cfg) >= 2:
            try:
                vertex = to_panel(_get_enhanced_xy(pose, side, points_cfg[1]))
                val = angles.get(f"{angle_id}_deg", 0.0)
                vx = clamp(vertex[0] + 0.08)
                vy = clamp(vertex[1] - 0.08)

                bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white',
                                  edgecolor=COLORS['accent'], alpha=0.95, linewidth=2)

                ax.text(vx, vy, f"{label}: {val:.1f}°",
                        fontsize=11, fontweight='bold', color=COLORS['text'],
                        bbox=bbox_props, ha='left', va='top',
                        transform=ax.transAxes)
            except Exception:
                continue

    ax.axis('off')


def _draw_angles_panel(ax, angles: Dict):
    """Draw enhanced angles information panel."""
    ax.set_facecolor(COLORS['panel_bg'])

    # Subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['border'])
        spine.set_linewidth(1)

    # Filter angle data
    angle_data = [(k.replace('_deg', '').replace('_', ' ').title(), v)
                  for k, v in angles.items()
                  if k.endswith('_deg') and k != '__side___deg']

    if not angle_data:
        ax.text(0.5, 0.5, 'No angle data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color=COLORS['text'], style='italic')
    else:
        y_pos = 0.85
        for name, value in angle_data[:5]:  # Max 5 angles
            # Klare Farbkodierung: Grün=gut, Gelb=mittel, Rot=schlecht
            if 'knee' in name.lower():
                # Kniewinkel: 80-110° optimal, 60-140° akzeptabel
                if 80 <= value <= 110:
                    color = COLORS['success']  # Grün
                elif 60 <= value <= 140:
                    color = COLORS['warning']  # Gelb
                else:
                    color = COLORS['error']    # Rot
            elif 'hip' in name.lower():
                # Hüftwinkel: 70-100° optimal, 50-120° akzeptabel
                if 70 <= value <= 100:
                    color = COLORS['success']
                elif 50 <= value <= 120:
                    color = COLORS['warning']
                else:
                    color = COLORS['error']
            elif 'ankle' in name.lower():
                # Sprunggelenk: 130-170° optimal, 110-180° akzeptabel
                if 130 <= value <= 170:
                    color = COLORS['success']
                elif 110 <= value <= 180:
                    color = COLORS['warning']
                else:
                    color = COLORS['error']
            elif 'trunk' in name.lower():
                # Rumpfneigung: 120-160° optimal, 100-170° akzeptabel
                if 120 <= value <= 160:
                    color = COLORS['success']
                elif 100 <= value <= 170:
                    color = COLORS['warning']
                else:
                    color = COLORS['error']
            else:
                # Standard: Mittelbereich grün, Extremwerte rot
                if 60 <= value <= 120:
                    color = COLORS['success']
                elif 40 <= value <= 150:
                    color = COLORS['warning']
                else:
                    color = COLORS['error']

            rect = patches.Rectangle((0.02, y_pos - 0.06), 0.96, 0.12,
                                     facecolor=COLORS['background'],
                                     edgecolor=COLORS['border'],
                                     alpha=0.3, transform=ax.transAxes)
            ax.add_patch(rect)

            ax.text(0.05, y_pos, f"{name}:", fontweight='bold',
                    transform=ax.transAxes, fontsize=11, color=COLORS['text'])

            ax.text(0.95, y_pos, f"{value:.1f}°", ha='right',
                    transform=ax.transAxes, fontsize=11, color=color,
                    fontweight='bold')

            y_pos -= 0.16

    ax.set_title('📐 Joint Angles', fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def _draw_moments_panel(ax, ai_feedback: str):
    """Draw pros/cons analysis panel based on AI feedback."""
    ax.set_facecolor(COLORS['panel_bg'])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['border'])
        spine.set_linewidth(1)

    # Extract pros and cons from structured AI feedback sections
    pros = []
    cons = []
    
    if ai_feedback:
        lines = ai_feedback.split('\n')
        
        # State machine: parse "Was gut ist" and "Was zu verbessern ist" sections
        in_pros_section = False
        in_cons_section = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detect section headers
            if 'was gut ist' in line_stripped.lower():
                in_pros_section = True
                in_cons_section = False
                continue
            elif 'was zu verbessern ist' in line_stripped.lower():
                in_pros_section = False
                in_cons_section = True
                continue
            elif line_stripped.startswith('#') or 'konkrete cues' in line_stripped.lower() or 'belastung' in line_stripped.lower():
                # New section starts, exit current sections
                in_pros_section = False
                in_cons_section = False
                continue
            
            # Extract bullet points from current section
            if line_stripped.startswith('-') or line_stripped.startswith('*'):
                # Remove bullet and markdown formatting
                text = line_stripped[1:].strip().replace('**', '')
                
                # Extract key term by splitting at delimiters
                key_term = text
                
                # Try to split by delimiters in priority order
                if ':' in text:
                    key_term = text.split(':', 1)[0].strip()
                elif '–' in text:  # Em-dash
                    key_term = text.split('–', 1)[0].strip()
                elif '(' in text:
                    key_term = text.split('(', 1)[0].strip()
                elif ',' in text:  # Comma as fallback
                    key_term = text.split(',', 1)[0].strip()
                
                # Limit to max 30 characters for clean display
                if len(key_term) > 30:
                    key_term = key_term[:27].strip() + "..."
                
                # Add to appropriate list
                if in_pros_section and len(pros) < 2 and len(key_term) > 3:
                    pros.append(f"✓ {key_term}")
                elif in_cons_section and len(cons) < 2 and len(key_term) > 3:
                    cons.append(f"✗ {key_term}")
    
    # Fallback if no feedback
    if not pros and not cons:
        ax.text(0.5, 0.5, 'Generiere Analyse...', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color=COLORS['text'], style='italic')
    else:
        y_pos = 0.80
        
        # Draw pros
        for pro in pros[:2]:
            # Truncate if too long
            if len(pro) > 45:
                pro = pro[:42] + "..."
            ax.text(0.05, y_pos, pro, fontsize=9, color=COLORS['success'],
                    fontweight='bold', transform=ax.transAxes, wrap=True)
            y_pos -= 0.20
        
        # Draw cons
        for con in cons[:2]:
            # Truncate if too long
            if len(con) > 45:
                con = con[:42] + "..."
            ax.text(0.05, y_pos, con, fontsize=9, color=COLORS['error'],
                    fontweight='bold', transform=ax.transAxes, wrap=True)
            y_pos -= 0.20

    ax.set_title('📋 Quick Assessment', fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def _draw_quality_panel(ax, pose: Dict):
    """Draw enhanced pose quality panel."""
    ax.set_facecolor(COLORS['panel_bg'])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['border'])
        spine.set_linewidth(1)

    quality = pose.get("quality_metrics", {})
    score = quality.get('pose_quality_score', 0)
    is_high_quality = quality.get('is_high_quality', False)
    avg_vis = quality.get('average_visibility', 0)

    if is_high_quality and score > 0.8:
        status_color = COLORS['success']; status_text = "EXCELLENT"; status_icon = "🟢"
    elif score > 0.6:
        status_color = COLORS['warning']; status_text = "GOOD"; status_icon = "🟡"
    elif score > 0.3:
        status_color = COLORS['error']; status_text = "FAIR"; status_icon = "🟠"
    else:
        status_color = COLORS['error']; status_text = "POOR"; status_icon = "🔴"

    circle = Circle((0.5, 0.65), 0.25, facecolor=status_color, alpha=0.2,
                    edgecolor=status_color, linewidth=3, transform=ax.transAxes)
    ax.add_patch(circle)

    ax.text(0.5, 0.65, f"{status_icon}\n{status_text}", ha='center', va='center',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            color=status_color, linespacing=1.5)

    ax.text(0.5, 0.35, f"Quality Score: {score:.3f}", ha='center', va='center',
            transform=ax.transAxes, fontsize=10, color=COLORS['text'],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['background'],
                      edgecolor=COLORS['border']))

    ax.text(0.5, 0.2, f"Visibility: {avg_vis:.3f}", ha='center', va='center',
            transform=ax.transAxes, fontsize=10, color=COLORS['text'],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['background'],
                      edgecolor=COLORS['border']))

    ax.set_title('🎯 Pose Quality', fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def _draw_info_panel(ax, profile_data: Dict):
    """Draw enhanced profile information panel."""
    ax.set_facecolor(COLORS['panel_bg'])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['border'])
        spine.set_linewidth(1)

    y_pos = 0.85
    info_items = [
        ("👤 Name", profile_data.get('name', 'Unknown')),
        ("⚖️ Weight", f"{profile_data.get('body_mass_kg', 0):.1f} kg"),
        ("📏 Height", f"{profile_data.get('height_cm', 0):.0f} cm")
    ]

    for label, value in info_items:
        rect = patches.Rectangle((0.02, y_pos - 0.06), 0.96, 0.12,
                                 facecolor=COLORS['background'],
                                 edgecolor=COLORS['border'],
                                 alpha=0.3, transform=ax.transAxes)
        ax.add_patch(rect)

        ax.text(0.05, y_pos, label, fontweight='bold',
                transform=ax.transAxes, fontsize=10, color=COLORS['text'])

        ax.text(0.95, y_pos, str(value), ha='right',
                transform=ax.transAxes, fontsize=10, color=COLORS['accent'],
                fontweight='medium', bbox=dict(boxstyle="round,pad=0.2",
                facecolor=COLORS['accent'], alpha=0.1))

        y_pos -= 0.18

    ax.set_title('👥 Profile Info', fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def _place_panel_titles(fig, ax_left, ax_right, y=0.895):
    """Platziert die beiden oberen Panel-Titel exakt auf einer Linie (Figure-Koordinaten)."""
    # Mittelpunkte der Achsen in Figure-Koordinaten
    posL = ax_left.get_position(fig)
    posR = ax_right.get_position(fig)
    xL = (posL.x0 + posL.x1) / 2.0
    xR = (posR.x0 + posR.x1) / 2.0

    # Einheitliches Styling
    kw = dict(ha='center', va='bottom', fontsize=14, fontweight='bold', color=COLORS['text'])

    # Texte an gleicher y-Position setzen
    fig.text(xL, y, 'Original Image with Pose Analysis', **kw)
    fig.text(xR, y, 'Vector Body Diagram', **kw)

