"""
Enhanced visualization module for fitness form analysis.
Provides professional-grade visualizations with improved aesthetics and accuracy.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import cv2
from typing import Dict, List, Tuple, Optional
from .engine import LMS, _get_xy
import seaborn as sns

# Set up professional color palette
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    # Fallback for older seaborn versions
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
COLORS = {
    'skeleton': '#2E86AB',
    'joints': '#A23B72',
    'force_high': '#F18F01',
    'force_medium': '#C73E1D',
    'force_low': '#4CAF50',
    'text': '#2C3E50',
    'background': '#FFFFFF',
    'grid': '#ECF0F1'
}

def draw_enhanced_force_overlay(image_path: str, pose: Dict, moments: Dict[str, float], 
                               target_joint: str, out_dir: str) -> str:
    """
    Enhanced force overlay with professional styling and better visual clarity.
    """
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    h, w = pose["image_size"]["height"], pose["image_size"]["width"]
    lms = pose["landmarks"]
    
    # Smart side selection based on visibility
    side = _choose_best_side(lms, target_joint)
    
    def get_point(joint_name: str) -> Tuple[int, int]:
        idx = LMS[f"{side}_{joint_name.upper()}"]
        lm = lms[idx]
        return (int(lm["x"] * w), int(lm["y"] * h))
    
    # Get all joint positions
    joints = {}
    for joint in ["shoulder", "hip", "knee", "ankle", "elbow", "wrist"]:
        try:
            joints[joint] = get_point(joint)
        except KeyError:
            continue
    
    # Draw enhanced skeleton with gradient lines
    _draw_enhanced_skeleton(img, joints)
    
    # Draw force visualization for target joint
    if target_joint in joints:
        moment_key = f"{target_joint}_moment_Nm"
        moment_value = float(moments.get(moment_key, 0.0))
        
        if moment_value > 0:
            _draw_enhanced_force_arrows(img, joints[target_joint], moment_value, target_joint)
            _draw_enhanced_info_panel(img, target_joint, moment_value, joints[target_joint])
    
    # Add professional watermark
    _add_watermark(img)
    
    # Save with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f"enhanced_overlay_{target_joint}_{timestamp}.png")
    cv2.imwrite(out_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    return out_path

def draw_enhanced_vector_body(cfg: Dict, pose: Dict, angles: Dict[str, float], 
                             moments: Dict[str, float], out_dir: str) -> Dict[str, str]:
    """
    Enhanced vector body visualization with professional styling and better layout.
    """
    os.makedirs(out_dir, exist_ok=True)
    side = angles.get("__side__", "LEFT")
    
    # Collect all points for normalization
    points = []
    segments = cfg.get("segments", [])
    
    for seg_start, seg_end in segments:
        points.extend([_get_xy(pose, side, seg_start), _get_xy(pose, side, seg_end)])
    
    if not points:
        raise ValueError("No segments found in configuration")
    
    points_array = np.vstack(points)
    min_coords = points_array.min(axis=0)
    max_coords = points_array.max(axis=0)
    span = np.maximum(max_coords - min_coords, 1)
    
    def normalize_point(point: np.ndarray) -> np.ndarray:
        return (point - min_coords) / span
    
    # Create professional figure
    fig, ax = plt.subplots(figsize=(8, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(1.2, -0.2)  # Inverted Y-axis
    ax.axis('off')
    
    # Add subtle grid
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linewidth=0.5)
    
    # Draw enhanced skeleton
    joint_positions = {}
    _draw_enhanced_skeleton_matplotlib(ax, cfg, pose, side, normalize_point, joint_positions)
    
    # Draw angle annotations with enhanced styling
    _draw_enhanced_angle_labels(ax, cfg, angles, pose, side, normalize_point)
    
    # Draw force vectors with professional styling
    _draw_enhanced_force_vectors(ax, cfg, moments, joint_positions)
    
    # Add title and metadata
    _add_enhanced_title_and_metadata(ax, angles, moments)
    
    # Save both formats with high quality
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    svg_path = os.path.join(out_dir, f"enhanced_vector_body_{timestamp}.svg")
    png_path = os.path.join(out_dir, f"enhanced_vector_body_{timestamp}.png")
    
    fig.savefig(svg_path, bbox_inches="tight", facecolor=COLORS['background'], 
                edgecolor='none', format='svg')
    fig.savefig(png_path, bbox_inches="tight", facecolor=COLORS['background'], 
                edgecolor='none', dpi=300, format='png')
    
    plt.close(fig)
    return {"svg": svg_path, "png": png_path}

def _choose_best_side(lms: List[Dict], target_joint: str) -> str:
    """Choose the side with better visibility for the target joint."""
    joint_groups = {
        'upper': ["shoulder", "elbow", "wrist"],
        'lower': ["hip", "knee", "ankle"]
    }
    
    group = 'upper' if target_joint in joint_groups['upper'] else 'lower'
    joints_to_check = joint_groups[group]
    
    left_visibility = sum(lms[LMS[f"LEFT_{j.upper()}"]]["visibility"] 
                         for j in joints_to_check if f"LEFT_{j.upper()}" in LMS)
    right_visibility = sum(lms[LMS[f"RIGHT_{j.upper()}"]]["visibility"] 
                          for j in joints_to_check if f"RIGHT_{j.upper()}" in LMS)
    
    return "LEFT" if left_visibility >= right_visibility else "RIGHT"

def _draw_enhanced_skeleton(img: np.ndarray, joints: Dict[str, Tuple[int, int]]):
    """Draw enhanced skeleton with gradient lines and professional styling."""
    # Define skeleton connections
    connections = [
        ("shoulder", "hip"),
        ("hip", "knee"),
        ("knee", "ankle"),
        ("shoulder", "elbow"),
        ("elbow", "wrist")
    ]
    
    for start_joint, end_joint in connections:
        if start_joint in joints and end_joint in joints:
            start_pos = joints[start_joint]
            end_pos = joints[end_joint]
            
            # Draw thick outline
            cv2.line(img, start_pos, end_pos, (255, 255, 255), 8, cv2.LINE_AA)
            # Draw colored line
            cv2.line(img, start_pos, end_pos, (171, 134, 46), 4, cv2.LINE_AA)
    
    # Draw enhanced joint markers
    for joint_name, position in joints.items():
        # Outer circle (white outline)
        cv2.circle(img, position, 12, (255, 255, 255), -1, cv2.LINE_AA)
        # Inner circle (colored)
        cv2.circle(img, position, 8, (162, 59, 114), -1, cv2.LINE_AA)

def _draw_enhanced_force_arrows(img: np.ndarray, center: Tuple[int, int], 
                               moment: float, joint_name: str):
    """Draw enhanced force arrows with dynamic scaling and professional styling."""
    # Determine force level and colors
    if moment > 200:
        color = (1, 143, 241)  # Orange for high force
        scale_factor = 1.0
    elif moment > 100:
        color = (29, 62, 199)  # Red for medium force
        scale_factor = 0.8
    else:
        color = (76, 175, 80)  # Green for low force
        scale_factor = 0.6
    
    # Calculate arrow properties
    base_length = 80
    arrow_length = int(base_length * scale_factor * min(1.0, moment / 300.0))
    thickness = max(2, int(4 * scale_factor))
    
    # Draw arrows in 8 directions
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707)
    ]
    
    for dx, dy in directions:
        end_point = (
            center[0] + int(dx * arrow_length),
            center[1] + int(dy * arrow_length)
        )
        
        # Draw arrow with enhanced styling
        cv2.arrowedLine(img, end_point, center, color, thickness, 
                       cv2.LINE_AA, tipLength=0.3)

def _draw_enhanced_info_panel(img: np.ndarray, joint_name: str, moment: float, 
                             position: Tuple[int, int]):
    """Draw enhanced information panel with professional styling."""
    # Panel dimensions and position
    panel_width, panel_height = 200, 60
    panel_x = max(10, min(position[0] + 20, img.shape[1] - panel_width - 10))
    panel_y = max(10, position[1] - 40)
    
    # Draw panel background with rounded corners
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (255, 255, 255), -1)
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (200, 200, 200), 2)
    
    # Blend with original image
    alpha = 0.9
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Add text with enhanced styling
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Joint name
    cv2.putText(img, f"{joint_name.capitalize()} Joint", 
               (panel_x + 10, panel_y + 20), font, 0.6, (44, 62, 80), 2, cv2.LINE_AA)
    
    # Moment value
    cv2.putText(img, f"Moment: {moment:.1f} N·m", 
               (panel_x + 10, panel_y + 45), font, 0.5, (52, 73, 94), 2, cv2.LINE_AA)

def _draw_enhanced_skeleton_matplotlib(ax, cfg: Dict, pose: Dict, side: str, 
                                     normalize_func, joint_positions: Dict):
    """Draw enhanced skeleton using matplotlib with professional styling."""
    segments = cfg.get("segments", [])
    
    # Draw segments with enhanced styling
    for seg_start, seg_end in segments:
        start_pos = normalize_func(_get_xy(pose, side, seg_start))
        end_pos = normalize_func(_get_xy(pose, side, seg_end))
        
        # Store joint positions
        joint_positions[seg_start] = start_pos
        joint_positions[seg_end] = end_pos
        
        # Draw segment with gradient effect
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
               color=COLORS['skeleton'], linewidth=6, alpha=0.8, 
               solid_capstyle='round')
    
    # Draw enhanced joint markers
    for joint_name, pos in joint_positions.items():
        # Outer circle
        circle_outer = Circle(pos, 0.03, facecolor='white', edgecolor=COLORS['joints'], 
                            linewidth=2, zorder=10)
        ax.add_patch(circle_outer)
        
        # Inner circle
        circle_inner = Circle(pos, 0.02, facecolor=COLORS['joints'], zorder=11)
        ax.add_patch(circle_inner)

def _draw_enhanced_angle_labels(ax, cfg: Dict, angles: Dict[str, float], 
                               pose: Dict, side: str, normalize_func):
    """Draw enhanced angle labels with professional styling."""
    angle_configs = cfg.get("angles", [])
    
    for angle_config in angle_configs:
        angle_id = angle_config["id"]
        label = angle_config.get("label", angle_id)
        points = angle_config["points"]
        
        if len(points) >= 2:
            # Position label at the vertex (middle point)
            vertex_pos = normalize_func(_get_xy(pose, side, points[1]))
            angle_value = angles.get(f"{angle_id}_deg", 0.0)
            
            # Create fancy text box
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white', 
                            edgecolor=COLORS['text'], alpha=0.9, linewidth=1)
            
            ax.text(vertex_pos[0] + 0.05, vertex_pos[1] - 0.05, 
                   f"{label}: {angle_value:.1f}°", 
                   fontsize=11, fontweight='bold', color=COLORS['text'],
                   bbox=bbox_props, ha='left', va='top')

def _draw_enhanced_force_vectors(ax, cfg: Dict, moments: Dict[str, float], 
                                joint_positions: Dict):
    """Draw enhanced force vectors with professional styling."""
    overlay_config = cfg.get("overlays", {})
    arrows_at = overlay_config.get("arrows_at", [])
    
    for joint in arrows_at:
        moment_key = f"{joint}_moment_Nm"
        moment_value = float(moments.get(moment_key, 0.0))
        
        if moment_value <= 0 or joint not in joint_positions:
            continue
        
        center = joint_positions[joint]
        
        # Determine color based on force magnitude
        if moment_value > 200:
            color = COLORS['force_high']
            alpha = 0.8
        elif moment_value > 100:
            color = COLORS['force_medium']
            alpha = 0.7
        else:
            color = COLORS['force_low']
            alpha = 0.6
        
        # Calculate arrow properties
        scale = min(1.0, moment_value / 300.0)
        arrow_length = 0.1 + 0.15 * scale
        
        # Draw arrows in multiple directions
        directions = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [0.707, 0.707], [-0.707, 0.707], [0.707, -0.707], [-0.707, -0.707]
        ])
        
        for direction in directions:
            tail_pos = center + direction * arrow_length
            ax.annotate("", xy=center, xytext=tail_pos,
                       arrowprops=dict(arrowstyle="->", color=color, 
                                     lw=3, alpha=alpha, shrinkA=5, shrinkB=5))
        
        # Add moment label
        label_pos = center + np.array([0.08, -0.08])
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, 
                         alpha=0.2, edgecolor=color, linewidth=1)
        
        ax.text(label_pos[0], label_pos[1], f"{joint.capitalize()}\n{moment_value:.1f} N·m",
               fontsize=10, fontweight='bold', color=color,
               bbox=bbox_props, ha='center', va='center')

def _add_enhanced_title_and_metadata(ax, angles: Dict[str, float], moments: Dict[str, float]):
    """Add enhanced title and metadata to the visualization."""
    # Main title
    ax.text(0.5, 1.1, "Biomechanical Analysis", 
           transform=ax.transAxes, fontsize=16, fontweight='bold',
           ha='center', va='bottom', color=COLORS['text'])
    
    # Subtitle with timestamp
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    ax.text(0.5, 1.05, f"Generated: {timestamp}", 
           transform=ax.transAxes, fontsize=10, 
           ha='center', va='bottom', color=COLORS['text'], alpha=0.7)
    
    # Summary statistics
    total_moment = sum(v for k, v in moments.items() if k.endswith('_moment_Nm'))
    side_used = angles.get('__side__', 'LEFT').lower()
    
    summary_text = f"Side: {side_used} | Total Moment: {total_moment:.1f} N·m"
    ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, 
           fontsize=9, ha='left', va='bottom', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

def _add_watermark(img: np.ndarray):
    """Add professional watermark to the image."""
    h, w = img.shape[:2]
    watermark_text = "Fitness Form Analyzer"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(watermark_text, font, font_scale, thickness)
    
    # Position at bottom right
    x = w - text_width - 10
    y = h - 10
    
    # Add semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), 
                 (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Add text
    cv2.putText(img, watermark_text, (x, y), font, font_scale, (100, 100, 100), thickness, cv2.LINE_AA)
