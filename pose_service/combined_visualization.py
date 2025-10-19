"""
Combined visualization module - creates a single comprehensive analysis image.
Combines pose overlay, vector body, angles, and moments in one professional layout.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import cv2
from typing import Dict, List, Tuple, Optional
from .engine import LMS, _get_xy
try:
    from .enhanced_engine import _get_enhanced_xy, _choose_optimal_side
except ImportError:
    # Fallback if enhanced_engine is not available
    from .engine import _get_xy as _get_enhanced_xy
    def _choose_optimal_side(lms, points):
        return "LEFT"  # Simple fallback

# Professional color palette
COLORS = {
    'skeleton': '#2E86AB',
    'joints': '#A23B72', 
    'force_high': '#F18F01',
    'force_medium': '#C73E1D',
    'force_low': '#4CAF50',
    'text': '#2C3E50',
    'background': '#FFFFFF',
    'grid': '#ECF0F1',
    'panel_bg': '#F8F9FA',
    'accent': '#3498DB'
}

def create_combined_analysis_image(image_path: str, pose: Dict, cfg: Dict, 
                                 angles: Dict[str, float], moments: Dict[str, float],
                                 profile_data: Dict, out_dir: str) -> str:
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
    
    orig_h, orig_w = original_img.shape[:2]
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 12), facecolor=COLORS['background'])
    
    # Define grid layout
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1, 1],
                         hspace=0.15, wspace=0.1)
    
    # 1. Original image with pose overlay (top-left, spans 2x2)
    ax_original = fig.add_subplot(gs[0:2, 0:2])
    _draw_original_with_overlay(ax_original, original_img, pose, moments, cfg)
    
    # 2. Vector body diagram (top-right, spans 2x2)  
    ax_vector = fig.add_subplot(gs[0:2, 2:4])
    _draw_vector_body_diagram(ax_vector, pose, cfg, angles, moments)
    
    # 3. Analysis panels (bottom row, spans full width)
    ax_angles = fig.add_subplot(gs[2, 0])
    ax_moments = fig.add_subplot(gs[2, 1]) 
    ax_quality = fig.add_subplot(gs[2, 2])
    ax_info = fig.add_subplot(gs[2, 3])
    
    _draw_angles_panel(ax_angles, angles)
    _draw_moments_panel(ax_moments, moments)
    _draw_quality_panel(ax_quality, pose)
    _draw_info_panel(ax_info, profile_data)
    
    # Add main title
    fig.suptitle('Fitness Form Analysis - Comprehensive Report', 
                fontsize=20, fontweight='bold', color=COLORS['text'], y=0.95)
    
    # Add timestamp
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.99, 0.01, f'Generated: {timestamp}', 
            ha='right', va='bottom', fontsize=10, color=COLORS['text'], alpha=0.7)
    
    # Save combined image
    os.makedirs(out_dir, exist_ok=True)
    timestamp_file = time.strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(out_dir, f'combined_analysis_{timestamp_file}.png')
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor=COLORS['background'], edgecolor='none')
    plt.close(fig)
    
    return output_path

def _draw_original_with_overlay(ax, img: np.ndarray, pose: Dict, moments: Dict, cfg: Dict):
    """Draw original image with pose overlay and force arrows."""
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw pose overlay on image
    img_with_overlay = _add_pose_overlay_to_image(img.copy(), pose, moments, cfg)
    img_with_overlay_rgb = cv2.cvtColor(img_with_overlay, cv2.COLOR_BGR2RGB)
    
    ax.imshow(img_with_overlay_rgb)
    ax.set_title('Original Image with Pose Analysis', fontsize=14, fontweight='bold', 
                color=COLORS['text'], pad=20)
    ax.axis('off')

def _add_pose_overlay_to_image(img: np.ndarray, pose: Dict, moments: Dict, cfg: Dict) -> np.ndarray:
    """Add pose skeleton and force arrows to the original image."""
    
    h, w = pose["image_size"]["height"], pose["image_size"]["width"]
    lms = pose["landmarks"]
    
    # Choose best side
    side = _choose_optimal_side(lms, ["shoulder", "hip", "knee", "ankle"])
    
    def get_point(joint_name: str) -> Tuple[int, int]:
        try:
            idx = LMS[f"{side}_{joint_name.upper()}"]
            lm = lms[idx]
            return (int(lm["x"] * w), int(lm["y"] * h))
        except:
            return None
    
    # Get joint positions
    joints = {}
    for joint in ["shoulder", "hip", "knee", "ankle", "elbow", "wrist"]:
        pos = get_point(joint)
        if pos:
            joints[joint] = pos
    
    # Draw skeleton
    connections = [
        ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"),
        ("shoulder", "elbow"), ("elbow", "wrist")
    ]
    
    for start_joint, end_joint in connections:
        if start_joint in joints and end_joint in joints:
            start_pos = joints[start_joint]
            end_pos = joints[end_joint]
            
            # Draw thick white outline
            cv2.line(img, start_pos, end_pos, (255, 255, 255), 8, cv2.LINE_AA)
            # Draw colored line
            cv2.line(img, start_pos, end_pos, (46, 134, 171), 4, cv2.LINE_AA)
    
    # Draw joints
    for joint_name, position in joints.items():
        cv2.circle(img, position, 12, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, position, 8, (114, 59, 162), -1, cv2.LINE_AA)
    
    # Draw force arrows
    overlay_joints = cfg.get("overlays", {}).get("arrows_at", [])
    for joint in overlay_joints:
        if joint in joints:
            moment_key = f"{joint}_moment_Nm"
            moment_value = float(moments.get(moment_key, 0.0))
            
            if moment_value > 0:
                _draw_force_arrows_on_image(img, joints[joint], moment_value, joint)
    
    return img

def _draw_force_arrows_on_image(img: np.ndarray, center: Tuple[int, int], 
                               moment: float, joint_name: str):
    """Draw force arrows on the image."""
    
    # Color based on force magnitude
    if moment > 200:
        color = (1, 143, 241)  # Orange (BGR)
        scale_factor = 1.0
    elif moment > 100:
        color = (29, 62, 199)  # Red (BGR)
        scale_factor = 0.8
    else:
        color = (76, 175, 80)  # Green (BGR)
        scale_factor = 0.6
    
    # Arrow properties
    base_length = 60
    arrow_length = int(base_length * scale_factor * min(1.0, moment / 300.0))
    thickness = max(2, int(3 * scale_factor))
    
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
        cv2.arrowedLine(img, end_point, center, color, thickness, 
                       cv2.LINE_AA, tipLength=0.3)
    
    # Add label
    label_pos = (center[0] + 15, center[1] - 15)
    cv2.putText(img, f"{moment:.0f} N·m", label_pos,
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{moment:.0f} N·m", label_pos,
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

def _draw_vector_body_diagram(ax, pose: Dict, cfg: Dict, angles: Dict, moments: Dict):
    """Draw the vector body diagram."""
    
    side = angles.get("__side__", "LEFT")
    
    # Collect points for normalization
    points = []
    segments = cfg.get("segments", [])
    
    for seg_start, seg_end in segments:
        try:
            points.extend([_get_enhanced_xy(pose, side, seg_start), 
                          _get_enhanced_xy(pose, side, seg_end)])
        except:
            continue
    
    if not points:
        ax.text(0.5, 0.5, 'No pose data available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12, color=COLORS['text'])
        ax.set_title('Vector Body Diagram', fontsize=14, fontweight='bold')
        ax.axis('off')
        return
    
    points_array = np.vstack(points)
    min_coords = points_array.min(axis=0)
    max_coords = points_array.max(axis=0)
    span = np.maximum(max_coords - min_coords, 1)
    
    def normalize_point(point: np.ndarray) -> np.ndarray:
        return (point - min_coords) / span
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1.1, -0.1)  # Inverted Y
    ax.set_aspect('equal')
    
    # Draw skeleton
    joint_positions = {}
    for seg_start, seg_end in segments:
        try:
            start_pos = normalize_point(_get_enhanced_xy(pose, side, seg_start))
            end_pos = normalize_point(_get_enhanced_xy(pose, side, seg_end))
            
            joint_positions[seg_start] = start_pos
            joint_positions[seg_end] = end_pos
            
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                   color=COLORS['skeleton'], linewidth=6, alpha=0.8, 
                   solid_capstyle='round')
        except:
            continue
    
    # Draw joints
    for joint_name, pos in joint_positions.items():
        circle = Circle(pos, 0.025, facecolor='white', edgecolor=COLORS['joints'], 
                       linewidth=2, zorder=10)
        ax.add_patch(circle)
        
        circle_inner = Circle(pos, 0.015, facecolor=COLORS['joints'], zorder=11)
        ax.add_patch(circle_inner)
    
    # Draw angle labels
    angle_configs = cfg.get("angles", [])
    for angle_config in angle_configs:
        angle_id = angle_config["id"]
        label = angle_config.get("label", angle_id)
        points_cfg = angle_config["points"]
        
        if len(points_cfg) >= 2:
            try:
                vertex_pos = normalize_point(_get_enhanced_xy(pose, side, points_cfg[1]))
                angle_value = angles.get(f"{angle_id}_deg", 0.0)
                
                bbox_props = dict(boxstyle="round,pad=0.2", facecolor='white', 
                                edgecolor=COLORS['text'], alpha=0.9, linewidth=1)
                
                ax.text(vertex_pos[0] + 0.05, vertex_pos[1] - 0.05, 
                       f"{label}: {angle_value:.1f}°", 
                       fontsize=10, fontweight='bold', color=COLORS['text'],
                       bbox=bbox_props, ha='left', va='top')
            except:
                continue
    
    # Draw force vectors
    overlay_joints = cfg.get("overlays", {}).get("arrows_at", [])
    for joint in overlay_joints:
        if joint in joint_positions:
            moment_key = f"{joint}_moment_Nm"
            moment_value = float(moments.get(moment_key, 0.0))
            
            if moment_value > 0:
                center = joint_positions[joint]
                
                # Color based on magnitude
                if moment_value > 200:
                    color = COLORS['force_high']
                elif moment_value > 100:
                    color = COLORS['force_medium']
                else:
                    color = COLORS['force_low']
                
                # Draw arrows
                scale = min(1.0, moment_value / 300.0)
                arrow_length = 0.08 + 0.12 * scale
                
                directions = np.array([
                    [1, 0], [-1, 0], [0, 1], [0, -1],
                    [0.707, 0.707], [-0.707, 0.707], [0.707, -0.707], [-0.707, -0.707]
                ])
                
                for direction in directions:
                    tail_pos = center + direction * arrow_length
                    ax.annotate("", xy=center, xytext=tail_pos,
                               arrowprops=dict(arrowstyle="->", color=color, 
                                             lw=2, alpha=0.8))
    
    ax.set_title('Vector Body Diagram', fontsize=14, fontweight='bold', 
                color=COLORS['text'], pad=20)
    ax.axis('off')

def _draw_angles_panel(ax, angles: Dict):
    """Draw angles information panel."""
    ax.set_facecolor(COLORS['panel_bg'])
    
    # Filter angle data
    angle_data = [(k.replace('_deg', '').title(), v) 
                  for k, v in angles.items() 
                  if k.endswith('_deg') and k != '__side___deg']
    
    if not angle_data:
        ax.text(0.5, 0.5, 'No angle data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12, color=COLORS['text'])
    else:
        y_pos = 0.9
        for name, value in angle_data[:6]:  # Show max 6 angles
            color = COLORS['force_low'] if 70 <= value <= 120 else COLORS['force_medium']
            ax.text(0.05, y_pos, f"{name}:", fontweight='bold', 
                   transform=ax.transAxes, fontsize=10, color=COLORS['text'])
            ax.text(0.95, y_pos, f"{value:.1f}°", ha='right',
                   transform=ax.transAxes, fontsize=10, color=color, fontweight='bold')
            y_pos -= 0.15
    
    ax.set_title('Joint Angles', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def _draw_moments_panel(ax, moments: Dict):
    """Draw moments information panel."""
    ax.set_facecolor(COLORS['panel_bg'])
    
    # Filter moment data
    moment_data = [(k.replace('_moment_Nm', '').title(), v) 
                   for k, v in moments.items() 
                   if k.endswith('_moment_Nm') and v > 0]
    
    if not moment_data:
        ax.text(0.5, 0.5, 'No moment data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12, color=COLORS['text'])
    else:
        y_pos = 0.9
        for name, value in moment_data:
            if value > 200:
                color = COLORS['force_high']
            elif value > 100:
                color = COLORS['force_medium']
            else:
                color = COLORS['force_low']
                
            ax.text(0.05, y_pos, f"{name}:", fontweight='bold',
                   transform=ax.transAxes, fontsize=10, color=COLORS['text'])
            ax.text(0.95, y_pos, f"{value:.1f} N·m", ha='right',
                   transform=ax.transAxes, fontsize=10, color=color, fontweight='bold')
            y_pos -= 0.2
    
    ax.set_title('Joint Moments', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def _draw_quality_panel(ax, pose: Dict):
    """Draw pose quality panel."""
    ax.set_facecolor(COLORS['panel_bg'])
    
    quality = pose.get("quality_metrics", {})
    
    # Quality score
    score = quality.get('pose_quality_score', 0)
    is_high_quality = quality.get('is_high_quality', False)
    
    if is_high_quality:
        status_color = COLORS['force_low']
        status_text = "EXCELLENT"
    elif score > 0.5:
        status_color = COLORS['force_medium'] 
        status_text = "GOOD"
    else:
        status_color = COLORS['force_high']
        status_text = "NEEDS WORK"
    
    ax.text(0.5, 0.8, status_text, ha='center', va='center',
           transform=ax.transAxes, fontsize=14, fontweight='bold', color=status_color)
    
    ax.text(0.5, 0.6, f"Score: {score:.3f}", ha='center', va='center',
           transform=ax.transAxes, fontsize=12, color=COLORS['text'])
    
    avg_vis = quality.get('average_visibility', 0)
    ax.text(0.5, 0.4, f"Visibility: {avg_vis:.3f}", ha='center', va='center',
           transform=ax.transAxes, fontsize=10, color=COLORS['text'])
    
    ax.set_title('Pose Quality', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def _draw_info_panel(ax, profile_data: Dict):
    """Draw profile information panel."""
    ax.set_facecolor(COLORS['panel_bg'])
    
    y_pos = 0.9
    info_items = [
        ("Name", profile_data.get('name', 'Unknown')),
        ("Weight", f"{profile_data.get('body_mass_kg', 0):.1f} kg"),
        ("Height", f"{profile_data.get('height_cm', 0):.0f} cm"),
        ("Level", profile_data.get('experience_level', 'Unknown'))
    ]
    
    for label, value in info_items:
        ax.text(0.05, y_pos, f"{label}:", fontweight='bold',
               transform=ax.transAxes, fontsize=10, color=COLORS['text'])
        ax.text(0.95, y_pos, str(value), ha='right',
               transform=ax.transAxes, fontsize=10, color=COLORS['accent'])
        y_pos -= 0.2
    
    ax.set_title('Profile Info', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
