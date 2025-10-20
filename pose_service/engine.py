"""
Unified pose analysis engine - combines basic and enhanced functionality.
"""

import math
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional

# Try to import scipy for enhanced features, fallback if not available
try:
    from scipy.spatial.distance import euclidean
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# --- MediaPipe Setup ---
def _load_mediapipe():
    """Load MediaPipe with proper error handling."""
    try:
        import mediapipe as mp
        return mp
    except Exception as e:
        raise RuntimeError("MediaPipe ist nicht installiert/kompatibel (nutze Python 3.11).") from e

# --- Landmarks Indices (MediaPipe) ---
LMS = {
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "NOSE": 0, "LEFT_EYE": 1, "RIGHT_EYE": 2,
    "LEFT_EAR": 7, "RIGHT_EAR": 8,
    "LEFT_HEEL": 29, "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32
}

# --- Enhanced Pose Detection ---
def get_pose_vector(image_path: str, confidence_threshold: float = 0.5, enhanced: bool = True) -> Dict[str, Any]:
    """
    Unified pose detection function with optional enhanced features.
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence for landmarks
        enhanced: Whether to use enhanced features (quality metrics, etc.)
    
    Returns:
        Dictionary containing pose data and optional quality metrics
    """
    mp = _load_mediapipe()
    mp_pose = mp.solutions.pose
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use higher model complexity for enhanced mode
    model_complexity = 2 if enhanced else 1
    min_detection_confidence = 0.7 if enhanced else 0.5
    
    with mp_pose.Pose(
        static_image_mode=True, 
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence
    ) as pose:
        results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        raise RuntimeError("Keine Pose erkannt.")
    
    # Extract landmarks
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.append({
            "x": lm.x, 
            "y": lm.y, 
            "z": lm.z, 
            "visibility": lm.visibility
        })
    
    pose_data = {
        "image_size": {"width": w, "height": h}, 
        "landmarks": landmarks
    }
    
    # Add enhanced quality metrics if requested
    if enhanced and SCIPY_AVAILABLE:
        quality_metrics = _calculate_pose_quality(landmarks, confidence_threshold)
        pose_data["quality_metrics"] = quality_metrics
    
    return pose_data

def _calculate_pose_quality(landmarks: List[Dict], confidence_threshold: float) -> Dict[str, Any]:
    """Calculate pose quality metrics."""
    visibilities = [lm["visibility"] for lm in landmarks]
    
    # Filter high-confidence landmarks
    high_conf_landmarks = [v for v in visibilities if v >= confidence_threshold]
    
    # Calculate metrics
    average_visibility = np.mean(visibilities)
    minimum_visibility = np.min(visibilities)
    pose_quality_score = len(high_conf_landmarks) / len(landmarks)
    
    # Determine if pose is high quality
    is_high_quality = (
        pose_quality_score >= 0.7 and 
        average_visibility >= 0.6 and 
        minimum_visibility >= 0.3
    )
    
    return {
        "average_visibility": float(average_visibility),
        "minimum_visibility": float(minimum_visibility),
        "pose_quality_score": float(pose_quality_score),
        "is_high_quality": bool(is_high_quality),
        "high_confidence_count": len(high_conf_landmarks),
        "total_landmarks": len(landmarks)
    }

def _choose_optimal_side(landmarks: List[Dict], joint_names: List[str]) -> str:
    """Choose the side with better visibility for analysis."""
    left_visibility = 0
    right_visibility = 0
    count = 0
    
    for joint in joint_names:
        try:
            left_idx = LMS.get(f"LEFT_{joint.upper()}")
            right_idx = LMS.get(f"RIGHT_{joint.upper()}")
            
            if left_idx is not None and right_idx is not None:
                left_visibility += landmarks[left_idx]["visibility"]
                right_visibility += landmarks[right_idx]["visibility"]
                count += 1
        except (IndexError, KeyError):
            continue
    
    if count == 0:
        return "LEFT"  # Default fallback
    
    avg_left = left_visibility / count
    avg_right = right_visibility / count
    
    return "LEFT" if avg_left >= avg_right else "RIGHT"

# --- Coordinate Extraction ---
def _get_xy(pose: Dict[str, Any], side: str, joint: str) -> np.ndarray:
    """Get 2D coordinates for a joint."""
    try:
        idx = LMS[f"{side}_{joint.upper()}"]
        lm = pose["landmarks"][idx]
        return np.array([lm["x"], lm["y"]])
    except (KeyError, IndexError):
        raise ValueError(f"Joint {side}_{joint} not found or invalid")

def _get_enhanced_xy(pose: Dict[str, Any], side: str, joint: str) -> np.ndarray:
    """Enhanced coordinate extraction with special joint handling."""
    # Handle special virtual joints
    if joint == "hip_center":
        left_hip = _get_xy(pose, "LEFT", "hip")
        right_hip = _get_xy(pose, "RIGHT", "hip")
        return (left_hip + right_hip) / 2
    elif joint == "shoulder_center":
        left_shoulder = _get_xy(pose, "LEFT", "shoulder")
        right_shoulder = _get_xy(pose, "RIGHT", "shoulder")
        return (left_shoulder + right_shoulder) / 2
    elif joint == "hip_down":
        # Virtual point below hip for back tilt calculation
        hip = _get_xy(pose, side, "hip")
        return hip + np.array([0, 0.1])  # 10% down from hip
    else:
        return _get_xy(pose, side, joint)

# --- Angle Calculations ---
def compute_angles_config(pose: Dict[str, Any], cfg: Dict[str, Any], use_3d: bool = False) -> Dict[str, float]:
    """
    Compute angles based on configuration with optional enhanced features.
    """
    if not pose:
        return {}
    
    # Choose optimal side
    joint_names = ["shoulder", "hip", "knee", "ankle"]
    side = _choose_optimal_side(pose["landmarks"], joint_names)
    
    angles = {"__side__": side}
    
    for angle_config in cfg.get("angles", []):
        angle_id = angle_config["id"]
        points = angle_config["points"]
        
        if len(points) != 3:
            continue
            
        try:
            # Get coordinates
            if use_3d and SCIPY_AVAILABLE:
                p1 = _get_enhanced_xyz(pose, side, points[0])
                p2 = _get_enhanced_xyz(pose, side, points[1])
                p3 = _get_enhanced_xyz(pose, side, points[2])
                angle_deg = _compute_3d_angle(p1, p2, p3)
            else:
                p1 = _get_enhanced_xy(pose, side, points[0])
                p2 = _get_enhanced_xy(pose, side, points[1])
                p3 = _get_enhanced_xy(pose, side, points[2])
                angle_deg = _compute_2d_angle(p1, p2, p3)
            
            angles[f"{angle_id}_deg"] = float(angle_deg)
            
        except Exception as e:
            print(f"Warning: Could not compute angle {angle_id}: {e}")
            angles[f"{angle_id}_deg"] = 0.0
    
    # Add enhanced biomechanical angles if available
    if SCIPY_AVAILABLE:
        try:
            # Trunk inclination
            shoulder = _get_enhanced_xy(pose, side, "shoulder")
            hip = _get_enhanced_xy(pose, side, "hip")
            vertical = np.array([0, -1])  # Upward vertical
            trunk_vector = shoulder - hip
            trunk_angle = _vector_angle_2d(trunk_vector, vertical)
            angles["trunk_inclination_deg"] = float(trunk_angle)
            
            # Knee alignment (valgus/varus assessment)
            hip_pos = _get_enhanced_xy(pose, side, "hip")
            knee_pos = _get_enhanced_xy(pose, side, "knee")
            ankle_pos = _get_enhanced_xy(pose, side, "ankle")
            
            # Calculate knee alignment relative to vertical line through hip
            hip_ankle_vector = ankle_pos - hip_pos
            hip_knee_vector = knee_pos - hip_pos
            alignment_angle = _vector_angle_2d(hip_knee_vector, hip_ankle_vector)
            angles["knee_alignment_deg"] = float(alignment_angle)
            
        except Exception as e:
            print(f"Warning: Could not compute enhanced angles: {e}")
    
    return angles

def _get_enhanced_xyz(pose: Dict[str, Any], side: str, joint: str) -> np.ndarray:
    """Get 3D coordinates for enhanced calculations."""
    try:
        idx = LMS[f"{side}_{joint.upper()}"]
        lm = pose["landmarks"][idx]
        return np.array([lm["x"], lm["y"], lm["z"]])
    except (KeyError, IndexError):
        raise ValueError(f"Joint {side}_{joint} not found or invalid")

def _compute_2d_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute angle between three 2D points."""
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
    
    v1_unit = v1 / v1_norm
    v2_unit = v2 / v2_norm
    
    # Calculate angle using dot product
    cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def _compute_3d_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute angle between three 3D points."""
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
    
    v1_unit = v1 / v1_norm
    v2_unit = v2 / v2_norm
    
    # Calculate angle using dot product
    cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def _vector_angle_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two 2D vectors."""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
    
    cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)

# --- Legacy Functions for Backward Compatibility ---
def get_enhanced_pose_vector(image_path: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """Legacy function - calls unified get_pose_vector with enhanced=True."""
    return get_pose_vector(image_path, confidence_threshold, enhanced=True)

def compute_enhanced_angles_config(pose: Dict[str, Any], cfg: Dict[str, Any], use_3d: bool = False) -> Dict[str, float]:
    """Legacy function - calls unified compute_angles_config."""
    return compute_angles_config(pose, cfg, use_3d)

# --- Simplified Moment Estimation (Optional) ---
def estimate_moments_config(cfg: Dict[str, Any], angles: Dict[str, float], 
                          body_mass_kg: float, external_load_kg: float) -> Dict[str, float]:
    """
    Basic moment estimation for backward compatibility.
    Note: Moments functionality has been simplified/removed from main analysis.
    """
    moments = {}
    
    # Only calculate if moments are defined in config (for backward compatibility)
    for moment_id, formula in cfg.get("moments", {}).items():
        try:
            # Simple evaluation context
            total_N = (body_mass_kg + external_load_kg) * 9.81
            
            # Replace angle references
            eval_formula = formula
            for angle_key, angle_value in angles.items():
                if angle_key.endswith("_deg"):
                    angle_name = angle_key.replace("_deg", "")
                    eval_formula = eval_formula.replace(f"angles.{angle_name}", str(angle_value))
            
            # Replace functions
            eval_formula = eval_formula.replace("sin_deg", "math.sin(math.radians")
            eval_formula = eval_formula.replace("cos_deg", "math.cos(math.radians")
            eval_formula = eval_formula.replace("total_N", str(total_N))
            
            # Evaluate (basic safety - only allow math operations)
            if all(char in "0123456789+-*/.() math.sincoradnstg" for char in eval_formula.replace(" ", "")):
                result = eval(eval_formula, {"math": math}, {})
                moments[f"{moment_id}_moment_Nm"] = max(0.0, float(result))
            else:
                moments[f"{moment_id}_moment_Nm"] = 0.0
                
        except Exception as e:
            print(f"Warning: Could not calculate moment {moment_id}: {e}")
            moments[f"{moment_id}_moment_Nm"] = 0.0
    
    return moments

def estimate_enhanced_moments_config(cfg: Dict[str, Any], angles: Dict[str, float], 
                                   body_mass_kg: float, external_load_kg: float, 
                                   height_cm: float) -> Dict[str, float]:
    """
    Legacy function - calls basic moment estimation.
    Enhanced moment calculation has been removed from main analysis.
    """
    return estimate_moments_config(cfg, angles, body_mass_kg, external_load_kg)
