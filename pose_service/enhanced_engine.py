"""
Enhanced pose analysis engine with improved accuracy and additional biomechanical calculations.
"""

import math
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
from scipy.spatial.distance import euclidean
from scipy import signal

def _load_mediapipe():
    """Load MediaPipe with proper error handling."""
    try:
        import mediapipe as mp
        return mp
    except Exception as e:
        raise RuntimeError("MediaPipe ist nicht installiert/kompatibel (nutze Python 3.11).") from e

def get_enhanced_pose_vector(image_path: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Enhanced pose detection with confidence filtering and quality metrics.
    """
    mp = _load_mediapipe()
    mp_pose = mp.solutions.pose
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Enhanced pose detection with better parameters
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,  # Higher accuracy
        enable_segmentation=False,
        min_detection_confidence=confidence_threshold,
        min_tracking_confidence=confidence_threshold
    ) as pose:
        results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        raise RuntimeError("Keine Pose erkannt. Versuche bessere Beleuchtung oder andere Kameraposition.")
    
    # Extract landmarks with enhanced metadata
    landmarks = []
    visibility_scores = []
    
    for i, lm in enumerate(results.pose_landmarks.landmark):
        landmark_data = {
            "x": lm.x, 
            "y": lm.y, 
            "z": lm.z, 
            "visibility": lm.visibility,
            "index": i
        }
        landmarks.append(landmark_data)
        visibility_scores.append(lm.visibility)
    
    # Calculate pose quality metrics
    avg_visibility = np.mean(visibility_scores)
    min_visibility = np.min(visibility_scores)
    pose_quality = _assess_pose_quality(landmarks, w, h)
    
    return {
        "image_size": {"width": w, "height": h},
        "landmarks": landmarks,
        "quality_metrics": {
            "average_visibility": round(avg_visibility, 3),
            "minimum_visibility": round(min_visibility, 3),
            "pose_quality_score": round(pose_quality, 3),
            "is_high_quality": pose_quality > 0.7 and avg_visibility > 0.6
        }
    }

# Enhanced MediaPipe landmark indices with additional points
LMS = {
    "NOSE": 0,
    "LEFT_EYE_INNER": 1, "LEFT_EYE": 2, "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4, "RIGHT_EYE": 5, "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7, "RIGHT_EAR": 8,
    "MOUTH_LEFT": 9, "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17, "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19, "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21, "RIGHT_THUMB": 22,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29, "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32
}

def _assess_pose_quality(landmarks: List[Dict], width: int, height: int) -> float:
    """
    Assess the quality of the detected pose based on various factors.
    Returns a score between 0 and 1.
    """
    quality_factors = []
    
    # Factor 1: Visibility of key joints
    key_joints = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP", 
                  "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
    
    key_visibilities = []
    for joint in key_joints:
        if joint in LMS:
            visibility = landmarks[LMS[joint]]["visibility"]
            key_visibilities.append(visibility)
    
    if key_visibilities:
        quality_factors.append(np.mean(key_visibilities))
    
    # Factor 2: Pose completeness (body parts present)
    body_parts_present = 0
    total_body_parts = 4  # head, torso, arms, legs
    
    # Check head
    if landmarks[LMS["NOSE"]]["visibility"] > 0.5:
        body_parts_present += 1
    
    # Check torso
    torso_vis = np.mean([landmarks[LMS["LEFT_SHOULDER"]]["visibility"],
                        landmarks[LMS["RIGHT_SHOULDER"]]["visibility"],
                        landmarks[LMS["LEFT_HIP"]]["visibility"],
                        landmarks[LMS["RIGHT_HIP"]]["visibility"]])
    if torso_vis > 0.5:
        body_parts_present += 1
    
    # Check arms
    arm_vis = np.mean([landmarks[LMS["LEFT_ELBOW"]]["visibility"],
                      landmarks[LMS["RIGHT_ELBOW"]]["visibility"],
                      landmarks[LMS["LEFT_WRIST"]]["visibility"],
                      landmarks[LMS["RIGHT_WRIST"]]["visibility"]])
    if arm_vis > 0.4:
        body_parts_present += 1
    
    # Check legs
    leg_vis = np.mean([landmarks[LMS["LEFT_KNEE"]]["visibility"],
                      landmarks[LMS["RIGHT_KNEE"]]["visibility"],
                      landmarks[LMS["LEFT_ANKLE"]]["visibility"],
                      landmarks[LMS["RIGHT_ANKLE"]]["visibility"]])
    if leg_vis > 0.4:
        body_parts_present += 1
    
    quality_factors.append(body_parts_present / total_body_parts)
    
    # Factor 3: Pose stability (check for reasonable proportions)
    try:
        shoulder_width = _calculate_distance(landmarks, "LEFT_SHOULDER", "RIGHT_SHOULDER", width, height)
        hip_width = _calculate_distance(landmarks, "LEFT_HIP", "RIGHT_HIP", width, height)
        torso_height = _calculate_distance(landmarks, "LEFT_SHOULDER", "LEFT_HIP", width, height)
        
        # Reasonable proportions check
        if 0.3 < hip_width/shoulder_width < 1.5 and torso_height > shoulder_width * 0.5:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.5)
    except:
        quality_factors.append(0.3)
    
    return np.mean(quality_factors) if quality_factors else 0.0

def _calculate_distance(landmarks: List[Dict], point1: str, point2: str, 
                       width: int, height: int) -> float:
    """Calculate Euclidean distance between two landmarks in pixels."""
    lm1 = landmarks[LMS[point1]]
    lm2 = landmarks[LMS[point2]]
    
    x1, y1 = lm1["x"] * width, lm1["y"] * height
    x2, y2 = lm2["x"] * width, lm2["y"] * height
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def _enhanced_angle_calculation(a: np.ndarray, b: np.ndarray, c: np.ndarray, 
                               use_3d: bool = False) -> float:
    """
    Enhanced angle calculation with 3D support and numerical stability.
    """
    if use_3d and len(a) == 3 and len(b) == 3 and len(c) == 3:
        # 3D angle calculation
        ba = a - b
        bc = c - b
        
        # Numerical stability improvements
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba < 1e-9 or norm_bc < 1e-9:
            return 0.0
        
        cosang = np.dot(ba, bc) / (norm_ba * norm_bc)
    else:
        # 2D angle calculation (default)
        ba = a[:2] - b[:2]
        bc = c[:2] - b[:2]
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba < 1e-9 or norm_bc < 1e-9:
            return 0.0
        
        cosang = np.dot(ba, bc) / (norm_ba * norm_bc)
    
    # Clamp to valid range and calculate angle
    cosang = np.clip(cosang, -1.0, 1.0)
    angle_rad = math.acos(cosang)
    return math.degrees(angle_rad)

def _choose_optimal_side(lms: List[Dict], points: List[str]) -> str:
    """
    Enhanced side selection based on visibility, completeness, and quality.
    """
    name_to_key = {
        "shoulder": ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
        "elbow": ("LEFT_ELBOW", "RIGHT_ELBOW"),
        "wrist": ("LEFT_WRIST", "RIGHT_WRIST"),
        "hip": ("LEFT_HIP", "RIGHT_HIP"),
        "knee": ("LEFT_KNEE", "RIGHT_KNEE"),
        "ankle": ("LEFT_ANKLE", "RIGHT_ANKLE"),
    }
    
    left_score = 0.0
    right_score = 0.0
    
    for point in points:
        base = point.split("_")[0]
        if base in name_to_key:
            left_key, right_key = name_to_key[base]
            
            if left_key in LMS and right_key in LMS:
                left_vis = lms[LMS[left_key]]["visibility"]
                right_vis = lms[LMS[right_key]]["visibility"]
                
                # Weight by visibility and add bonus for high visibility
                left_score += left_vis + (0.2 if left_vis > 0.8 else 0)
                right_score += right_vis + (0.2 if right_vis > 0.8 else 0)
    
    return "LEFT" if left_score >= right_score else "RIGHT"

def _get_enhanced_xy(pose: Dict, side: str, name: str, use_3d: bool = False) -> np.ndarray:
    """
    Enhanced coordinate extraction with 3D support and special point handling.
    """
    w = pose["image_size"]["width"]
    h = pose["image_size"]["height"]
    lms = pose["landmarks"]
    
    # Handle special points
    if name == "hip_down":
        i = LMS[f"{side}_HIP"]
        lm = lms[i]
        if use_3d:
            return np.array([lm["x"] * w, lm["y"] * h + 1.0, lm["z"]], dtype=float)
        else:
            return np.array([lm["x"] * w, lm["y"] * h + 1.0], dtype=float)
    
    elif name == "hip_center":
        # Calculate center point between hips
        left_hip = lms[LMS["LEFT_HIP"]]
        right_hip = lms[LMS["RIGHT_HIP"]]
        center_x = (left_hip["x"] + right_hip["x"]) / 2 * w
        center_y = (left_hip["y"] + right_hip["y"]) / 2 * h
        if use_3d:
            center_z = (left_hip["z"] + right_hip["z"]) / 2
            return np.array([center_x, center_y, center_z], dtype=float)
        else:
            return np.array([center_x, center_y], dtype=float)
    
    elif name == "shoulder_center":
        # Calculate center point between shoulders
        left_shoulder = lms[LMS["LEFT_SHOULDER"]]
        right_shoulder = lms[LMS["RIGHT_SHOULDER"]]
        center_x = (left_shoulder["x"] + right_shoulder["x"]) / 2 * w
        center_y = (left_shoulder["y"] + right_shoulder["y"]) / 2 * h
        if use_3d:
            center_z = (left_shoulder["z"] + right_shoulder["z"]) / 2
            return np.array([center_x, center_y, center_z], dtype=float)
        else:
            return np.array([center_x, center_y], dtype=float)
    
    # Standard joint mapping
    key_mapping = {
        "shoulder": "SHOULDER", "elbow": "ELBOW", "wrist": "WRIST",
        "hip": "HIP", "knee": "KNEE", "ankle": "ANKLE",
        "heel": "HEEL", "foot_index": "FOOT_INDEX"
    }
    
    key = key_mapping.get(name)
    if not key:
        raise ValueError(f"Unknown point '{name}'")
    
    landmark_key = f"{side}_{key}"
    if landmark_key not in LMS:
        raise ValueError(f"Landmark '{landmark_key}' not found")
    
    i = LMS[landmark_key]
    lm = lms[i]
    
    if use_3d:
        return np.array([lm["x"] * w, lm["y"] * h, lm["z"]], dtype=float)
    else:
        return np.array([lm["x"] * w, lm["y"] * h], dtype=float)

def compute_enhanced_angles_config(pose: Dict, cfg: Dict, use_3d: bool = False) -> Dict[str, float]:
    """
    Enhanced angle computation with 3D support and additional biomechanical angles.
    """
    # Collect all needed points
    needed_points = []
    for seg in cfg.get("segments", []):
        needed_points.extend(seg)
    for angle_cfg in cfg.get("angles", []):
        needed_points.extend(angle_cfg.get("points", []))
    
    # Choose optimal side
    side = _choose_optimal_side(pose["landmarks"], needed_points)
    
    result = {"__side__": side}
    
    # Calculate configured angles
    for angle_cfg in cfg.get("angles", []):
        angle_id = angle_cfg["id"]
        points = angle_cfg["points"]
        
        if len(points) >= 3:
            try:
                A = _get_enhanced_xy(pose, side, points[0], use_3d)
                B = _get_enhanced_xy(pose, side, points[1], use_3d)
                C = _get_enhanced_xy(pose, side, points[2], use_3d)
                
                angle_value = _enhanced_angle_calculation(A, B, C, use_3d)
                result[f"{angle_id}_deg"] = round(float(angle_value), 2)
                
            except Exception as e:
                print(f"Warning: Could not calculate angle {angle_id}: {e}")
                result[f"{angle_id}_deg"] = 0.0
    
    # Add additional biomechanical angles
    result.update(_calculate_additional_angles(pose, side, use_3d))
    
    return result

def _calculate_additional_angles(pose: Dict, side: str, use_3d: bool = False) -> Dict[str, float]:
    """
    Calculate additional biomechanically relevant angles.
    """
    additional_angles = {}
    
    try:
        # Trunk inclination (forward lean)
        shoulder = _get_enhanced_xy(pose, side, "shoulder", use_3d)
        hip = _get_enhanced_xy(pose, side, "hip", use_3d)
        
        # Create vertical reference
        vertical_ref = hip + np.array([0, -100] + ([0] if use_3d else []))
        trunk_angle = _enhanced_angle_calculation(vertical_ref, hip, shoulder, use_3d)
        additional_angles["trunk_inclination_deg"] = round(trunk_angle, 2)
        
        # Knee valgus/varus (frontal plane deviation)
        if not use_3d:  # Only meaningful in 2D frontal view
            try:
                knee = _get_enhanced_xy(pose, side, "knee", use_3d)
                ankle = _get_enhanced_xy(pose, side, "ankle", use_3d)
                
                # Calculate knee alignment
                vertical_line = knee + np.array([0, 100])
                knee_alignment = _enhanced_angle_calculation(hip, knee, vertical_line, use_3d)
                additional_angles["knee_alignment_deg"] = round(knee_alignment, 2)
            except:
                pass
        
        # Ankle dorsiflexion
        try:
            ankle = _get_enhanced_xy(pose, side, "ankle", use_3d)
            knee = _get_enhanced_xy(pose, side, "knee", use_3d)
            
            # Approximate foot direction (using heel if available)
            try:
                heel = _get_enhanced_xy(pose, side, "heel", use_3d)
                foot_ref = heel
            except:
                # Fallback: create horizontal reference
                foot_ref = ankle + np.array([50, 0] + ([0] if use_3d else []))
            
            ankle_angle = _enhanced_angle_calculation(knee, ankle, foot_ref, use_3d)
            additional_angles["ankle_dorsiflexion_deg"] = round(ankle_angle, 2)
        except:
            pass
        
    except Exception as e:
        print(f"Warning: Could not calculate additional angles: {e}")
    
    return additional_angles

def estimate_enhanced_moments_config(cfg: Dict, angles_deg: Dict[str, float], 
                                   body_mass_kg: float, external_load_kg: float,
                                   height_cm: float = 175) -> Dict[str, float]:
    """
    Enhanced moment estimation with anthropometric scaling and improved biomechanics.
    """
    import math as _m
    
    def sin_deg(x): return _m.sin(_m.radians(x))
    def cos_deg(x): return _m.cos(_m.radians(x))
    
    g = 9.81
    
    # Anthropometric scaling based on height
    height_m = height_cm / 100.0
    scale_factor = height_m / 1.75  # Normalize to 175cm reference
    
    # Enhanced environment with anthropometric data
    env = {
        "g": g,
        "total_N": (body_mass_kg + external_load_kg) * g,
        "body_weight_N": body_mass_kg * g,
        "external_load_N": external_load_kg * g,
        "per_arm_N": (external_load_kg / 2.0) * g,
        "height_m": height_m,
        "scale_factor": scale_factor,
        "sin_deg": sin_deg, 
        "cos_deg": cos_deg,
        "angles": {k.replace("_deg", ""): v for k, v in angles_deg.items() if k.endswith("_deg")},
        # Anthropometric segment lengths (scaled)
        "thigh_length": 0.245 * height_m,
        "shank_length": 0.246 * height_m,
        "torso_length": 0.288 * height_m,
        "upper_arm_length": 0.186 * height_m,
        "forearm_length": 0.146 * height_m,
        # Segment mass percentages
        "thigh_mass_pct": 0.100,
        "shank_mass_pct": 0.0465,
        "torso_mass_pct": 0.497,
        "upper_arm_mass_pct": 0.028,
        "forearm_mass_pct": 0.016
    }
    
    result = {}
    
    # Calculate configured moments
    moments_config = cfg.get("moments", {})
    for joint, expression in moments_config.items():
        try:
            moment_value = eval(expression, {"__builtins__": {}}, env)
            result[f"{joint}_moment_Nm"] = round(float(moment_value), 2)
        except Exception as e:
            print(f"Warning: Could not calculate moment for {joint}: {e}")
            result[f"{joint}_moment_Nm"] = 0.0
    
    # Add enhanced moment calculations if not configured
    if "knee_moment_Nm" not in result:
        result.update(_calculate_enhanced_knee_moment(env))
    
    if "hip_moment_Nm" not in result:
        result.update(_calculate_enhanced_hip_moment(env))
    
    return result

def _calculate_enhanced_knee_moment(env: Dict) -> Dict[str, float]:
    """Calculate enhanced knee moment with biomechanical considerations."""
    try:
        knee_angle = env["angles"].get("knee", 180)  # Default to straight
        
        # Enhanced knee moment calculation
        # Consider both gravitational and inertial effects
        thigh_mass = env["body_weight_N"] / env["g"] * env["thigh_mass_pct"]
        shank_mass = env["body_weight_N"] / env["g"] * env["shank_mass_pct"]
        
        # Moment arms (simplified but more accurate)
        knee_flexion = 180 - knee_angle
        moment_arm_factor = env["sin_deg"](knee_flexion) * env["scale_factor"]
        
        # Gravitational moment
        gravity_moment = (thigh_mass * env["thigh_length"] * 0.5 + 
                         shank_mass * env["shank_length"] * 0.5) * env["g"] * moment_arm_factor
        
        # External load contribution
        external_moment = env["external_load_N"] * env["thigh_length"] * 0.6 * moment_arm_factor
        
        total_moment = gravity_moment + external_moment
        
        return {"knee_moment_Nm": round(total_moment, 2)}
    
    except Exception:
        return {"knee_moment_Nm": 0.0}

def _calculate_enhanced_hip_moment(env: Dict) -> Dict[str, float]:
    """Calculate enhanced hip moment with biomechanical considerations."""
    try:
        hip_angle = env["angles"].get("hip", 180)
        trunk_inclination = env["angles"].get("trunk_inclination", 0)
        
        # Enhanced hip moment calculation
        torso_mass = env["body_weight_N"] / env["g"] * env["torso_mass_pct"]
        
        # Consider trunk inclination in moment calculation
        inclination_factor = env["sin_deg"](trunk_inclination)
        hip_flexion_factor = env["sin_deg"](180 - hip_angle)
        
        # Gravitational moment from trunk
        trunk_moment = torso_mass * env["torso_length"] * 0.5 * env["g"] * inclination_factor
        
        # Hip flexion moment
        hip_flexion_moment = env["total_N"] * env["thigh_length"] * 0.4 * hip_flexion_factor
        
        total_moment = trunk_moment + hip_flexion_moment
        
        return {"hip_moment_Nm": round(total_moment, 2)}
    
    except Exception:
        return {"hip_moment_Nm": 0.0}

# Maintain backward compatibility
get_pose_vector = get_enhanced_pose_vector
compute_angles_config = compute_enhanced_angles_config
estimate_moments_config = estimate_enhanced_moments_config
_get_xy = _get_enhanced_xy
