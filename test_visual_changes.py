#!/usr/bin/env python3
"""
Test script to verify visual changes are working.
"""

import sys
import os

# Force reload of modules
if 'pose_service.combined_visualization' in sys.modules:
    del sys.modules['pose_service.combined_visualization']

try:
    from pose_service.combined_visualization import create_combined_analysis_image
    from pose_service.engine import get_pose_vector, compute_angles_config
    from utils.exercises import find_config
    
    print("✅ Modules loaded successfully")
    
    # Test if we can create a dummy analysis
    if os.path.exists("squat.jpg"):
        print("🧪 Testing with squat.jpg...")
        
        # Load config
        cfg = find_config(".", "Squat")
        if cfg:
            print("✅ Config loaded")
            
            # Get pose
            pose = get_pose_vector("squat.jpg", enhanced=True)
            print("✅ Pose detected")
            
            # Calculate angles
            angles = compute_angles_config(pose, cfg, use_3d=False)
            print("✅ Angles calculated")
            
            # Calculate simple moments
            import math
            moments = {}
            total_weight_N = 75 * 9.81  # 75kg person
            
            knee_angle = angles.get('knee_deg', 90)
            if knee_angle < 180:
                knee_moment = total_weight_N * 0.25 * math.sin(math.radians(180 - knee_angle))
                moments['knee_moment_Nm'] = max(0, knee_moment)
            
            hip_angle = angles.get('hip_deg', 90)
            if hip_angle < 180:
                hip_moment = total_weight_N * 0.3 * math.sin(math.radians(180 - hip_angle))
                moments['hip_moment_Nm'] = max(0, hip_moment)
            
            print(f"✅ Moments calculated: {moments}")
            
            # Create visualization
            profile_data = {'name': 'Test', 'body_mass_kg': 75, 'height_cm': 175}
            
            output_path = create_combined_analysis_image(
                "squat.jpg", pose, cfg, angles, moments, profile_data, "output"
            )
            
            print(f"🎉 Visualization created: {output_path}")
            print("✅ All visual changes should now be applied!")
            
        else:
            print("❌ No config found for Squat")
    else:
        print("❌ No squat.jpg found for testing")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
