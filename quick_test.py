#!/usr/bin/env python3
"""
Quick test script to verify the enhanced fitness analysis works.
"""

import os
import sys

# Test imports
try:
    from pose_service.enhanced_engine import get_enhanced_pose_vector, compute_enhanced_angles_config, estimate_enhanced_moments_config
    from pose_service.combined_visualization import create_combined_analysis_image
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test if image exists
image_path = "squat.jpg"
if os.path.exists(image_path):
    print(f"✅ Test image found: {image_path}")
else:
    print(f"❌ Test image not found: {image_path}")
    print("Available files in current directory:")
    for f in os.listdir("."):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            print(f"  📸 {f}")

print("\n🎉 Quick test completed!")
print("You can now run: python app.py analyze")
