#!/usr/bin/env python3
"""
Fresh restart script - clears all caches and starts analysis.
"""

import sys
import os
import shutil

print("🔄 Clearing Python caches...")

# Remove all __pycache__ directories
for root, dirs, files in os.walk("."):
    if "__pycache__" in dirs:
        cache_path = os.path.join(root, "__pycache__")
        try:
            shutil.rmtree(cache_path)
            print(f"✅ Removed: {cache_path}")
        except:
            pass

# Clear sys.modules of our custom modules
modules_to_clear = []
for module_name in sys.modules.keys():
    if any(x in module_name for x in ['pose_service', 'utils']):
        modules_to_clear.append(module_name)

for module_name in modules_to_clear:
    del sys.modules[module_name]
    print(f"✅ Cleared module: {module_name}")

print("🚀 All caches cleared! Now run:")
print("   python app.py analyze")
print("\nThe visual changes should now be applied!")
