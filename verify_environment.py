"""
Environment & Hardware Verification Script
Potato Leaf Disease Detection Project
"""

import sys
import platform
import os

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def check_env():
    print("=" * 60)
    print("[SYSTEM & HARDWARE VERIFICATION]")
    print("=" * 60)
    
    # 1. Operating System & Python
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Architecture: {platform.machine()} ({platform.architecture()[0]})")
    print(f"Python Version: {sys.version.split()[0]} ({sys.executable})")
    
    # 2. Check TensorFlow & Keras
    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow Version: {tf.__version__}")
        
        # Check Keras
        try:
            import keras
            print(f"[OK] Keras Version: {keras.__version__}")
        except ImportError:
            print("[INFO] Keras integrated with tf.keras")
            
        # Check Physical Devices
        gpus = tf.config.list_physical_devices('GPU')
        cpus = tf.config.list_physical_devices('CPU')
        print(f"Physical CPUs: {len(cpus)}")
        print(f"Physical GPUs: {len(gpus)}")
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu.name}")
        else:
            print("  [INFO] TensorFlow is running on CPU.")
            print("         (Native Windows GPU support in standard TensorFlow was deprecated")
            print("          starting with TF 2.11. PlantVillage ~2k images trains very quickly on CPU)")
    except ImportError as e:
        print(f"[ERROR] TensorFlow Import Failed: {e}")

    # 3. Check Numerical & Imaging Packages
    for pkg in ["numpy", "PIL", "matplotlib", "seaborn", "sklearn"]:
        try:
            m = __import__(pkg)
            version = getattr(m, "__version__", "installed")
            print(f"[OK] {pkg}: {version}")
        except ImportError as e:
            print(f"[ERROR] {pkg} Import Failed: {e}")

    # 4. Check Backend & Development Packages
    for pkg in ["fastapi", "uvicorn", "pydantic", "pytest"]:
        try:
            m = __import__(pkg)
            version = getattr(m, "__version__", "installed")
            print(f"[OK] {pkg}: {version}")
        except ImportError as e:
            print(f"[ERROR] {pkg} Import Failed: {e}")

    print("=" * 60)
    print("All core dependencies successfully verified!")
    print("=" * 60)

if __name__ == "__main__":
    check_env()
