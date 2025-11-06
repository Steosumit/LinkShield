"""
GPU Setup Verification Script
This script checks if GPU acceleration is properly configured for LinkShield
"""

import sys

print("="*80)
print("GPU SETUP VERIFICATION")
print("="*80)

# Check CUDA availability
print("\n1. Checking CUDA/GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ PyTorch: CUDA available - {torch.cuda.get_device_name(0)}")
        print(f"     CUDA Version: {torch.version.cuda}")
    else:
        print("   ⚠ PyTorch: CUDA not available (install with: pip install torch)")
except ImportError:
    print("   ℹ PyTorch not installed (optional)")

# Check XGBoost
print("\n2. Checking XGBoost...")
try:
    import xgboost as xgb
    print(f"   ✓ XGBoost {xgb.__version__} installed")

    # Test GPU training
    try:
        import numpy as np
        X_test = np.random.rand(100, 10)
        y_test = np.random.randint(0, 2, 100)
        model = xgb.XGBClassifier(device='cuda', tree_method='hist', n_estimators=10)
        model.fit(X_test, y_test)
        print("   ✓ XGBoost GPU training: WORKING")
    except Exception as e:
        print(f"   ⚠ XGBoost GPU training not available: {str(e)}")
        print("     Falling back to CPU mode (will still work in notebook)")
except ImportError:
    print("   ✗ XGBoost not installed")
    print("     Install with: pip install xgboost")

# Check LightGBM
print("\n3. Checking LightGBM...")
try:
    import lightgbm as lgb
    print(f"   ✓ LightGBM {lgb.__version__} installed")

    # Test GPU training
    try:
        import numpy as np
        X_test = np.random.rand(100, 10)
        y_test = np.random.randint(0, 2, 100)
        model = lgb.LGBMClassifier(device='gpu', n_estimators=10)
        model.fit(X_test, y_test)
        print("   ✓ LightGBM GPU training: WORKING")
    except Exception as e:
        print(f"   ⚠ LightGBM GPU training not available: {str(e)}")
        print("     Falling back to CPU mode")
except ImportError:
    print("   ✗ LightGBM not installed")
    print("     Install with: pip install lightgbm")

# Check CuML/RAPIDS
print("\n4. Checking CuML (RAPIDS)...")
try:
    import cuml
    print(f"   ✓ CuML {cuml.__version__} installed")

    # Test GPU training
    try:
        import cupy as cp
        from cuml.ensemble import RandomForestClassifier
        X_test = cp.random.rand(100, 10, dtype=cp.float32)
        y_test = cp.random.randint(0, 2, 100, dtype=cp.int32)
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X_test, y_test)
        print("   ✓ CuML GPU training: WORKING")
    except Exception as e:
        print(f"   ✗ CuML GPU training failed: {str(e)}")
except ImportError:
    print("   ℹ CuML not installed (optional for maximum GPU acceleration)")
    print("     Install with: pip install cuml-cu11 (CUDA 11) or cuml-cu12 (CUDA 12)")

# Check NVIDIA GPU
print("\n5. Checking NVIDIA GPU...")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                           '--format=csv,noheader'],
                          capture_output=True, text=True)
    if result.returncode == 0:
        gpu_info = result.stdout.strip()
        print(f"   ✓ GPU detected: {gpu_info}")
    else:
        print("   ✗ nvidia-smi failed - GPU may not be available")
except FileNotFoundError:
    print("   ✗ nvidia-smi not found - NVIDIA drivers may not be installed")
except Exception as e:
    print(f"   ⚠ Could not check GPU: {str(e)}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

recommendations = []

try:
    import xgboost
    recommendations.append("✓ XGBoost is ready - Use XGBoost_GPU models")
except:
    recommendations.append("✗ Install XGBoost: pip install xgboost")

try:
    import lightgbm
    recommendations.append("✓ LightGBM is ready - Use LightGBM_GPU models")
except:
    recommendations.append("✗ Install LightGBM: pip install lightgbm")

try:
    import cuml
    recommendations.append("✓ CuML is ready - Maximum GPU acceleration available!")
except:
    recommendations.append("ℹ Optional: Install CuML for more GPU models")

if not recommendations:
    print("⚠ No GPU libraries found. Install at least XGBoost for GPU acceleration.")
else:
    for rec in recommendations:
        print(rec)

print("\n" + "="*80)
print("Run the notebook's Manual Approach cell to start GPU-accelerated training!")
print("="*80)

