# Quick GPU Monitoring Guide

## To See GPU Usage During Training

### Open a NEW terminal and run:
```cmd
nvidia-smi -l 1
```

This refreshes every second. You should see:

```
+-----------------------------------------------------------------------------+
| GPU  Name            | GPU-Util | Memory-Usage |
|======================|==========|==============|
|   0  GTX 1050 Ti     |   75%    |  2500/4096MB |  👈 Should increase during training
+-----------------------------------------------------------------------------+
```

## What to Look For:

✅ **GPU-Util**: Should be >30% during LightGBM/XGBoost GPU training
✅ **Memory-Usage**: Should increase from baseline
✅ **Processes**: Should show "python.exe" using GPU

## In Notebook Output, Look For:

```
Training: LightGBM_GPU
[LightGBM] [Info] Using GPU Device: NVIDIA GeForce GTX 1050 Ti  👈 Confirms GPU usage
...
Time: 8.2s (GPU)  👈 Should be faster than CPU version
```

## Quick Comparison:

```
LightGBM_GPU:  8.2s  (GPU) 👈 Fast
LightGBM_CPU: 25.7s  (CPU) 👈 Slower = GPU is working!
```

## If GPU-Util Stays at 0%:

1. Check if XGBoost/LightGBM installed: `pip show xgboost lightgbm`
2. Look for "Using GPU Device" in training output
3. Make sure you're training models with "_GPU" suffix
4. Try increasing dataset size (GPU works better with more data)

## Expected GPU Utilization on GTX 1050 Ti:

- **LightGBM_GPU**: 60-90% GPU utilization ✅
- **XGBoost_GPU**: 50-80% GPU utilization ✅
- **CPU models**: 0% GPU (expected)

## Commands to Run:

```cmd
# Verify GPU setup
python D:\Work\Projects\LinkShield\check_gpu_setup.py

# Monitor GPU (in separate terminal)
nvidia-smi -l 1

# Then run your notebook cells!
```

**Your system is ready for GPU training!** 🚀

